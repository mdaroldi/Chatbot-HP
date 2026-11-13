"""Retention on the real Postgres: the purge, and the advisory lock between holders."""

import io
import pathlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine

from app import retention
from app.domain.retention import PurgeCounts
from app.infra.chat_store import SqlChatStore
from app.infra.db import create_engine, create_session_factory
from app.infra.orm import ConversationRow, MessageRow
from app.infra.retention import RETENTION_LOCK_KEY, PostgresPurgeLock, SqlExpiredChats
from tests.chat_rows import CLIENT, OTHER_CLIENT
from tests.integration.conftest import SettingsFactory, run_alembic

pytestmark = pytest.mark.integration

# The acceptance check's SQL (criterion 11), with the id bound instead of pasted in.
AGE_CONVERSATION = text(
    "UPDATE conversations SET created_at = now() - interval '31 days', "
    "updated_at = now() - interval '31 days' WHERE id = :id"
)


@pytest.fixture
async def engine(
    scratch_db: str, make_settings: SettingsFactory, tmp_path: pathlib.Path
) -> AsyncIterator[AsyncEngine]:
    run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path)
    engine = create_engine(make_settings(postgres_db=scratch_db))
    yield engine
    await engine.dispose()


async def conversation_with_turn(engine: AsyncEngine, client_id: str = CLIENT) -> str:
    store = SqlChatStore(create_session_factory(engine), abandon_after=timedelta(minutes=3))
    created = await store.create_conversation(client_id)
    await store.start_turn(client_id, created.id, "Synthetic question?", title="Synthetic")
    return created.id


async def age(engine: AsyncEngine, conversation_id: str) -> None:
    async with engine.begin() as connection:
        await connection.execute(AGE_CONVERSATION, {"id": conversation_id})


async def remaining(engine: AsyncEngine) -> tuple[set[str], int]:
    async with engine.connect() as connection:
        ids = set((await connection.execute(select(ConversationRow.id))).scalars())
        messages = (
            await connection.execute(select(func.count()).select_from(MessageRow))
        ).scalar_one()
    return ids, messages


async def test_an_old_conversation_is_deleted_with_its_messages_and_a_newer_one_kept(
    engine: AsyncEngine,
) -> None:
    old = await conversation_with_turn(engine)
    theirs = await conversation_with_turn(engine, OTHER_CLIENT)
    newer = await conversation_with_turn(engine)
    await age(engine, old)
    await age(engine, theirs)

    cutoff = datetime.now(UTC) - timedelta(days=30)
    counts = await SqlExpiredChats(create_session_factory(engine), batch_size=1).purge(cutoff)

    assert counts == PurgeCounts(conversations=2, messages=4)
    assert await remaining(engine) == ({newer}, 2)


async def test_only_one_holder_gets_the_advisory_lock_until_it_is_released(
    engine: AsyncEngine,
) -> None:
    first, second = PostgresPurgeLock(engine), PostgresPurgeLock(engine)

    async with first.hold() as held_first, second.hold() as held_second:
        assert (held_first, held_second) == (True, False)

    async with second.hold() as held_again:
        assert held_again is True
    async with engine.connect() as connection:
        held_now = await connection.execute(
            text("SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND objid = :low"),
            {"low": RETENTION_LOCK_KEY & 0xFFFFFFFF},
        )
        assert held_now.scalar_one() == 0


async def test_the_command_purges_and_skips_while_another_session_holds_the_lock(
    engine: AsyncEngine, scratch_db: str, make_settings: SettingsFactory
) -> None:
    old = await conversation_with_turn(engine)
    await age(engine, old)
    settings = make_settings(postgres_db=scratch_db)

    async with engine.connect() as other_session:
        await other_session.execute(select(func.pg_advisory_lock(RETENTION_LOCK_KEY)))
        busy_out = io.StringIO()
        assert await retention.run(settings, out=busy_out) == 1
        await other_session.execute(select(func.pg_advisory_unlock(RETENTION_LOCK_KEY)))
    assert "another purge holds the lock" in busy_out.getvalue()
    assert old in (await remaining(engine))[0]

    out = io.StringIO()
    assert await retention.run(settings, out=out) == 0
    assert out.getvalue() == "purged conversations=1 messages=2 retention_days=30\n"
    assert await remaining(engine) == (set(), 0)

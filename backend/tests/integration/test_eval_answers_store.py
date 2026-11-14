"""The answer benchmark's store reader on the real Postgres: rows read back, writes refused."""

import pathlib
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from sqlalchemy import update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.domain.chat import AnswerRecord, Outcome
from app.evaluation.answers.store import SqlAnswerRows, read_only_session
from app.infra.chat_store import SqlChatStore
from app.infra.db import create_engine, create_session_factory
from app.infra.orm import ConversationRow
from tests.chat_rows import CLIENT
from tests.integration.conftest import SettingsFactory, run_alembic

pytestmark = pytest.mark.integration


@pytest.fixture
async def engine(
    scratch_db: str, make_settings: SettingsFactory, tmp_path: pathlib.Path
) -> AsyncIterator[AsyncEngine]:
    run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path)
    engine = create_engine(make_settings(postgres_db=scratch_db))
    yield engine
    await engine.dispose()


@pytest.fixture
def sessions(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return create_session_factory(engine)


async def test_rows_are_read_back_and_a_write_in_the_read_session_is_refused(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    chat = SqlChatStore(sessions, abandon_after=timedelta(minutes=3))
    conversation = (await chat.create_conversation(CLIENT)).id
    started = await chat.start_turn(CLIENT, conversation, "Synthetic question?", title=None)
    record = AnswerRecord(
        outcome=Outcome.COMPLETED,
        content="Synthetic answer.",
        chunk_ids=("chunk-2", "chunk-1"),
        model="synthetic-model",
        input_tokens=4100,
        output_tokens=310,
    )
    await chat.finish_answer(CLIENT, started.assistant.id, record)

    (row,) = await SqlAnswerRows(sessions).answers(conversation, CLIENT)
    assert (row.message_id, row.chunk_ids, row.input_tokens) == (
        started.assistant.id,
        ("chunk-2", "chunk-1"),
        4100,
    )

    with pytest.raises(DBAPIError, match="read-only transaction"):
        async with read_only_session(sessions) as session:
            await session.execute(
                update(ConversationRow)
                .where(ConversationRow.id == conversation)
                .values(title="changed")
            )

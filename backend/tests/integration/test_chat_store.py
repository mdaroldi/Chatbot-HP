"""SqlChatStore on the real Postgres: ownership, the one-pending rule under concurrency."""

import asyncio
import pathlib
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from app.domain.chat import AnswerRecord, Outcome
from app.domain.chat_store import ConversationBusyError, ConversationNotFoundError, Turn
from app.infra.chat_store import SqlChatStore
from app.infra.db import create_engine, create_session_factory
from tests.chat_rows import CLIENT, OTHER_CLIENT
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
def store(engine: AsyncEngine) -> SqlChatStore:
    return SqlChatStore(create_session_factory(engine), abandon_after=timedelta(minutes=3))


async def test_a_conversation_round_trip(store: SqlChatStore) -> None:
    created = await store.create_conversation(CLIENT)
    turn = await store.start_turn(CLIENT, created.id, "Synthetic question?", title="Synthetic")
    answer = await store.finish_answer(
        CLIENT,
        turn.assistant.id,
        AnswerRecord(outcome=Outcome.COMPLETED, content="Synthetic answer.", chunk_ids=("c1",)),
    )

    thread = await store.get_thread(CLIENT, created.id)

    assert thread is not None
    assert thread.messages == (turn.user, answer)
    assert thread.conversation.title == "Synthetic"
    assert thread.conversation.created_at.tzinfo is not None
    page = await store.list_conversations(CLIENT, limit=5)
    assert [c.id for c in page.items] == [created.id]


async def test_one_client_never_reaches_another_clients_conversation(store: SqlChatStore) -> None:
    created = await store.create_conversation(CLIENT)
    turn = await store.start_turn(CLIENT, created.id, "Question?", title="Question")

    assert await store.get_thread(OTHER_CLIENT, created.id) is None
    assert (await store.list_conversations(OTHER_CLIENT, limit=5)).items == ()
    with pytest.raises(ConversationNotFoundError):
        await store.start_turn(OTHER_CLIENT, created.id, "Mine?", title="Mine")
    assert (
        await store.finish_answer(
            OTHER_CLIENT, turn.assistant.id, AnswerRecord(outcome=Outcome.FAILED, content="")
        )
        is None
    )
    await store.discard_turn(OTHER_CLIENT, turn)
    assert await store.delete_conversation(OTHER_CLIENT, created.id) is False

    thread = await store.get_thread(CLIENT, created.id)
    assert thread is not None
    assert thread.messages == (turn.user, turn.assistant)


async def test_concurrent_turns_on_one_conversation_let_exactly_one_through(
    store: SqlChatStore,
) -> None:
    created = await store.create_conversation(CLIENT)

    results = await asyncio.gather(
        *(store.start_turn(CLIENT, created.id, f"Q{i}?", title=f"Q{i}") for i in range(5)),
        return_exceptions=True,
    )

    turns = [r for r in results if isinstance(r, Turn)]
    assert len(turns) == 1
    assert all(isinstance(r, ConversationBusyError) for r in results if not isinstance(r, Turn))
    thread = await store.get_thread(CLIENT, created.id)
    assert thread is not None
    assert [m.seq for m in thread.messages] == [1, 2]


async def test_deleting_a_conversation_removes_its_messages(store: SqlChatStore) -> None:
    created = await store.create_conversation(CLIENT)
    await store.start_turn(CLIENT, created.id, "Question?", title="Question")

    assert await store.delete_conversation(CLIENT, created.id) is True

    assert await store.get_thread(CLIENT, created.id) is None

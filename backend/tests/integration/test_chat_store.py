"""SqlChatStore on the real Postgres: ownership, the one-pending rule under concurrency."""

import asyncio
import pathlib
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.domain.chat import AnswerRecord, Outcome
from app.domain.chat_store import ConversationBusyError, ConversationNotFoundError, Turn
from app.infra.chat_store import SqlChatStore, is_pending_answer_conflict
from app.infra.db import create_engine, create_session_factory
from tests.chat_rows import (
    CLIENT,
    CONFLICT_CASES,
    OTHER_CLIENT,
    PENDING_ANSWER,
    Row,
    conversation_row,
    insert_conversation,
    insert_message,
)
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


async def test_a_check_violation_during_a_turn_is_not_reported_as_busy(store: SqlChatStore) -> None:
    created = await store.create_conversation(CLIENT)

    with pytest.raises(IntegrityError, match="ck_conversations_title_not_empty"):
        await store.start_turn(CLIENT, created.id, "Question?", title="")


@pytest.mark.parametrize(("row", "pending"), CONFLICT_CASES)
async def test_only_the_pending_answer_index_reads_as_a_pending_conflict(
    engine: AsyncEngine, row: Row, pending: bool
) -> None:
    """Real Postgres errors: matched on the constraint name the server reports."""
    async with engine.begin() as connection:
        await insert_conversation(connection, conversation_row())
        await insert_message(connection, PENDING_ANSWER)
    with pytest.raises(IntegrityError) as raised:
        async with engine.begin() as connection:
            await insert_message(connection, row)

    assert is_pending_answer_conflict(raised.value) is pending


async def test_a_thread_read_mid_answer_shows_the_pending_answer(store: SqlChatStore) -> None:
    """A client reloading while an answer streams sees it pending, empty, uncited."""
    created = await store.create_conversation(CLIENT)
    turn = await store.start_turn(CLIENT, created.id, "Synthetic question?", title="Synthetic")

    thread = await store.get_thread(CLIENT, created.id)

    assert thread is not None
    (_, pending) = thread.messages
    assert pending.id == turn.assistant.id
    assert (pending.outcome, pending.content, pending.refusal) == (Outcome.PENDING, "", False)
    assert (pending.citations, pending.sources) == ((), ())

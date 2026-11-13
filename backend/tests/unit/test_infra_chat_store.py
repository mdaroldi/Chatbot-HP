"""SqlChatStore on SQLite (aiosqlite); the Postgres integration tests repeat the core cases.

Tables come from the ORM metadata here; migration 0005 builds the same tables on
Postgres, and `alembic check` proves the two agree.
"""

import pathlib
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import event, select
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.middleware import RequestIdMiddleware
from app.domain.chat import (
    AnswerFlags,
    AnswerRecord,
    Outcome,
    RefusalReason,
    Role,
    Source,
)
from app.domain.chat_store import (
    ChatStore,
    ConversationBusyError,
    ConversationCursor,
    ConversationNotFoundError,
)
from app.domain.ingestion import PageRange
from app.domain.retrieval import Relevance, StageTimings
from app.infra.chat_store import SqlChatStore, is_pending_answer_conflict
from app.infra.db import create_session_factory
from app.infra.orm import Base, MessageRow
from tests.chat_rows import (
    CLIENT,
    CONFLICT_CASES,
    CONVERSATION_VIOLATIONS,
    MESSAGE_VIOLATIONS,
    OTHER_CLIENT,
    PENDING_ANSWER,
    T0,
    VALID_MESSAGES,
    Row,
    conversation_row,
    insert_conversation,
    insert_message,
)
from tests.unit.conftest import LogReader

ABANDON_AFTER = timedelta(minutes=3)
MISSING_ID = "00000000-0000-4000-8000-00000000dead"
SOURCE = Source(
    tag="S1",
    doc_id="synthetic-doc",
    document_title="Synthetic manual",
    section_path=("Chapter", "Section"),
    pages=PageRange(3, 4, 1, 2),
    has_warning=True,
    snippet="A short synthetic snippet.",
)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


@pytest.fixture
async def engine(tmp_path: pathlib.Path) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'chat.db'}")

    @event.listens_for(engine.sync_engine, "connect")
    def enable_foreign_keys(dbapi_connection: Any, _: object) -> None:  # Any: DBAPI connection
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def sessions(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return create_session_factory(engine)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(sessions: async_sessionmaker[AsyncSession], clock: Clock) -> SqlChatStore:
    return SqlChatStore(sessions, abandon_after=ABANDON_AFTER, clock=clock)


def test_the_sql_store_implements_the_port(store: SqlChatStore) -> None:
    port: ChatStore = store  # mypy checks the structural match
    assert port is store


def test_the_store_reports_its_abandon_window(store: SqlChatStore) -> None:
    assert store.abandon_after == ABANDON_AFTER


def completed(**overrides: object) -> AnswerRecord:
    fields: dict[str, object] = {
        "outcome": Outcome.COMPLETED,
        "content": "Synthetic answer [S1].",
        "citations": ("S1",),
        "sources": (SOURCE,),
        "chunk_ids": ("chunk-1", "chunk-2"),
        "model": "stub",
        "input_tokens": 120,
        "output_tokens": 8,
        "timings": StageTimings(1.5, 0.5, 3.0, 40.0, 45.0),
        "rewrite_ms": 12.0,
        "ttft_ms": 60.0,
        "generation_ms": 80.0,
        "total_ms": 150.0,
        "relevance": Relevance.from_top_score(-1.25, threshold=-6.54),
        "index_build_id": "20261113t173108z",
        "rewritten_query": "synthetic standalone query",
        "flags": AnswerFlags(invalid_citations=1),
    }
    fields.update(overrides)
    return AnswerRecord(**fields)  # type: ignore[arg-type]  # fields typed per test


# Conversations


async def test_without_a_clock_the_store_stamps_the_current_utc_time(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    store = SqlChatStore(sessions, abandon_after=ABANDON_AFTER)
    before = datetime.now(UTC)

    created = await store.create_conversation(CLIENT)

    assert before <= created.created_at <= datetime.now(UTC)
    assert created.created_at.tzinfo is UTC


async def test_a_new_conversation_has_no_title_and_no_messages(
    store: SqlChatStore, clock: Clock
) -> None:
    created = await store.create_conversation(CLIENT)

    thread = await store.get_thread(CLIENT, created.id)
    assert thread is not None
    assert thread.conversation == created
    assert (created.title, created.created_at, created.updated_at) == (None, clock.now, clock.now)
    assert thread.messages == ()


async def test_conversation_ids_are_unique(store: SqlChatStore) -> None:
    ids = {(await store.create_conversation(CLIENT)).id for _ in range(3)}

    assert len(ids) == 3


async def test_another_clients_conversation_does_not_exist(store: SqlChatStore) -> None:
    created = await store.create_conversation(CLIENT)

    assert await store.get_thread(OTHER_CLIENT, created.id) is None
    assert await store.get_thread(CLIENT, MISSING_ID) is None


async def test_list_is_newest_first_and_only_the_clients_own(
    store: SqlChatStore, clock: Clock
) -> None:
    older = await store.create_conversation(CLIENT)
    clock.advance(seconds=1)
    await store.create_conversation(OTHER_CLIENT)
    clock.advance(seconds=1)
    newer = await store.create_conversation(CLIENT)

    page = await store.list_conversations(CLIENT, limit=10)

    assert [c.id for c in page.items] == [newer.id, older.id]
    assert page.next_cursor is None


async def test_list_pages_with_a_keyset_cursor(store: SqlChatStore, clock: Clock) -> None:
    created = []
    for _ in range(5):
        created.append(await store.create_conversation(CLIENT))
        clock.advance(seconds=1)
    expected = [c.id for c in reversed(created)]

    first = await store.list_conversations(CLIENT, limit=2)
    assert first.next_cursor is not None
    second = await store.list_conversations(CLIENT, limit=2, after=first.next_cursor)
    assert second.next_cursor is not None
    last = await store.list_conversations(CLIENT, limit=2, after=second.next_cursor)

    assert [c.id for c in (*first.items, *second.items, *last.items)] == expected
    assert first.next_cursor == ConversationCursor(first.items[-1].updated_at, first.items[-1].id)
    assert last.next_cursor is None


async def test_a_page_that_ends_exactly_at_the_limit_has_no_cursor(store: SqlChatStore) -> None:
    await store.create_conversation(CLIENT)
    await store.create_conversation(CLIENT)

    page = await store.list_conversations(CLIENT, limit=2)

    assert len(page.items) == 2
    assert page.next_cursor is None


async def test_equal_update_times_page_by_id_without_gaps_or_repeats(store: SqlChatStore) -> None:
    created = [await store.create_conversation(CLIENT) for _ in range(4)]  # one clock tick

    seen: list[str] = []
    cursor: ConversationCursor | None = None
    while True:
        page = await store.list_conversations(CLIENT, limit=1, after=cursor)
        seen.extend(c.id for c in page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor

    assert seen == sorted((c.id for c in created), reverse=True)


async def test_a_turn_moves_its_conversation_to_the_top(store: SqlChatStore, clock: Clock) -> None:
    first = await store.create_conversation(CLIENT)
    clock.advance(seconds=1)
    second = await store.create_conversation(CLIENT)
    clock.advance(seconds=1)

    await store.start_turn(CLIENT, first.id, "Synthetic question?", title="Synthetic question?")

    page = await store.list_conversations(CLIENT, limit=10)
    assert [c.id for c in page.items] == [first.id, second.id]
    assert page.items[0].updated_at == clock.now


async def test_deleting_removes_the_conversation_and_its_messages(
    store: SqlChatStore, sessions: async_sessionmaker[AsyncSession]
) -> None:
    created = await store.create_conversation(CLIENT)
    await store.start_turn(CLIENT, created.id, "Synthetic question?", title="Synthetic question?")

    assert await store.delete_conversation(CLIENT, created.id) is True

    assert await store.get_thread(CLIENT, created.id) is None
    async with sessions() as session:
        assert (await session.execute(select(MessageRow))).all() == []
    assert await store.delete_conversation(CLIENT, created.id) is False


async def test_another_client_cannot_delete_a_conversation(store: SqlChatStore) -> None:
    created = await store.create_conversation(CLIENT)

    assert await store.delete_conversation(OTHER_CLIENT, created.id) is False

    assert await store.get_thread(CLIENT, created.id) is not None


# Turns


async def test_the_first_turn_stores_the_question_a_pending_answer_and_the_title(
    store: SqlChatStore, clock: Clock
) -> None:
    created = await store.create_conversation(CLIENT)
    clock.advance(seconds=5)

    turn = await store.start_turn(CLIENT, created.id, "Synthetic question?", title="Synthetic")

    assert (turn.user.seq, turn.user.role, turn.user.content) == (
        1,
        Role.USER,
        "Synthetic question?",
    )
    assert (turn.assistant.seq, turn.assistant.outcome) == (2, Outcome.PENDING)
    assert turn.assistant.content == ""
    assert turn.user.created_at == turn.assistant.created_at == clock.now
    thread = await store.get_thread(CLIENT, created.id)
    assert thread is not None
    assert thread.messages == (turn.user, turn.assistant)
    assert (thread.conversation.title, thread.conversation.updated_at) == ("Synthetic", clock.now)


async def test_later_turns_continue_the_sequence_and_keep_the_first_title(
    store: SqlChatStore,
) -> None:
    created = await store.create_conversation(CLIENT)
    first = await store.start_turn(CLIENT, created.id, "First?", title="First")
    await store.finish_answer(CLIENT, first.assistant.id, completed())

    second = await store.start_turn(CLIENT, created.id, "Second?", title="Second")

    assert (second.user.seq, second.assistant.seq) == (3, 4)
    thread = await store.get_thread(CLIENT, created.id)
    assert thread is not None
    assert [m.seq for m in thread.messages] == [1, 2, 3, 4]
    assert thread.conversation.title == "First"


async def test_a_turn_needs_the_clients_own_conversation(
    store: SqlChatStore, sessions: async_sessionmaker[AsyncSession]
) -> None:
    created = await store.create_conversation(CLIENT)

    for client, conversation_id in ((OTHER_CLIENT, created.id), (CLIENT, MISSING_ID)):
        with pytest.raises(ConversationNotFoundError):
            await store.start_turn(client, conversation_id, "Question?", title="Question?")

    async with sessions() as session:
        assert (await session.execute(select(MessageRow))).all() == []
    thread = await store.get_thread(CLIENT, created.id)
    assert thread is not None
    assert thread.conversation.title is None


async def test_a_second_turn_while_an_answer_is_pending_is_busy(store: SqlChatStore) -> None:
    created = await store.create_conversation(CLIENT)
    first = await store.start_turn(CLIENT, created.id, "First?", title="First")

    with pytest.raises(ConversationBusyError):
        await store.start_turn(CLIENT, created.id, "Second?", title="Second")

    thread = await store.get_thread(CLIENT, created.id)
    assert thread is not None
    assert thread.messages == (first.user, first.assistant)


async def test_a_pending_answer_still_inside_the_window_keeps_the_conversation_busy(
    store: SqlChatStore, clock: Clock
) -> None:
    created = await store.create_conversation(CLIENT)
    await store.start_turn(CLIENT, created.id, "First?", title="First")
    clock.advance(seconds=ABANDON_AFTER.total_seconds())

    with pytest.raises(ConversationBusyError):
        await store.start_turn(CLIENT, created.id, "Second?", title="Second")


async def test_a_stale_pending_answer_is_marked_abandoned_and_the_turn_proceeds(
    store: SqlChatStore, sessions: async_sessionmaker[AsyncSession], clock: Clock
) -> None:
    created = await store.create_conversation(CLIENT)
    crashed = await store.start_turn(CLIENT, created.id, "First?", title="First")
    clock.advance(seconds=ABANDON_AFTER.total_seconds() + 1)

    turn = await store.start_turn(CLIENT, created.id, "Second?", title="Second")

    thread = await store.get_thread(CLIENT, created.id)
    assert thread is not None
    leftover = thread.messages[1]
    assert (leftover.id, leftover.outcome) == (crashed.assistant.id, Outcome.FAILED)
    assert (turn.user.seq, turn.assistant.outcome) == (3, Outcome.PENDING)
    assert await error_code(sessions, crashed.assistant.id) == "abandoned"


async def error_code(sessions: async_sessionmaker[AsyncSession], message_id: str) -> str | None:
    row = await message_row(sessions, message_id)
    return row.error_code


async def message_row(sessions: async_sessionmaker[AsyncSession], message_id: str) -> MessageRow:
    async with sessions() as session:
        return (
            await session.execute(select(MessageRow).where(MessageRow.id == message_id))
        ).scalar_one()


async def test_discarding_a_first_turn_leaves_nothing_of_it(store: SqlChatStore) -> None:
    created = await store.create_conversation(CLIENT)
    turn = await store.start_turn(CLIENT, created.id, "Personal question?", title="Personal")

    await store.discard_turn(CLIENT, turn)

    thread = await store.get_thread(CLIENT, created.id)
    assert thread is not None
    assert thread.messages == ()
    assert thread.conversation.title is None  # the title came from the discarded message


async def test_discarding_a_later_turn_keeps_earlier_messages_and_the_title(
    store: SqlChatStore,
) -> None:
    created = await store.create_conversation(CLIENT)
    first = await store.start_turn(CLIENT, created.id, "First?", title="First")
    answered = await store.finish_answer(CLIENT, first.assistant.id, completed())
    second = await store.start_turn(CLIENT, created.id, "Second?", title="Second")

    await store.discard_turn(CLIENT, second)

    thread = await store.get_thread(CLIENT, created.id)
    assert thread is not None
    assert thread.messages == (first.user, answered)
    assert thread.conversation.title == "First"
    retry = await store.start_turn(CLIENT, created.id, "Second again?", title="ignored")
    assert retry.user.seq == 3


async def test_another_client_cannot_discard_a_turn(store: SqlChatStore) -> None:
    created = await store.create_conversation(CLIENT)
    turn = await store.start_turn(CLIENT, created.id, "Question?", title="Question")

    await store.discard_turn(OTHER_CLIENT, turn)

    thread = await store.get_thread(CLIENT, created.id)
    assert thread is not None
    assert thread.messages == (turn.user, turn.assistant)
    assert thread.conversation.title == "Question"


# Finishing answers


async def test_a_finished_answer_is_returned_and_read_back(
    store: SqlChatStore, clock: Clock
) -> None:
    created = await store.create_conversation(CLIENT)
    turn = await store.start_turn(CLIENT, created.id, "Question?", title="Question")
    clock.advance(seconds=2)

    answer = await store.finish_answer(CLIENT, turn.assistant.id, completed())

    assert answer is not None
    assert (answer.id, answer.seq, answer.created_at) == (
        turn.assistant.id,
        turn.assistant.seq,
        turn.assistant.created_at,
    )
    assert (answer.outcome, answer.content) == (Outcome.COMPLETED, "Synthetic answer [S1].")
    assert (answer.citations, answer.sources) == (("S1",), (SOURCE,))
    thread = await store.get_thread(CLIENT, created.id)
    assert thread is not None
    assert thread.messages[-1] == answer
    assert thread.conversation.updated_at == clock.now


async def test_the_full_answer_trace_is_stored(
    store: SqlChatStore, sessions: async_sessionmaker[AsyncSession]
) -> None:
    created = await store.create_conversation(CLIENT)
    turn = await store.start_turn(CLIENT, created.id, "Question?", title="Question")

    await store.finish_answer(CLIENT, turn.assistant.id, completed())

    row = await message_row(sessions, turn.assistant.id)
    assert row.chunk_ids == ["chunk-1", "chunk-2"]
    assert (row.model, row.input_tokens, row.output_tokens) == ("stub", 120, 8)
    assert (row.dense_embed_ms, row.sparse_embed_ms, row.search_ms, row.rerank_ms) == (
        1.5,
        0.5,
        3.0,
        40.0,
    )
    assert row.retrieval_ms == 45.0
    assert (row.rewrite_ms, row.ttft_ms, row.generation_ms, row.total_ms) == (
        12.0,
        60.0,
        80.0,
        150.0,
    )
    assert (row.relevance_signal, row.relevance_threshold) == (-1.25, -6.54)
    assert (row.index_build_id, row.rewritten_query) == (
        "20261113t173108z",
        "synthetic standalone query",
    )
    assert row.flags == {
        "late_refusal_marker": False,
        "uncited": False,
        "invalid_citations": 1,
        "rewrite_fallback": None,
    }
    assert (row.refusal, row.refusal_reason, row.error_code) == (False, None, None)


async def test_a_rewrite_fallback_is_stored_as_its_code(
    store: SqlChatStore, sessions: async_sessionmaker[AsyncSession]
) -> None:
    created = await store.create_conversation(CLIENT)
    turn = await store.start_turn(CLIENT, created.id, "Question?", title="Question")

    await store.finish_answer(
        CLIENT, turn.assistant.id, completed(flags=AnswerFlags(rewrite_fallback="timeout"))
    )

    row = await message_row(sessions, turn.assistant.id)
    assert row.flags["rewrite_fallback"] == "timeout"


async def test_a_refusal_before_the_model_stores_retrieval_but_no_model_data(
    store: SqlChatStore,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    created = await store.create_conversation(CLIENT)
    turn = await store.start_turn(CLIENT, created.id, "Unrelated?", title="Unrelated")
    refusal = AnswerRecord(
        outcome=Outcome.REFUSED,
        content="Fixed refusal text.",
        refusal_reason=RefusalReason.BELOW_THRESHOLD,
        chunk_ids=("chunk-9",),
        timings=StageTimings(None, None, 2.0, None, 2.0),
        relevance=Relevance(signal=None, threshold=None, above_threshold=None),
    )

    answer = await store.finish_answer(CLIENT, turn.assistant.id, refusal)

    assert answer is not None
    assert (answer.refusal, answer.refusal_reason) == (True, RefusalReason.BELOW_THRESHOLD)
    row = await message_row(sessions, turn.assistant.id)
    assert (row.refusal, row.chunk_ids, row.model, row.input_tokens) == (
        True,
        ["chunk-9"],
        None,
        None,
    )
    assert (row.dense_embed_ms, row.search_ms, row.rerank_ms, row.retrieval_ms) == (
        None,
        2.0,
        None,
        2.0,
    )
    assert (row.relevance_signal, row.relevance_threshold) == (None, None)


async def test_a_failed_answer_keeps_its_partial_text_and_error_code(
    store: SqlChatStore, sessions: async_sessionmaker[AsyncSession]
) -> None:
    created = await store.create_conversation(CLIENT)
    turn = await store.start_turn(CLIENT, created.id, "Question?", title="Question")
    failed = AnswerRecord(outcome=Outcome.FAILED, content="Partial", error_code="llm_unavailable")

    answer = await store.finish_answer(CLIENT, turn.assistant.id, failed)

    assert answer is not None
    assert (answer.outcome, answer.content) == (Outcome.FAILED, "Partial")
    assert await error_code(sessions, turn.assistant.id) == "llm_unavailable"
    after = await store.start_turn(CLIENT, created.id, "Again?", title="ignored")
    assert after.user.seq == 3  # a finished answer, even a failed one, frees the conversation


async def test_only_a_pending_answer_of_the_client_can_be_finished(
    store: SqlChatStore, clock: Clock
) -> None:
    created = await store.create_conversation(CLIENT)
    turn = await store.start_turn(CLIENT, created.id, "Question?", title="Question")
    clock.advance(seconds=1)

    assert await store.finish_answer(OTHER_CLIENT, turn.assistant.id, completed()) is None
    assert await store.finish_answer(CLIENT, turn.user.id, completed()) is None
    assert await store.finish_answer(CLIENT, MISSING_ID, completed()) is None
    thread = await store.get_thread(CLIENT, created.id)
    assert thread is not None
    assert thread.messages == (turn.user, turn.assistant)
    assert thread.conversation.updated_at == T0  # untouched by the refused writes

    assert await store.finish_answer(CLIENT, turn.assistant.id, completed()) is not None
    assert await store.finish_answer(CLIENT, turn.assistant.id, completed(content="x")) is None


async def test_finishing_after_the_conversation_was_deleted_stores_nothing(
    store: SqlChatStore,
) -> None:
    created = await store.create_conversation(CLIENT)
    turn = await store.start_turn(CLIENT, created.id, "Question?", title="Question")
    await store.delete_conversation(CLIENT, created.id)

    assert await store.finish_answer(CLIENT, turn.assistant.id, completed()) is None


# Ownership in every statement


@pytest.fixture
def statements(engine: AsyncEngine) -> Iterator[list[tuple[str, Any]]]:
    """Every SQL statement the engine sends, with its bound parameters."""
    seen: list[tuple[str, Any]] = []

    def record(
        _conn: Connection, _cursor: object, statement: str, parameters: Any, *_: object
    ) -> None:  # Any: DBAPI parameters are a tuple or a list of tuples
        seen.append((statement, parameters))

    event.listen(engine.sync_engine, "before_cursor_execute", record)
    yield seen
    event.remove(engine.sync_engine, "before_cursor_execute", record)


def flatten(parameters: Any) -> list[object]:  # Any: tuple, or list of tuples for executemany
    if isinstance(parameters, list):
        return [value for row in parameters for value in row]
    return list(parameters)


async def exercise_every_method(store: SqlChatStore, clock: Clock) -> None:
    created = await store.create_conversation(CLIENT)
    await store.list_conversations(CLIENT, limit=1)
    await store.create_conversation(CLIENT)
    page = await store.list_conversations(CLIENT, limit=1)
    await store.list_conversations(CLIENT, limit=1, after=page.next_cursor)
    turn = await store.start_turn(CLIENT, created.id, "Question?", title="Question")
    await store.finish_answer(CLIENT, turn.assistant.id, completed())
    stale = await store.start_turn(CLIENT, created.id, "Again?", title="ignored")
    clock.advance(seconds=ABANDON_AFTER.total_seconds() + 1)
    retry = await store.start_turn(CLIENT, created.id, "Retry?", title="ignored")
    await store.discard_turn(CLIENT, retry)
    assert stale.assistant.id
    await store.get_thread(CLIENT, created.id)
    await store.delete_conversation(CLIENT, created.id)


async def test_every_statement_on_chat_tables_binds_the_client_id(
    store: SqlChatStore, clock: Clock, statements: list[tuple[str, Any]]
) -> None:
    await exercise_every_method(store, clock)

    chat = [
        (sql, params) for sql, params in statements if "conversations" in sql or "messages" in sql
    ]
    assert len(chat) >= 20, "expected the exercise to reach every repository statement"
    for sql, params in chat:
        assert "client_id" in sql, sql
        assert CLIENT in flatten(params), sql
    verbs = {sql.split()[0] for sql, _ in chat}
    assert verbs == {"INSERT", "SELECT", "UPDATE", "DELETE"}


async def test_no_other_client_id_is_ever_bound(
    store: SqlChatStore, clock: Clock, statements: list[tuple[str, Any]]
) -> None:
    """Guards the audit above: a statement binding the client id somewhere irrelevant."""
    await exercise_every_method(store, clock)

    assert all(OTHER_CLIENT not in flatten(params) for _, params in statements)


# The same rules hold in the schema (migration 0005 repeats them on Postgres).


async def test_the_schema_accepts_valid_rows(engine: AsyncEngine) -> None:
    """The baseline rows are valid, so each rejection below comes from its one rule."""
    async with engine.begin() as connection:
        await insert_conversation(connection, conversation_row(title="Synthetic"))
        for row in VALID_MESSAGES:
            await insert_message(connection, row)


@pytest.mark.parametrize(("row", "constraint"), MESSAGE_VIOLATIONS)
async def test_the_schema_rejects_a_message_that_breaks_a_rule(
    engine: AsyncEngine, row: Row, constraint: str
) -> None:
    async with engine.begin() as connection:
        await insert_conversation(connection, conversation_row())

    with pytest.raises(IntegrityError, match=constraint):
        async with engine.begin() as connection:
            await insert_message(connection, row)


@pytest.mark.parametrize(("row", "constraint"), CONVERSATION_VIOLATIONS)
async def test_the_schema_rejects_a_conversation_that_breaks_a_rule(
    engine: AsyncEngine, row: Row, constraint: str
) -> None:
    with pytest.raises(IntegrityError, match=constraint):
        async with engine.begin() as connection:
            await insert_conversation(connection, row)


# Which integrity errors mean "busy"


async def test_a_check_violation_during_a_turn_surfaces_instead_of_reading_as_busy(
    store: SqlChatStore, sessions: async_sessionmaker[AsyncSession]
) -> None:
    created = await store.create_conversation(CLIENT)

    with pytest.raises(IntegrityError, match="ck_conversations_title_not_empty"):
        await store.start_turn(CLIENT, created.id, "Question?", title="")

    async with sessions() as session:
        assert (await session.execute(select(MessageRow))).all() == []


async def violation(engine: AsyncEngine, row: Row) -> IntegrityError:
    async with engine.begin() as connection:
        await insert_conversation(connection, conversation_row())
        await insert_message(connection, PENDING_ANSWER)
    with pytest.raises(IntegrityError) as raised:
        async with engine.begin() as connection:
            await insert_message(connection, row)
    return raised.value


@pytest.mark.parametrize(("row", "pending"), CONFLICT_CASES)
async def test_only_a_second_pending_answer_is_a_pending_conflict_on_sqlite(
    engine: AsyncEngine, row: Row, pending: bool
) -> None:
    """SQLite names the indexed columns, not the index (Postgres: see integration)."""
    assert is_pending_answer_conflict(await violation(engine, row)) is pending


@pytest.mark.parametrize(
    ("constraint", "pending"),
    [
        ("ix_messages_one_pending", True),
        ("fk_messages_conversation_id_conversations", False),
        ("ck_messages_seq_positive", False),
        ("uq_messages_conversation_id_seq", False),
    ],
)
def test_a_named_driver_constraint_is_matched_by_name(constraint: str, pending: bool) -> None:
    """asyncpg reports the violated constraint as driver_exception.constraint_name."""
    driver = SimpleNamespace(constraint_name=constraint)
    orig = SimpleNamespace(driver_exception=driver)
    error = IntegrityError("INSERT INTO messages ...", None, orig)  # type: ignore[arg-type]

    assert is_pending_answer_conflict(error) is pending


async def test_an_unexpected_integrity_error_becomes_a_safe_500(
    store: SqlChatStore, json_logs: LogReader
) -> None:
    """The driver text holds SQL parameters (user content); none of it may leave."""
    canary = "canary-7f3a9c"
    created = await store.create_conversation(CLIENT)
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.post("/turn")
    async def turn() -> None:
        await store.start_turn(CLIENT, created.id, canary, title="")

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        response = await http.post("/turn")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    logs = json_logs()
    assert [line["exception_type"] for line in logs if line["level"] == "ERROR"] == [
        "sqlalchemy.exc.IntegrityError"
    ]
    for text_out in (response.text, *map(str, logs)):
        assert canary not in text_out
        assert "ck_conversations" not in text_out
        assert "INSERT" not in text_out
        assert "UPDATE" not in text_out

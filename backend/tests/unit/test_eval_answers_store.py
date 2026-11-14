"""The answer benchmark's store reader on SQLite; the Postgres test proves it is read only.

Rows are written through SqlChatStore, so they have exactly the shape the backend stores.
"""

import pathlib
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine as create_sqlite_engine

from app.domain.chat import AnswerFlags, AnswerRecord, Outcome, RefusalReason
from app.domain.retrieval import Relevance, StageTimings
from app.evaluation.answers.store import AnswerRows, SqlAnswerRows, StoredAnswer
from app.infra.chat_store import SqlChatStore
from app.infra.db import create_session_factory
from app.infra.orm import Base
from tests.chat_rows import CLIENT, OTHER_CLIENT


@pytest.fixture
async def engine(tmp_path: pathlib.Path) -> AsyncIterator[AsyncEngine]:
    engine = create_sqlite_engine(f"sqlite+aiosqlite:///{tmp_path / 'chat.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def sessions(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return create_session_factory(engine)


@pytest.fixture
def chat(sessions: async_sessionmaker[AsyncSession]) -> SqlChatStore:
    return SqlChatStore(sessions, abandon_after=timedelta(minutes=3))


def answered(**overrides: object) -> AnswerRecord:
    fields: dict[str, object] = {
        "outcome": Outcome.COMPLETED,
        "content": "Synthetic answer.",
        "chunk_ids": ("chunk-b", "chunk-a", "chunk-c"),
        "model": "synthetic-model",
        "input_tokens": 4100,
        "output_tokens": 310,
        "timings": StageTimings(1.5, 0.5, 3.0, 40.0, 45.0),
        "rewrite_ms": 12.0,
        "ttft_ms": 600.0,
        "generation_ms": 2400.0,
        "total_ms": 3100.0,
        "relevance": Relevance.from_top_score(-1.25, threshold=-6.54),
        "index_build_id": "20261113t173108z",
        "rewritten_query": "synthetic standalone query",
        "flags": AnswerFlags(),
    }
    fields.update(overrides)
    return AnswerRecord(**fields)  # type: ignore[arg-type]  # fields typed per test


async def turn(chat: SqlChatStore, conversation: str, record: AnswerRecord) -> str:
    started = await chat.start_turn(CLIENT, conversation, "Synthetic question?", title=None)
    await chat.finish_answer(CLIENT, started.assistant.id, record)
    return started.assistant.id


def test_the_sql_reader_implements_the_port(sessions: async_sessionmaker[AsyncSession]) -> None:
    reader = SqlAnswerRows(sessions)
    port: AnswerRows = reader  # mypy checks the structural match
    assert port is reader


async def test_every_assistant_row_is_read_in_turn_order_with_its_trace(
    chat: SqlChatStore, sessions: async_sessionmaker[AsyncSession]
) -> None:
    conversation = (await chat.create_conversation(CLIENT)).id
    first = await turn(
        chat,
        conversation,
        answered(
            outcome=Outcome.REFUSED,
            content="Refused.",
            refusal_reason=RefusalReason.BELOW_THRESHOLD,
            model=None,
            input_tokens=None,
            output_tokens=None,
            rewrite_ms=None,
            rewritten_query=None,
            ttft_ms=None,
            generation_ms=None,
            relevance=Relevance.from_top_score(-9.0, threshold=-6.54),
        ),
    )
    second = await turn(chat, conversation, answered())

    rows = await SqlAnswerRows(sessions).answers(conversation, CLIENT)

    assert rows == (
        StoredAnswer(
            message_id=first,
            outcome=Outcome.REFUSED,
            refusal_reason=RefusalReason.BELOW_THRESHOLD,
            model=None,
            input_tokens=None,
            output_tokens=None,
            chunk_ids=("chunk-b", "chunk-a", "chunk-c"),
            rewritten_query=None,
            rewrite_ms=None,
            index_build_id="20261113t173108z",
            relevance_signal=-9.0,
            retrieval_ms=45.0,
            ttft_ms=None,
            generation_ms=None,
            total_ms=3100.0,
        ),
        StoredAnswer(
            message_id=second,
            outcome=Outcome.COMPLETED,
            refusal_reason=None,
            model="synthetic-model",
            input_tokens=4100,
            output_tokens=310,
            chunk_ids=("chunk-b", "chunk-a", "chunk-c"),
            rewritten_query="synthetic standalone query",
            rewrite_ms=12.0,
            index_build_id="20261113t173108z",
            relevance_signal=-1.25,
            retrieval_ms=45.0,
            ttft_ms=600.0,
            generation_ms=2400.0,
            total_ms=3100.0,
        ),
    )


async def test_an_answer_still_pending_is_read_with_an_empty_trace(
    chat: SqlChatStore, sessions: async_sessionmaker[AsyncSession]
) -> None:
    conversation = (await chat.create_conversation(CLIENT)).id
    started = await chat.start_turn(CLIENT, conversation, "Synthetic question?", title=None)

    (row,) = await SqlAnswerRows(sessions).answers(conversation, CLIENT)

    assert (row.message_id, row.outcome, row.model, row.chunk_ids) == (
        started.assistant.id,
        Outcome.PENDING,
        None,
        (),
    )
    assert (row.relevance_signal, row.retrieval_ms, row.total_ms) == (None, None, None)


async def test_another_clients_conversation_reads_as_empty(
    chat: SqlChatStore, sessions: async_sessionmaker[AsyncSession]
) -> None:
    conversation = (await chat.create_conversation(CLIENT)).id
    message = await turn(chat, conversation, answered())
    reader = SqlAnswerRows(sessions)

    assert await reader.answers(conversation, OTHER_CLIENT) == ()
    assert [row.message_id for row in await reader.answers(conversation, CLIENT)] == [message]

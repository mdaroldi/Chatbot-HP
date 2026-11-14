"""What the backend stored for an answered item, read back for scoring and charging.

The harness reads the assistant rows of the item's conversation from Postgres in a
READ ONLY transaction, so a bug here can never change the backend's data (no backend
change: the row shape is the chat store's own). Rows are selected by conversation and
client ID together, as the chat store's ownership rule does. The trace holds what the
API does not send: the chunk ids in rank order (tags S1..Sk), the serving model, token
usage, the rewritten query, the index build, the relevance signal and stage timings.

Message content is not read: the harness already has it from the `done` event.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.chat import Outcome, RefusalReason, Role
from app.infra.orm import MessageRow


@dataclass(frozen=True)
class StoredAnswer:
    """One assistant row's trace; fields its outcome never reached are None."""

    message_id: str
    outcome: Outcome
    refusal_reason: RefusalReason | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    chunk_ids: tuple[str, ...]
    rewritten_query: str | None
    rewrite_ms: float | None  # set exactly when the rewrite model was called
    index_build_id: str | None
    relevance_signal: float | None
    retrieval_ms: float | None
    ttft_ms: float | None
    generation_ms: float | None
    total_ms: float | None


class AnswerRows(Protocol):
    async def answers(self, conversation_id: str, client_id: str) -> tuple[StoredAnswer, ...]:
        """The conversation's assistant rows in turn order; empty when the client does
        not own it.
        """
        ...


@asynccontextmanager
async def read_only_session(
    sessions: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """A session whose transaction Postgres runs READ ONLY: any write fails."""
    async with sessions() as session, session.begin():
        await session.connection(execution_options={"postgresql_readonly": True})
        yield session


def _stored(row: MessageRow) -> StoredAnswer:
    return StoredAnswer(
        message_id=row.id,
        # An assistant row always has one (check outcome_role); str() keeps mypy honest.
        outcome=Outcome(str(row.outcome)),
        refusal_reason=None if row.refusal_reason is None else RefusalReason(row.refusal_reason),
        model=row.model,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        chunk_ids=tuple(row.chunk_ids),
        rewritten_query=row.rewritten_query,
        rewrite_ms=row.rewrite_ms,
        index_build_id=row.index_build_id,
        relevance_signal=row.relevance_signal,
        retrieval_ms=row.retrieval_ms,
        ttft_ms=row.ttft_ms,
        generation_ms=row.generation_ms,
        total_ms=row.total_ms,
    )


class SqlAnswerRows:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def answers(self, conversation_id: str, client_id: str) -> tuple[StoredAnswer, ...]:
        statement = (
            select(MessageRow)
            .where(
                MessageRow.conversation_id == conversation_id,
                MessageRow.client_id == client_id,
                MessageRow.role == Role.ASSISTANT.value,
            )
            .order_by(MessageRow.seq)
        )
        async with read_only_session(self._sessions) as session:
            rows = (await session.execute(statement)).scalars().all()
        return tuple(_stored(row) for row in rows)

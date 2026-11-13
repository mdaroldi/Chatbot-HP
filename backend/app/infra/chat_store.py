"""ChatStore over SQLAlchemy: conversations and messages in the relational database.

Ownership lives in the statements themselves: every SELECT, UPDATE and DELETE filters
on the client ID, and every INSERT writes it, so no code path can read or change
another client's rows by forgetting a check. The composite foreign key repeats the
rule in the schema (a message cannot point at another client's conversation).
"""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from sqlalchemy import ColumnElement, and_, delete, exists, func, insert, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.chat import (
    AnswerFlags,
    AnswerRecord,
    Conversation,
    Message,
    Outcome,
    RefusalReason,
    Role,
    Source,
)
from app.domain.chat_store import (
    ConversationBusyError,
    ConversationCursor,
    ConversationNotFoundError,
    ConversationPage,
    ConversationThread,
    Turn,
)
from app.domain.ingestion import PageRange
from app.domain.retrieval import Relevance, StageTimings
from app.infra.orm import ConversationRow, FlagsJson, MessageRow, SourceJson

# Error code stored on a pending answer whose request died without finishing it.
ABANDONED = "abandoned"

Values = dict[str, object]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return str(uuid.uuid4())


def _utc(value: datetime) -> datetime:
    # Stored in UTC. asyncpg returns it as UTC; SQLite (unit tests) drops the zone.
    return value.replace(tzinfo=UTC)


def _conversation(row: ConversationRow) -> Conversation:
    return Conversation(row.id, row.title, _utc(row.created_at), _utc(row.updated_at))


def _source_json(source: Source) -> SourceJson:
    pages = source.pages
    return {
        "tag": source.tag,
        "doc_id": source.doc_id,
        "document_title": source.document_title,
        "section_path": list(source.section_path),
        "pages": {
            "pdf_start": pages.pdf_start,
            "pdf_end": pages.pdf_end,
            "printed_start": pages.printed_start,
            "printed_end": pages.printed_end,
        },
        "has_warning": source.has_warning,
        "snippet": source.snippet,
    }


def _source(data: SourceJson) -> Source:
    pages = data["pages"]
    return Source(
        tag=data["tag"],
        doc_id=data["doc_id"],
        document_title=data["document_title"],
        section_path=tuple(data["section_path"]),
        pages=PageRange(
            pages["pdf_start"], pages["pdf_end"], pages["printed_start"], pages["printed_end"]
        ),
        has_warning=data["has_warning"],
        snippet=data["snippet"],
    )


def _message(row: MessageRow) -> Message:
    return Message(
        id=row.id,
        conversation_id=row.conversation_id,
        seq=row.seq,
        role=Role(row.role),
        content=row.content,
        created_at=_utc(row.created_at),
        outcome=None if row.outcome is None else Outcome(row.outcome),
        refusal_reason=None if row.refusal_reason is None else RefusalReason(row.refusal_reason),
        citations=tuple(row.citations),
        sources=tuple(_source(s) for s in row.sources),
    )


def _flags_json(flags: AnswerFlags) -> FlagsJson:
    return {
        "late_refusal_marker": flags.late_refusal_marker,
        "uncited": flags.uncited,
        "invalid_citations": flags.invalid_citations,
    }


def _timing_values(timings: StageTimings | None) -> Values:
    if timings is None:
        return {}
    return {
        "dense_embed_ms": timings.dense_embed_ms,
        "sparse_embed_ms": timings.sparse_embed_ms,
        "search_ms": timings.search_ms,
        "rerank_ms": timings.rerank_ms,
        "retrieval_ms": timings.total_ms,
    }


def _relevance_values(relevance: Relevance | None) -> Values:
    if relevance is None:
        return {}
    return {"relevance_signal": relevance.signal, "relevance_threshold": relevance.threshold}


def _answer_values(answer: AnswerRecord) -> Values:
    reason = answer.refusal_reason
    return {
        "outcome": answer.outcome.value,
        "content": answer.content,
        "refusal": reason is not None,
        "refusal_reason": None if reason is None else reason.value,
        "error_code": answer.error_code,
        "flags": _flags_json(answer.flags),
        "citations": list(answer.citations),
        "sources": [_source_json(s) for s in answer.sources],
        "chunk_ids": list(answer.chunk_ids),
        "model": answer.model,
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
        "rewrite_ms": answer.rewrite_ms,
        "ttft_ms": answer.ttft_ms,
        "generation_ms": answer.generation_ms,
        "total_ms": answer.total_ms,
        "index_build_id": answer.index_build_id,
        "rewritten_query": answer.rewritten_query,
        **_timing_values(answer.timings),
        **_relevance_values(answer.relevance),
    }


def _new_row(client_id: str, message: Message) -> Values:
    return {
        "id": message.id,
        "conversation_id": message.conversation_id,
        "client_id": client_id,
        "seq": message.seq,
        "role": message.role.value,
        "content": message.content,
        "created_at": message.created_at,
        "outcome": None if message.outcome is None else message.outcome.value,
        "refusal": False,
        "flags": {},
        "citations": [],
        "sources": [],
        "chunk_ids": [],
    }


def _owned_conversation(client_id: str, conversation_id: str) -> ColumnElement[bool]:
    return and_(ConversationRow.id == conversation_id, ConversationRow.client_id == client_id)


class SqlChatStore:
    """ChatStore over SQLAlchemy (see the module docstring for the ownership rule).

    `abandon_after` is how old a pending answer must be before a new turn treats it as
    left over from a crash; it must exceed the longest a live request can take.
    """

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        abandon_after: timedelta,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._sessions = sessions
        self._abandon_after = abandon_after
        self._clock = clock

    async def create_conversation(self, client_id: str) -> Conversation:
        now = self._clock()
        conversation = Conversation(_new_id(), None, now, now)
        async with self._sessions() as session, session.begin():
            await session.execute(
                insert(ConversationRow).values(
                    id=conversation.id,
                    client_id=client_id,
                    title=None,
                    created_at=now,
                    updated_at=now,
                )
            )
        return conversation

    async def list_conversations(
        self, client_id: str, *, limit: int, after: ConversationCursor | None = None
    ) -> ConversationPage:
        query = select(ConversationRow).where(ConversationRow.client_id == client_id)
        if after is not None:
            query = query.where(
                or_(
                    ConversationRow.updated_at < after.updated_at,
                    and_(
                        ConversationRow.updated_at == after.updated_at,
                        ConversationRow.id < after.id,
                    ),
                )
            )
        query = query.order_by(ConversationRow.updated_at.desc(), ConversationRow.id.desc())
        async with self._sessions() as session:
            rows = (await session.execute(query.limit(limit + 1))).scalars().all()
        items = tuple(_conversation(row) for row in rows[:limit])
        more = len(rows) > limit
        cursor = ConversationCursor(items[-1].updated_at, items[-1].id) if more else None
        return ConversationPage(items, cursor)

    async def get_thread(self, client_id: str, conversation_id: str) -> ConversationThread | None:
        conversation_query = select(ConversationRow).where(
            _owned_conversation(client_id, conversation_id)
        )
        messages_query = (
            select(MessageRow)
            .where(MessageRow.conversation_id == conversation_id, MessageRow.client_id == client_id)
            .order_by(MessageRow.seq)
        )
        async with self._sessions() as session, session.begin():
            row = (await session.execute(conversation_query)).scalar_one_or_none()
            if row is None:
                return None
            messages = (await session.execute(messages_query)).scalars().all()
        return ConversationThread(_conversation(row), tuple(_message(m) for m in messages))

    async def delete_conversation(self, client_id: str, conversation_id: str) -> bool:
        statement = delete(ConversationRow).where(_owned_conversation(client_id, conversation_id))
        async with self._sessions() as session, session.begin():
            result = await session.execute(statement)
        # Messages go with it (ON DELETE CASCADE on the composite foreign key).
        return bool(result.rowcount)  # type: ignore[attr-defined]  # CursorResult

    async def start_turn(
        self, client_id: str, conversation_id: str, content: str, *, title: str | None
    ) -> Turn:
        now = self._clock()
        try:
            async with self._sessions() as session, session.begin():
                # Locks the conversation row, so concurrent turns on it run one at a time.
                stored_title = await self._lock_conversation(session, client_id, conversation_id)
                await self._abandon_stale_pending(session, client_id, conversation_id, now)
                last_seq = await self._last_seq(session, client_id, conversation_id)
                turn = Turn(
                    Message(_new_id(), conversation_id, last_seq + 1, Role.USER, content, now),
                    Message(
                        _new_id(),
                        conversation_id,
                        last_seq + 2,
                        Role.ASSISTANT,
                        "",
                        now,
                        outcome=Outcome.PENDING,
                    ),
                )
                rows = [_new_row(client_id, turn.user), _new_row(client_id, turn.assistant)]
                await session.execute(insert(MessageRow).values(rows))
                await session.execute(
                    update(ConversationRow)
                    .where(_owned_conversation(client_id, conversation_id))
                    .values(updated_at=now, title=title if stored_title is None else stored_title)
                )
        except IntegrityError:
            # The one-pending index (or a racing seq) refused the rows: a turn is running.
            raise ConversationBusyError from None
        return turn

    @staticmethod
    async def _lock_conversation(
        session: AsyncSession, client_id: str, conversation_id: str
    ) -> str | None:
        query = (
            select(ConversationRow.title)
            .where(_owned_conversation(client_id, conversation_id))
            .with_for_update()
        )
        row = (await session.execute(query)).first()
        if row is None:
            raise ConversationNotFoundError
        title: str | None = row.title
        return title

    async def _abandon_stale_pending(
        self, session: AsyncSession, client_id: str, conversation_id: str, now: datetime
    ) -> None:
        await session.execute(
            update(MessageRow)
            .where(
                MessageRow.conversation_id == conversation_id,
                MessageRow.client_id == client_id,
                MessageRow.outcome == Outcome.PENDING.value,
                MessageRow.created_at < now - self._abandon_after,
            )
            .values(outcome=Outcome.FAILED.value, error_code=ABANDONED)
        )

    @staticmethod
    async def _last_seq(session: AsyncSession, client_id: str, conversation_id: str) -> int:
        query = select(func.max(MessageRow.seq)).where(
            MessageRow.conversation_id == conversation_id, MessageRow.client_id == client_id
        )
        last: int | None = (await session.execute(query)).scalar_one()
        return last or 0

    async def discard_turn(self, client_id: str, turn: Turn) -> None:
        conversation_id = turn.user.conversation_id
        remaining = exists().where(
            MessageRow.conversation_id == conversation_id, MessageRow.client_id == client_id
        )
        async with self._sessions() as session, session.begin():
            await session.execute(
                delete(MessageRow).where(
                    MessageRow.id.in_((turn.user.id, turn.assistant.id)),
                    MessageRow.conversation_id == conversation_id,
                    MessageRow.client_id == client_id,
                )
            )
            # The title comes from the first message; with no message left, neither is it.
            await session.execute(
                update(ConversationRow)
                .where(_owned_conversation(client_id, conversation_id), ~remaining)
                .values(title=None)
            )

    async def finish_answer(
        self, client_id: str, message_id: str, answer: AnswerRecord
    ) -> Message | None:
        now = self._clock()
        statement = (
            update(MessageRow)
            .where(
                MessageRow.id == message_id,
                MessageRow.client_id == client_id,
                MessageRow.outcome == Outcome.PENDING.value,
            )
            .values(**_answer_values(answer))
            .returning(MessageRow)
        )
        async with self._sessions() as session, session.begin():
            row = (await session.execute(statement)).scalar_one_or_none()
            if row is None:
                return None
            await session.execute(
                update(ConversationRow)
                .where(_owned_conversation(client_id, row.conversation_id))
                .values(updated_at=now)
            )
        return _message(row)

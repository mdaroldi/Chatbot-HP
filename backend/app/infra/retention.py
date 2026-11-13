"""Retention over SQLAlchemy: the purge statements and a Postgres advisory lock.

The purge is the one statement on the chat tables that is not scoped by client ID:
retention is a policy over everyone's data, never reachable from a request. It lives
here, apart from SqlChatStore, so the store's rule (every statement binds the client
ID) stays true without exceptions.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, async_sessionmaker

from app.core.tasks import shielded
from app.domain.retention import PurgeCounts
from app.infra.orm import ConversationRow, MessageRow

# The advisory lock's key: "hpdocs" and a 1, as one positive bigint. Any process that
# purges uses this key, so at most one of them purges at a time.
RETENTION_LOCK_KEY = int.from_bytes(b"hpdocs\x00\x01")
# Conversations per transaction: bounds the locks held and the size of the id list.
PURGE_BATCH_SIZE = 500


class SqlExpiredChats:
    """Deletes expired conversations batch by batch; messages go by ON DELETE CASCADE.

    Each batch locks its conversations first (`FOR UPDATE`, ignored by SQLite), so a
    turn that starts meanwhile either waits and then finds the conversation gone, or
    bumped `updated_at` before and keeps it: the delete re-checks the cutoff.
    """

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        batch_size: int = PURGE_BATCH_SIZE,
        on_batch: Callable[[int], object] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._sessions, self._batch_size, self._on_batch = sessions, batch_size, on_batch

    async def purge(self, updated_before: datetime) -> PurgeCounts:
        total = PurgeCounts.none()
        while True:
            batch = await self._purge_batch(updated_before)
            if batch.conversations == 0:
                return total
            total += batch
            if self._on_batch is not None:
                self._on_batch(batch.conversations)
            if batch.conversations < self._batch_size:
                return total

    async def _purge_batch(self, updated_before: datetime) -> PurgeCounts:
        expired = ConversationRow.updated_at < updated_before
        async with self._sessions() as session, session.begin():
            ids = list(
                (
                    await session.execute(
                        select(ConversationRow.id)
                        .where(expired)
                        .order_by(ConversationRow.updated_at, ConversationRow.id)
                        .limit(self._batch_size)
                        .with_for_update()
                    )
                ).scalars()
            )
            if not ids:
                return PurgeCounts.none()
            messages = (
                await session.execute(
                    select(func.count())
                    .select_from(MessageRow)
                    .where(MessageRow.conversation_id.in_(ids))
                )
            ).scalar_one()
            await session.execute(
                delete(ConversationRow).where(ConversationRow.id.in_(ids), expired)
            )
        return PurgeCounts(len(ids), int(messages))


class PostgresPurgeLock:
    """A session-level `pg_try_advisory_lock` on one connection taken for the block.

    Session-level, so it outlives the short transaction that takes it, and it is gone
    the moment its connection closes: a replica that dies mid-purge frees it. The
    connection goes back to the pool afterwards, so the unlock must succeed; when it
    cannot, the connection is invalidated (closed), which ends the session and its
    lock instead of leaving a pooled connection that would keep every later purge out.
    """

    def __init__(self, engine: AsyncEngine, *, key: int = RETENTION_LOCK_KEY) -> None:
        self._engine, self._key = engine, key

    @asynccontextmanager
    async def hold(self) -> AsyncIterator[bool]:
        async with self._engine.connect() as connection:
            result = await connection.execute(select(func.pg_try_advisory_lock(self._key)))
            held = bool(result.scalar_one())
            await connection.commit()  # ends the transaction; the lock stays with the session
            try:
                yield held
            finally:
                if held:
                    await shielded(self._release(connection))

    async def _release(self, connection: AsyncConnection) -> None:
        try:
            await connection.execute(select(func.pg_advisory_unlock(self._key)))
            await connection.commit()
        except (Exception, asyncio.CancelledError):
            await connection.invalidate()
            raise

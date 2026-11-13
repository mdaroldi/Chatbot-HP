"""SqlExpiredChats and PostgresPurgeLock on SQLite; the integration tests repeat both
on Postgres, where the lock is a real session-level advisory lock.
"""

import asyncio
import pathlib
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.domain.chat import Outcome
from app.domain.retention import ExpiredChats, PurgeCounts, PurgeLock
from app.infra.chat_store import SqlChatStore
from app.infra.db import create_session_factory
from app.infra.orm import ConversationRow, MessageRow
from app.infra.retention import RETENTION_LOCK_KEY, PostgresPurgeLock, SqlExpiredChats
from tests.chat_rows import CLIENT, OTHER_CLIENT, T0
from tests.unit.retention_fakes import AdvisoryLocks, sqlite_engine

CUTOFF = T0 - timedelta(days=30)


@pytest.fixture
def locks() -> AdvisoryLocks:
    return AdvisoryLocks()


@pytest.fixture
async def engine(tmp_path: pathlib.Path, locks: AdvisoryLocks) -> AsyncIterator[AsyncEngine]:
    async with sqlite_engine(tmp_path / "retention.db", locks) as engine:
        yield engine


@pytest.fixture
def sessions(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return create_session_factory(engine)


async def conversation_with_turn(
    sessions: async_sessionmaker[AsyncSession], client_id: str = CLIENT
) -> str:
    """A conversation with one question and its pending answer: two messages."""
    store = SqlChatStore(sessions, abandon_after=timedelta(minutes=3), clock=lambda: T0)
    created = await store.create_conversation(client_id)
    await store.start_turn(client_id, created.id, "Synthetic question?", title="Synthetic")
    return created.id


async def age(
    sessions: async_sessionmaker[AsyncSession], conversation_id: str, by: timedelta
) -> None:
    """What `make purge`'s acceptance check does in SQL: move both timestamps back
    (`updated_at` may not precede `created_at`).
    """
    async with sessions() as session, session.begin():
        await session.execute(
            update(ConversationRow)
            .where(ConversationRow.id == conversation_id)
            .values(created_at=T0 - by, updated_at=T0 - by)
        )


async def remaining(sessions: async_sessionmaker[AsyncSession]) -> tuple[set[str], int]:
    async with sessions() as session:
        ids = set((await session.execute(select(ConversationRow.id))).scalars())
        messages = (await session.execute(select(func.count()).select_from(MessageRow))).scalar()
    return ids, int(messages or 0)


# SqlExpiredChats


def test_the_adapters_implement_the_ports(
    sessions: async_sessionmaker[AsyncSession], engine: AsyncEngine
) -> None:
    sql_chats, sql_lock = SqlExpiredChats(sessions), PostgresPurgeLock(engine)
    chats: ExpiredChats = sql_chats  # mypy checks the structural matches
    lock: PurgeLock = sql_lock
    assert (chats, lock) == (sql_chats, sql_lock)


async def test_an_old_conversation_is_deleted_with_its_messages_and_a_newer_one_kept(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    old = await conversation_with_turn(sessions)
    newer = await conversation_with_turn(sessions)
    await age(sessions, old, timedelta(days=31))
    await age(sessions, newer, timedelta(days=29))

    counts = await SqlExpiredChats(sessions).purge(CUTOFF)

    assert counts == PurgeCounts(conversations=1, messages=2)
    assert await remaining(sessions) == ({newer}, 2)


async def test_a_conversation_updated_exactly_at_the_cutoff_is_kept(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    kept = await conversation_with_turn(sessions)
    await age(sessions, kept, timedelta(days=30))

    assert await SqlExpiredChats(sessions).purge(CUTOFF) == PurgeCounts.none()
    assert await remaining(sessions) == ({kept}, 2)


async def test_retention_applies_to_every_client(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """The one statement not scoped by client: retention is a policy, not a request."""
    mine = await conversation_with_turn(sessions, CLIENT)
    theirs = await conversation_with_turn(sessions, OTHER_CLIENT)
    for conversation_id in (mine, theirs):
        await age(sessions, conversation_id, timedelta(days=31))

    assert await SqlExpiredChats(sessions).purge(CUTOFF) == PurgeCounts(2, 4)
    assert await remaining(sessions) == (set(), 0)


async def test_a_large_purge_runs_in_bounded_batches(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    ids = [await conversation_with_turn(sessions) for _ in range(5)]
    for conversation_id in ids:
        await age(sessions, conversation_id, timedelta(days=40))
    batches: list[int] = []

    counts = await SqlExpiredChats(sessions, batch_size=2, on_batch=batches.append).purge(CUTOFF)

    assert counts == PurgeCounts(5, 10)
    assert batches == [2, 2, 1]
    assert await remaining(sessions) == (set(), 0)


async def test_a_purge_with_nothing_expired_deletes_nothing(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    kept = await conversation_with_turn(sessions)

    assert await SqlExpiredChats(sessions).purge(CUTOFF) == PurgeCounts.none()
    assert await remaining(sessions) == ({kept}, 2)


def test_the_batch_size_must_be_positive(sessions: async_sessionmaker[AsyncSession]) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        SqlExpiredChats(sessions, batch_size=0)


async def test_an_expired_conversation_with_an_unfinished_answer_is_purged_too(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """A pending answer that old is a crash leftover; no live request can own it."""
    old = await conversation_with_turn(sessions)
    await age(sessions, old, timedelta(days=31))
    async with sessions() as session:
        outcomes = set((await session.execute(select(MessageRow.outcome))).scalars())
    assert Outcome.PENDING.value in outcomes

    assert await SqlExpiredChats(sessions).purge(CUTOFF) == PurgeCounts(1, 2)


# PostgresPurgeLock


def test_the_lock_key_is_a_fixed_positive_bigint() -> None:
    assert 0 < RETENTION_LOCK_KEY < 2**63


async def test_the_lock_is_held_for_the_block_and_released_after_it(
    engine: AsyncEngine, locks: AdvisoryLocks
) -> None:
    lock = PostgresPurgeLock(engine)

    async with lock.hold() as held:
        assert held is True
        assert RETENTION_LOCK_KEY in locks.holders

    assert locks.holders == {}
    assert locks.unlocks == 1


async def test_a_second_holder_is_told_the_lock_is_busy_and_does_not_unlock_it(
    engine: AsyncEngine, locks: AdvisoryLocks
) -> None:
    first, second = PostgresPurgeLock(engine), PostgresPurgeLock(engine)

    async with first.hold() as held_first:
        async with second.hold() as held_second:
            assert (held_first, held_second) == (True, False)
        assert RETENTION_LOCK_KEY in locks.holders
        assert locks.unlocks == 0

    assert locks.holders == {}


async def test_the_lock_is_released_when_the_block_raises(
    engine: AsyncEngine, locks: AdvisoryLocks
) -> None:
    seen: list[bool] = []

    async def fail_inside() -> None:
        async with PostgresPurgeLock(engine).hold() as held:
            seen.append(held)
            raise LookupError

    with pytest.raises(LookupError):
        await fail_inside()

    assert seen == [True]
    assert locks.holders == {}
    assert locks.unlocks == 1


async def test_the_lock_is_released_when_the_holder_is_cancelled(
    engine: AsyncEngine, locks: AdvisoryLocks
) -> None:
    inside = asyncio.Event()

    async def hold_forever() -> None:
        async with PostgresPurgeLock(engine).hold():
            inside.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold_forever())
    await asyncio.wait_for(inside.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert locks.holders == {}
    assert locks.unlocks == 1


async def test_a_failed_unlock_closes_the_connection_so_the_server_drops_the_lock(
    engine: AsyncEngine, locks: AdvisoryLocks
) -> None:
    """A pooled connection that still held the lock would keep every later purge out."""
    with pytest.raises(OperationalError):
        async with PostgresPurgeLock(engine).hold():
            locks.fail_unlock = True

    assert locks.invalidated == 1
    assert locks.holders == {}
    locks.fail_unlock = False
    async with PostgresPurgeLock(engine).hold() as held:
        assert held is True


async def test_the_lock_uses_one_dedicated_connection_returned_afterwards(
    engine: AsyncEngine,
) -> None:
    pool = engine.pool
    async with PostgresPurgeLock(engine).hold():
        assert pool.checkedout() == 1  # type: ignore[attr-defined]  # QueuePool API

    assert pool.checkedout() == 0  # type: ignore[attr-defined]  # QueuePool API

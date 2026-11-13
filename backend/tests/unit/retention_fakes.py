"""SQLite stand-ins for Postgres session-level advisory locks, and a purge recorder.

`pg_try_advisory_lock` and `pg_advisory_unlock` are registered as SQL functions on each
SQLite connection, keyed by that connection, so a lock held on one connection is busy
for every other one until it is unlocked or its connection is closed (as on Postgres).
"""

import pathlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.domain.retention import PurgeCounts
from app.infra.orm import Base


class AdvisoryLocks:
    """Holders by key; `fail_unlock` makes the next unlock raise inside the database."""

    def __init__(self) -> None:
        self.holders: dict[int, int] = {}
        self.unlocks = 0
        self.fail_unlock = False
        self.invalidated = 0

    def install(self, dbapi_connection: Any) -> None:  # Any: the aiosqlite DBAPI adapter
        owner = id(dbapi_connection)

        def try_lock(key: int) -> int:
            if self.holders.get(key, owner) != owner:
                return 0
            self.holders[key] = owner
            return 1

        def unlock(key: int) -> int:
            if self.fail_unlock:
                raise RuntimeError("unlock failed")  # surfaces as an OperationalError
            self.unlocks += 1
            return int(self.holders.pop(key, None) == owner)

        dbapi_connection.create_function("pg_try_advisory_lock", 1, try_lock)
        dbapi_connection.create_function("pg_advisory_unlock", 1, unlock)

    def release_all_of(self, dbapi_connection: Any) -> None:  # Any: as above
        """A closed connection ends its session, and Postgres drops its locks."""
        owner = id(dbapi_connection)
        self.holders = {k: v for k, v in self.holders.items() if v != owner}


@asynccontextmanager
async def sqlite_engine(
    path: pathlib.Path, locks: AdvisoryLocks | None = None
) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    sync_engine = engine.sync_engine

    @event.listens_for(sync_engine, "connect")
    def on_connect(dbapi_connection: Any, _: object) -> None:  # Any: DBAPI connection
        dbapi_connection.execute("PRAGMA foreign_keys=ON")
        if locks is not None:
            locks.install(dbapi_connection)

    if locks is not None:

        @event.listens_for(sync_engine, "invalidate")
        def on_invalidate(dbapi_connection: Any, *_: object) -> None:  # Any: as above
            locks.invalidated += 1
            locks.release_all_of(dbapi_connection)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@dataclass
class FakeExpiredChats:
    """Records each cutoff it was asked to purge and returns fixed counts."""

    counts: PurgeCounts = field(default_factory=lambda: PurgeCounts(2, 5))
    error: Exception | None = None
    cutoffs: list[datetime] = field(default_factory=list)

    async def purge(self, updated_before: datetime) -> PurgeCounts:
        self.cutoffs.append(updated_before)
        if self.error is not None:
            raise self.error
        return self.counts


@dataclass
class FakePurgeLock:
    available: bool = True
    entered: int = 0
    exited: int = 0

    @asynccontextmanager
    async def _hold(self) -> AsyncIterator[bool]:
        self.entered += 1
        try:
            yield self.available
        finally:
            self.exited += 1

    def hold(self) -> Any:  # Any: AbstractAsyncContextManager[bool], kept loose for the fake
        return self._hold()

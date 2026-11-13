"""The retention purge: one replica at a time, counts-only logs, a loop that survives."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import pytest

from app.domain.retention import PurgeCounts
from app.services.retention import purge_expired, purge_periodically
from tests.unit.retention_fakes import FakeExpiredChats, FakePurgeLock

NOW = datetime(2026, 11, 14, 9, 0, tzinfo=UTC)
CANARY = "CANARY-retention-7f3a"


def records(caplog: pytest.LogCaptureFixture, message: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == message]


async def test_a_purge_under_the_lock_deletes_before_the_cutoff_and_logs_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chats, lock = FakeExpiredChats(PurgeCounts(3, 11)), FakePurgeLock()
    caplog.set_level(logging.INFO)

    counts = await purge_expired(chats, lock, days=30, clock=lambda: NOW)

    assert counts == PurgeCounts(3, 11)
    assert chats.cutoffs == [NOW - timedelta(days=30)]
    assert (lock.entered, lock.exited) == (1, 1)
    (record,) = records(caplog, "retention purge")
    fields = {k: record.__dict__[k] for k in ("conversations", "messages", "retention_days")}
    assert fields == {"conversations": 3, "messages": 11, "retention_days": 30}


async def test_the_purge_runs_inside_the_lock() -> None:
    lock = FakePurgeLock()

    class InsideCheck(FakeExpiredChats):
        async def purge(self, updated_before: datetime) -> PurgeCounts:
            assert (lock.entered, lock.exited) == (1, 0)
            return await super().purge(updated_before)

    chats = InsideCheck()

    await purge_expired(chats, lock, days=30, clock=lambda: NOW)

    assert len(chats.cutoffs) == 1


async def test_without_the_lock_nothing_is_purged_and_the_skip_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chats = FakeExpiredChats()
    caplog.set_level(logging.INFO)

    counts = await purge_expired(chats, FakePurgeLock(available=False), days=30, clock=lambda: NOW)

    assert counts is None
    assert chats.cutoffs == []
    (record,) = records(caplog, "retention purge skipped")
    assert record.__dict__["reason"] == "lock_busy"
    assert records(caplog, "retention purge") == []


async def test_without_a_clock_the_cutoff_is_from_the_current_utc_time() -> None:
    chats = FakeExpiredChats()
    before = datetime.now(UTC)

    await purge_expired(chats, FakePurgeLock(), days=30)

    (cutoff,) = chats.cutoffs
    assert before - timedelta(days=30) <= cutoff <= datetime.now(UTC) - timedelta(days=30)


class Sleeps:
    """Records each pause; cancels the loop on the `stop_after`-th one."""

    def __init__(self, stop_after: int) -> None:
        self.seconds: list[float] = []
        self.stop_after = stop_after

    async def __call__(self, seconds: float) -> None:
        self.seconds.append(seconds)
        if len(self.seconds) == self.stop_after:
            raise asyncio.CancelledError


async def test_the_loop_purges_at_once_then_after_every_interval() -> None:
    runs: list[int] = []

    async def purge() -> None:
        runs.append(len(runs))

    sleeps = Sleeps(stop_after=3)
    with pytest.raises(asyncio.CancelledError):
        await purge_periodically(purge, interval_seconds=3600, sleep=sleeps)

    assert runs == [0, 1, 2]
    assert sleeps.seconds == [3600, 3600, 3600]


async def test_a_failed_purge_is_logged_by_type_only_and_the_loop_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts: list[int] = []

    async def purge() -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError(CANARY)

    caplog.set_level(logging.INFO)
    with pytest.raises(asyncio.CancelledError):
        await purge_periodically(purge, interval_seconds=60, sleep=Sleeps(stop_after=2))

    assert len(attempts) == 2
    (record,) = records(caplog, "retention purge failed")
    assert record.levelno == logging.WARNING
    assert record.__dict__["error_type"] == "ConnectionError"
    assert record.exc_info is None
    assert CANARY not in caplog.text


async def test_cancelling_the_loop_during_a_purge_stops_it() -> None:
    started = asyncio.Event()

    async def purge() -> None:
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(purge_periodically(purge, interval_seconds=60))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("seconds", [0, -1.0])
async def test_the_interval_must_be_positive(seconds: float) -> None:
    async def purge() -> None:
        raise AssertionError("never runs")

    with pytest.raises(ValueError, match="interval"):
        await purge_periodically(purge, interval_seconds=seconds)

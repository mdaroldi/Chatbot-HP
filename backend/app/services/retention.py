"""The retention purge: once on demand (`make purge`) or periodically in the backend.

Logs carry counts and codes only, never ids, titles or content: what was deleted is
personal data, and the point of deleting it is that nothing else keeps it.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import NoReturn

from app.domain.retention import ExpiredChats, PurgeCounts, PurgeLock, retention_cutoff

logger = logging.getLogger(__name__)

LOCK_BUSY = "lock_busy"


def _utc_now() -> datetime:
    return datetime.now(UTC)


async def purge_expired(
    chats: ExpiredChats,
    lock: PurgeLock,
    *,
    days: int,
    clock: Callable[[], datetime] = _utc_now,
) -> PurgeCounts | None:
    """Purge under the lock; None when another holder has it (that one is purging, and
    the next run catches whatever it missed).
    """
    async with lock.hold() as held:
        if not held:
            logger.info("retention purge skipped", extra={"reason": LOCK_BUSY})
            return None
        counts = await chats.purge(retention_cutoff(clock(), days=days))
    logger.info(
        "retention purge",
        extra={
            "conversations": counts.conversations,
            "messages": counts.messages,
            "retention_days": days,
        },
    )
    return counts


async def purge_periodically(
    purge: Callable[[], Awaitable[object]],
    *,
    interval_seconds: float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> NoReturn:
    """Purge at once, then after every interval, until cancelled. A failed purge (the
    database down, say) is logged by type and retried next time: retention is not
    worth stopping the backend for, and an unhandled error would end the loop for good.
    """
    if interval_seconds <= 0:
        raise ValueError("the purge interval must be positive")
    while True:
        try:
            await purge()
        except Exception as error:
            logger.warning("retention purge failed", extra={"error_type": type(error).__name__})
        await sleep(interval_seconds)

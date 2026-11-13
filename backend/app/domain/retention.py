"""Retention: conversations untouched for longer than the period are deleted.

Chats are personal data, so they are kept only as long as the retention period
(`CHAT_RETENTION_DAYS`) after their last activity (`updated_at`, which every new turn
bumps); messages go with their conversation. Several replicas may run the purge, so
it runs under a lock that only one holder gets at a time.
"""

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, Self


@dataclass(frozen=True)
class PurgeCounts:
    """What one purge deleted; the only facts about it that are logged."""

    conversations: int
    messages: int

    def __post_init__(self) -> None:
        if self.conversations < 0 or self.messages < 0:
            raise ValueError("purge counts cannot be negative")

    def __add__(self, other: Self) -> Self:
        return type(self)(self.conversations + other.conversations, self.messages + other.messages)

    @classmethod
    def none(cls) -> Self:
        return cls(0, 0)


class ExpiredChats(Protocol):
    async def purge(self, updated_before: datetime) -> PurgeCounts:
        """Delete every conversation (any client's) last updated before the cutoff."""
        ...


class PurgeLock(Protocol):
    def hold(self) -> AbstractAsyncContextManager[bool]:
        """Try to take the lock for the block: True when this holder got it, False when
        another one holds it. It is released when the block exits, however it exits.
        """
        ...


def retention_cutoff(now: datetime, *, days: int) -> datetime:
    """Conversations last updated before this instant have expired."""
    if days < 1:
        raise ValueError("the retention period must be at least one day")
    if now.tzinfo is None:
        raise ValueError("now must carry a timezone")
    return now - timedelta(days=days)

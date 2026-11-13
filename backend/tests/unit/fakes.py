"""In-memory fakes implementing domain interfaces (no network, no containers)."""

import asyncio
from dataclasses import dataclass


@dataclass
class FakeProbe:
    """DependencyProbe that succeeds, fails, waits or hangs on demand."""

    name: str
    error: BaseException | None = None
    delay_seconds: float = 0.0
    hang: bool = False
    info: dict[str, str] | None = None
    calls: int = 0

    async def ping(self) -> dict[str, str] | None:
        self.calls += 1
        if self.hang:
            await asyncio.Event().wait()
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.error is not None:
            raise self.error
        return self.info


class WordTokenCounter:
    """TokenCounter that counts whitespace-separated words; the real bge-small
    tokenizer arrives with the embedder, and its limit is checked by corpus tests.
    """

    def count(self, text: str) -> int:
        return len(text.split())

"""A scripted language model for service tests: fixed pieces, failures, stalls."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field

from app.domain.llm import GenerationRequest, GenerationStream, StopReason, Usage


@dataclass
class FakeStream:
    pieces: tuple[str, ...]
    stop: StopReason | None
    error: Exception | None
    fail_at: int
    stall: bool
    stall_at: int | None
    stalled: asyncio.Event
    _stop_reason: StopReason | None = None

    async def _block(self) -> None:
        self.stalled.set()
        await asyncio.Event().wait()

    async def _generate(self) -> AsyncIterator[str]:
        if self.stall:
            await self._block()
        for index, piece in enumerate(self.pieces):
            if self.error is not None and index == self.fail_at:
                raise self.error
            if index == self.stall_at:
                await self._block()
            yield piece
        self._stop_reason = self.stop

    def __aiter__(self) -> AsyncIterator[str]:
        return self._generate()

    @property
    def served_model(self) -> str | None:
        return "fake-served"

    @property
    def usage(self) -> Usage:
        return Usage(1, 1)

    @property
    def stop_reason(self) -> StopReason | None:
        return self._stop_reason


@dataclass
class FakeProvider:
    """Streams `pieces`; `error` is raised before piece `fail_at` (0: before any text);
    `stall` blocks before the first piece, `stall_at` before that piece, until the
    caller gives up; `stalled` is set once it blocks.
    """

    pieces: tuple[str, ...] = ()
    stop: StopReason | None = StopReason.END_TURN
    error: Exception | None = None
    fail_at: int = 0
    stall: bool = False
    stall_at: int | None = None
    stalled: asyncio.Event = field(default_factory=asyncio.Event)
    requests: list[GenerationRequest] = field(default_factory=list)
    closed: int = 0

    @property
    def name(self) -> str:
        return "fake"

    def stream(self, request: GenerationRequest) -> AbstractAsyncContextManager[GenerationStream]:
        self.requests.append(request)
        return self._open()

    @asynccontextmanager
    async def _open(self) -> AsyncIterator[GenerationStream]:
        try:
            yield FakeStream(
                self.pieces,
                self.stop,
                self.error,
                self.fail_at,
                self.stall,
                self.stall_at,
                self.stalled,
            )
        finally:
            self.closed += 1

"""The stub language model: deterministic, streaming, no network.

It is a product feature, not a test fake: `LLM_PROVIDER=stub` runs the whole chat
path (CI, integration tests, the Phase 6 load tests) without a key. Answers cite the
top excerpts of the prompt, a prompt without excerpts gets the refusal marker, and
outside production three directives in the user's text force a refusal or a failure,
so every outcome can be shown end to end. Only this module parses them.
"""

import asyncio
import math
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from app.domain.chat import REFUSAL_MARKER
from app.domain.llm import (
    GenerationRequest,
    GenerationStream,
    LLMUnavailableError,
    Purpose,
    StopReason,
    Usage,
)

Sleep = Callable[[float], Awaitable[None]]

SERVED_MODEL = "stub"
REWRITE_MAX_CHARS = 300
_CHARS_PER_TOKEN = 4

_PIECE = re.compile(r"\S+\s*")
# The prompt builder writes one `tag="S<n>"` attribute per excerpt.
_SECOND_EXCERPT = 'tag="S2"'
_FIRST_EXCERPT = 'tag="S1"'
_REFUSE, _FAIL_BEFORE, _FAIL_AFTER = "[stub:refuse]", "[stub:fail-before]", "[stub:fail-after]"

_ANSWER = "This is a deterministic stub answer from the retrieved excerpts [S1]."
_SECOND = " A second sentence cites the next excerpt [S2]."


def _tokens(chars: int) -> int:
    return math.ceil(chars / _CHARS_PER_TOKEN)


def _rewrite(text: str) -> str:
    line = " ".join(text.split())
    if len(line) <= REWRITE_MAX_CHARS:
        return line
    return line[:REWRITE_MAX_CHARS].rsplit(" ", 1)[0]


def _marker_pieces() -> list[str]:
    third = len(REFUSAL_MARKER) // 3
    return [REFUSAL_MARKER[:third], REFUSAL_MARKER[third : 2 * third], REFUSAL_MARKER[2 * third :]]


class StubStream:
    """One scripted answer; a failure directive raises at its point in the script."""

    def __init__(
        self,
        pieces: list[str],
        *,
        input_tokens: int,
        fail_at: int | None,
        delay_seconds: float,
        sleep: Sleep,
    ) -> None:
        self._pieces, self._fail_at = pieces, fail_at
        self._delay, self._sleep = delay_seconds, sleep
        self._input_tokens, self._output_chars = input_tokens, 0
        self._stop: StopReason | None = None
        self._closed = False

    async def _generate(self) -> AsyncIterator[str]:
        for index, piece in enumerate(self._pieces):
            if index == self._fail_at:
                raise LLMUnavailableError
            if self._delay:
                await self._sleep(self._delay)
            self._output_chars += len(piece)
            yield piece
        self._stop = StopReason.END_TURN

    def __aiter__(self) -> AsyncIterator[str]:
        return self._generate()

    @property
    def served_model(self) -> str | None:
        return SERVED_MODEL

    @property
    def usage(self) -> Usage:
        return Usage(self._input_tokens, _tokens(self._output_chars))

    @property
    def stop_reason(self) -> StopReason | None:
        return self._stop

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True


class StubProvider:
    def __init__(
        self, *, honour_directives: bool, token_delay_ms: float = 0, sleep: Sleep = asyncio.sleep
    ) -> None:
        self._honour = honour_directives
        self._delay, self._sleep = token_delay_ms / 1000, sleep

    @property
    def name(self) -> str:
        return SERVED_MODEL

    def stream(self, request: GenerationRequest) -> AbstractAsyncContextManager[GenerationStream]:
        return self._open(request)

    @asynccontextmanager
    async def _open(self, request: GenerationRequest) -> AsyncIterator[GenerationStream]:
        pieces, fail_at = self._script(request)
        prompt_chars = len(request.system) + sum(len(m.text) for m in request.messages)
        stream = StubStream(
            pieces,
            input_tokens=_tokens(prompt_chars),
            fail_at=fail_at,
            delay_seconds=self._delay,
            sleep=self._sleep,
        )
        try:
            yield stream
        finally:
            stream.close()

    def _script(self, request: GenerationRequest) -> tuple[list[str], int | None]:
        last = request.messages[-1].text
        if request.purpose is Purpose.REWRITE:
            return _PIECE.findall(_rewrite(last)), None
        directive = self._directive(last) if self._honour else None
        if directive == _REFUSE or _FIRST_EXCERPT not in last:
            return _marker_pieces(), None
        answer = _ANSWER + (_SECOND if _SECOND_EXCERPT in last else "")
        fail_at = {_FAIL_BEFORE: 0, _FAIL_AFTER: 1}.get(directive) if directive else None
        return _PIECE.findall(answer), fail_at

    @staticmethod
    def _directive(text: str) -> str | None:
        return next((d for d in (_REFUSE, _FAIL_BEFORE, _FAIL_AFTER) if d in text), None)

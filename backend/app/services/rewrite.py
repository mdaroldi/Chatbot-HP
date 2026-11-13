"""Follow-up rewriting: a standalone search query, or the raw message as a fallback.

The rewrite only improves retrieval, so it never fails a request: a provider error,
the timeout, a stream that did not end its turn, or an output that is not one short
line all fall back to the user's own message, and the reason is logged as a code.
Anything else (a bug, a cancellation) propagates.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.domain.chat import Message, Role
from app.domain.llm import (
    GenerationRequest,
    LLMError,
    LLMProvider,
    PromptMessage,
    Purpose,
    StopReason,
)
from app.domain.prompt import HistoryBudget
from app.domain.rewrite import accept_rewrite, rewrite_system_prompt, rewrite_turns

logger = logging.getLogger(__name__)

_REWRITTEN = "rewritten"


class RewriteFallback(StrEnum):
    LLM_ERROR = "llm_error"
    TIMEOUT = "timeout"
    INCOMPLETE = "incomplete"
    INVALID_OUTPUT = "invalid_output"


@dataclass(frozen=True)
class RewriteConfig:
    model: str
    max_tokens: int
    effort: str | None
    timeout_seconds: float
    budget: HistoryBudget

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True)
class RewriteResult:
    """`query` is what retrieval searches; `rewritten_query` and `rewrite_ms` are
    stored with the answer (None when no rewrite was produced or none was tried).
    """

    query: str
    rewritten_query: str | None
    rewrite_ms: float | None
    fallback: RewriteFallback | None


class QueryRewriter:
    def __init__(
        self,
        provider: LLMProvider,
        config: RewriteConfig,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._provider, self._config, self._clock = provider, config, clock

    async def rewrite(self, question: str, history: Sequence[Message]) -> RewriteResult:
        """`history` is the conversation before `question`; with no finished turn in
        the budget there is nothing to resolve, so the model is not called.
        """
        turns = rewrite_turns(history, self._config.budget)
        if not turns:
            return RewriteResult(question, None, None, None)
        request = GenerationRequest(
            purpose=Purpose.REWRITE,
            model=self._config.model,
            system=rewrite_system_prompt(turns),
            messages=(PromptMessage(Role.USER, question),),
            max_tokens=self._config.max_tokens,
            effort=self._config.effort,
        )
        start = self._clock()
        query, fallback = await self._generate(request)
        elapsed_ms = (self._clock() - start) * 1000
        logger.info(
            "query rewrite",
            extra={
                "outcome": fallback or _REWRITTEN,
                "rewrite_ms": elapsed_ms,
                "history_turns": len(turns),
            },
        )
        if query is None:
            return RewriteResult(question, None, elapsed_ms, fallback)
        return RewriteResult(query, query, elapsed_ms, None)

    async def _generate(
        self, request: GenerationRequest
    ) -> tuple[str | None, RewriteFallback | None]:
        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                async with self._provider.stream(request) as stream:
                    text = "".join([piece async for piece in stream])
                    stop = stream.stop_reason
        except LLMError:
            return None, RewriteFallback.LLM_ERROR
        except TimeoutError:
            return None, RewriteFallback.TIMEOUT
        if stop is not StopReason.END_TURN:
            return None, RewriteFallback.INCOMPLETE
        query = accept_rewrite(text)
        return query, None if query is not None else RewriteFallback.INVALID_OUTPUT

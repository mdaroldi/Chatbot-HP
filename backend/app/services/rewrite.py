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

from app.core.settings import Settings
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
from app.domain.rewrite import (
    RewriteCheck,
    RewriteRejection,
    check_rewrite,
    rewrite_system_prompt,
    rewrite_turns,
)

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
    # Why an invalid output was rejected (a code); None unless fallback is invalid_output.
    rejection: RewriteRejection | None = None


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
        fallback, check = await self._generate(request)
        elapsed_ms = (self._clock() - start) * 1000
        extra: dict[str, object] = {
            "outcome": fallback or _REWRITTEN,
            "rewrite_ms": elapsed_ms,
            "history_turns": len(turns),
        }
        rejection = None if check is None else check.rejection
        if check is not None and rejection is not None:
            # Sizes only: they tell a label or a second line from an overlong query.
            extra |= {
                "rejection": rejection.value,
                "output_chars": check.chars,
                "output_words": check.words,
                "output_lines": check.lines,
            }
        logger.info("query rewrite", extra=extra)
        if check is None or check.query is None:
            return RewriteResult(question, None, elapsed_ms, fallback, rejection)
        return RewriteResult(check.query, check.query, elapsed_ms, None)

    async def _generate(
        self, request: GenerationRequest
    ) -> tuple[RewriteFallback | None, RewriteCheck | None]:
        """The fallback, if any, and the output check when the model finished its turn."""
        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                async with self._provider.stream(request) as stream:
                    text = "".join([piece async for piece in stream])
                    stop = stream.stop_reason
        except LLMError:
            return RewriteFallback.LLM_ERROR, None
        except TimeoutError:
            return RewriteFallback.TIMEOUT, None
        if stop is not StopReason.END_TURN:
            return RewriteFallback.INCOMPLETE, None
        check = check_rewrite(text)
        return (None if check.query is not None else RewriteFallback.INVALID_OUTPUT), check


def rewrite_config_from_settings(settings: Settings) -> RewriteConfig:
    return RewriteConfig(
        model=settings.llm_rewrite_model,
        max_tokens=settings.llm_rewrite_max_tokens,
        effort=settings.llm_rewrite_effort,
        timeout_seconds=settings.llm_rewrite_timeout_seconds,
        budget=HistoryBudget(
            max_turns=settings.chat_rewrite_history_turns,
            max_tokens=settings.chat_rewrite_history_tokens,
        ),
    )

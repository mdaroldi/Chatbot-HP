"""The language-model port (pure, no I/O): requests, streamed answers and failures.

Providers (the deterministic stub, the Anthropic adapter) stream text pieces and report
usage, the model that actually served the request and why it stopped. Failures are
domain errors with fixed messages, so no upstream text can reach a client or a log.
"""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Protocol

from app.domain.chat import Role


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class Purpose(StrEnum):
    ANSWER = "answer"
    REWRITE = "rewrite"


class StopReason(StrEnum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    OTHER = "other"


@dataclass(frozen=True)
class PromptMessage:
    role: Role
    text: str

    def __post_init__(self) -> None:
        _require(bool(self.text), "a prompt message has text")


@dataclass(frozen=True)
class GenerationRequest:
    purpose: Purpose
    model: str
    system: str
    messages: tuple[PromptMessage, ...]
    max_tokens: int
    effort: str | None = None

    def __post_init__(self) -> None:
        _require(bool(self.model), "a request names a model")
        _require(self.max_tokens >= 1, "max_tokens must be at least 1")
        _require(bool(self.messages), "a request needs at least one message")
        _require(self.messages[0].role is Role.USER, "the first message must be the user's")
        _require(self.messages[-1].role is Role.USER, "the last message must be the user's")
        _require(
            all(a.role is not b.role for a, b in pairwise(self.messages)),
            "roles must alternate",
        )


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None
    output_tokens: int | None


class LLMError(Exception):
    """Base of the provider failures; subclasses fix the code and the message.

    The constructor takes no message on purpose: whatever the upstream said (which can
    echo the prompt) stays out of the exception, its logs and the HTTP response.
    """

    code = "llm_error"
    message = "The language model failed."

    def __init__(self) -> None:
        super().__init__(self.message)


class LLMUnavailableError(LLMError):
    """Transient: the provider is down, overloaded, rate-limited or unreachable."""

    code = "llm_unavailable"
    message = "The language model is unavailable."


class LLMRequestError(LLMError):
    """Permanent: the provider rejected the request (a configuration problem)."""

    code = "llm_error"
    message = "The language model rejected the request."


class LLMTimeoutError(LLMError):
    """No visible text arrived within the first-token deadline."""

    code = "llm_timeout"
    message = "The language model did not answer in time."


class GenerationStream(Protocol):
    """Text pieces of one answer; usage and the stop reason fill in as it streams, so a
    cancelled or failed stream still reports what was known.
    """

    def __aiter__(self) -> AsyncIterator[str]: ...

    @property
    def served_model(self) -> str | None: ...

    @property
    def usage(self) -> Usage: ...

    @property
    def stop_reason(self) -> StopReason | None: ...


class LLMProvider(Protocol):
    """Opening the context starts the request; leaving it, even early or on
    cancellation, closes the upstream call.
    """

    @property
    def name(self) -> str: ...

    def stream(
        self, request: GenerationRequest
    ) -> AbstractAsyncContextManager[GenerationStream]: ...

"""The Anthropic adapter: one streamed Messages API call per generation request.

The client gets the key and the public API URL explicitly, so no ambient credential
or endpoint (`ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, profiles) is ever used.
Answers send the effort and, behind a setting, the server-side refusal fallback;
neither sends `thinking` (Claude Opus 5.5 thinks adaptively and cannot turn it off)
nor any sampling parameter (rejected by API version 2026-10-21). Rewrites turn
thinking off, for a quick one-line query.

Deadlines: the first answer text must arrive within the first-token timeout of the
request, the whole answer within the total timeout. Each wait is bounded on its own
(`asyncio.timeout_at`), because the chat endpoint reads every event in a fresh task:
a deadline tied to the opening task would not follow the stream. The SDK retries
before the stream opens (`max_retries`); once it is open, nothing is retried, since
the user may already see text.

Every failure becomes a domain error with a fixed message, raised `from None`: no
upstream text reaches a client or a log. The upstream response is closed when the
provider context exits, however it exits, not by the piece iterator.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any

import anthropic
import httpx2
from anthropic import AsyncAnthropic, AsyncStream
from pydantic import SecretStr

from app.core.settings import Settings
from app.domain.llm import (
    GenerationRequest,
    GenerationStream,
    LLMError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
    Purpose,
    StopReason,
    Usage,
)

API_URL = "https://api.anthropic.com"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
NAME = "anthropic"

_STOP_REASONS = {
    "end_turn": StopReason.END_TURN,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
}
# Worth trying again later: the service, not the request, is the problem.
_TRANSIENT_STATUS = frozenset({408, 409, 429})
# An error event inside an open stream carries the stream's own status.
_IN_STREAM = 200
_TRANSIENT_TYPES = frozenset({"overloaded_error", "api_error", "rate_limit_error", "timeout_error"})


@dataclass(frozen=True)
class AnthropicConfig:
    api_key: SecretStr
    connect_timeout_seconds: float
    first_token_timeout_seconds: float
    total_timeout_seconds: float
    max_retries: int
    refusal_fallback: bool


def anthropic_config_from_settings(settings: Settings) -> AnthropicConfig:
    key = settings.anthropic_api_key
    if key is None:
        # Settings already reject this for LLM_PROVIDER=anthropic; other callers too.
        raise ValueError("the Anthropic provider needs ANTHROPIC_API_KEY")
    return AnthropicConfig(
        api_key=key,
        connect_timeout_seconds=settings.llm_connect_timeout_seconds,
        first_token_timeout_seconds=settings.llm_first_token_timeout_seconds,
        total_timeout_seconds=settings.llm_total_timeout_seconds,
        max_retries=settings.llm_max_retries,
        refusal_fallback=settings.llm_refusal_fallback_enabled,
    )


def _error_type(error: anthropic.APIStatusError) -> str | None:
    body = error.body
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        kind = body["error"].get("type")
        return kind if isinstance(kind, str) else None
    return None


def domain_error(error: anthropic.AnthropicError) -> LLMError:
    """The fixed-message error for an SDK failure: the HTTP status decides, except for
    an error event inside an open stream (status 200), where its error type does.
    """
    if isinstance(error, anthropic.APIConnectionError):  # includes the SDK's timeouts
        return LLMUnavailableError()
    if isinstance(error, anthropic.APIStatusError):
        status = error.status_code
        if status == _IN_STREAM:
            transient = _error_type(error) in _TRANSIENT_TYPES
        else:
            transient = status >= 500 or status in _TRANSIENT_STATUS
        return LLMUnavailableError() if transient else LLMRequestError()
    return LLMError()


class _Deadlines:
    def __init__(self, config: AnthropicConfig) -> None:
        now = asyncio.get_running_loop().time()
        self.first_token = now + config.first_token_timeout_seconds
        self.total = now + config.total_timeout_seconds

    def next(self, text_seen: bool) -> float:
        return self.total if text_seen else min(self.first_token, self.total)


class AnthropicStream:
    """Text deltas of one answer; thinking and other blocks are skipped.

    A plain iterator, not an async generator, so nothing of it is left to the garbage
    collector; closing the upstream is the provider context's job.
    """

    def __init__(self, events: AsyncStream[Any], deadlines: _Deadlines) -> None:
        # Any: the SDK's beta stream event union; each event is read by its `type`.
        self._events, self._deadlines = events, deadlines
        self._model: str | None = None
        self._usage = Usage(None, None)
        self._stop: StopReason | None = None
        self._text_seen = False

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        while True:
            event = await self._next_event()
            text = self._apply(event)
            if text:
                self._text_seen = True
                return text

    async def _next_event(self) -> Any:  # Any: one beta stream event
        try:
            async with asyncio.timeout_at(self._deadlines.next(self._text_seen)):
                return await anext(self._events)
        except TimeoutError:
            raise LLMTimeoutError from None
        except anthropic.AnthropicError as error:
            raise domain_error(error) from None

    def _apply(self, event: Any) -> str | None:  # Any: one beta stream event
        kind = event.type
        if kind == "message_start":
            message = event.message
            self._model = message.model
            self._usage = Usage(message.usage.input_tokens, message.usage.output_tokens)
        elif kind == "content_block_start" and event.content_block.type == "fallback":
            self._model = event.content_block.to.model  # the model now writing
        elif kind == "content_block_delta" and event.delta.type == "text_delta":
            return str(event.delta.text)
        elif kind == "message_delta":
            reason = event.delta.stop_reason
            if reason is not None:
                self._stop = _STOP_REASONS.get(reason, StopReason.OTHER)
            self._usage = Usage(self._usage.input_tokens, event.usage.output_tokens)
        return None

    @property
    def served_model(self) -> str | None:
        return self._model

    @property
    def usage(self) -> Usage:
        return self._usage

    @property
    def stop_reason(self) -> StopReason | None:
        return self._stop


class AnthropicProvider:
    def __init__(
        self, config: AnthropicConfig, *, http_client: httpx2.AsyncClient | None = None
    ) -> None:
        self._config = config
        self.client = AsyncAnthropic(
            api_key=config.api_key.get_secret_value(),
            base_url=API_URL,
            max_retries=config.max_retries,
            timeout=httpx2.Timeout(
                config.total_timeout_seconds, connect=config.connect_timeout_seconds
            ),
            http_client=http_client,
        )

    @property
    def name(self) -> str:
        return NAME

    def stream(self, request: GenerationRequest) -> AbstractAsyncContextManager[GenerationStream]:
        return self._open(request)

    async def aclose(self) -> None:
        await self.client.close()

    @asynccontextmanager
    async def _open(self, request: GenerationRequest) -> AsyncIterator[GenerationStream]:
        deadlines = _Deadlines(self._config)
        try:
            async with asyncio.timeout_at(deadlines.next(text_seen=False)):
                events = await self.client.beta.messages.create(**self._body(request))
        except TimeoutError:
            raise LLMTimeoutError from None
        except anthropic.AnthropicError as error:
            raise domain_error(error) from None
        try:
            yield AnthropicStream(events, deadlines)
        finally:
            await events.close()

    def _body(self, request: GenerationRequest) -> dict[str, Any]:
        # Any: keyword arguments of the SDK call, which types each one itself.
        body: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": [{"role": m.role.value, "content": m.text} for m in request.messages],
            "stream": True,
        }
        if request.effort is not None:
            body["output_config"] = {"effort": request.effort}
        if request.purpose is Purpose.REWRITE:
            body["thinking"] = {"type": "disabled"}
        elif self._config.refusal_fallback:
            body["betas"] = [FALLBACK_BETA]
            body["fallbacks"] = "default"
        return body

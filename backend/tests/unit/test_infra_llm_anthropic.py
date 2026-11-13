"""The Anthropic adapter over a fake Messages API (httpx2.MockTransport): the request
it sends, the stream it reads, deadlines, retries, error mapping and closing.

No test reaches the network; the key is a placeholder.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import SecretStr

from app.core.settings import Settings
from app.domain.chat import Role
from app.domain.llm import (
    GenerationRequest,
    GenerationStream,
    LLMError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
    PromptMessage,
    Purpose,
    StopReason,
    Usage,
)
from app.infra.llm_anthropic import (
    API_URL,
    FALLBACK_BETA,
    AnthropicConfig,
    AnthropicProvider,
    anthropic_config_from_settings,
    domain_error,
)
from tests.unit.conftest import SettingsFactory

KEY = "placeholder-not-a-key"  # not a real key
UPSTREAM_TEXT = "UPSTREAM-5e1d-detail"  # text only the fake API ever says
Handler = Callable[[httpx2.Request], Any]  # Any: a Response or an awaitable of one


def config(**overrides: Any) -> AnthropicConfig:  # Any: per-test field overrides
    fields: dict[str, Any] = {
        "api_key": SecretStr(KEY),
        "connect_timeout_seconds": 5.0,
        "first_token_timeout_seconds": 5.0,
        "total_timeout_seconds": 10.0,
        "max_retries": 0,
        "refusal_fallback": True,
    }
    fields.update(overrides)
    return AnthropicConfig(**fields)


def request(
    purpose: Purpose = Purpose.ANSWER, text: str = "Question?", effort: str | None = "low"
) -> GenerationRequest:
    return GenerationRequest(
        purpose=purpose,
        model="configured-model",
        system="System prompt.",
        messages=(
            PromptMessage(Role.USER, "Earlier?"),
            PromptMessage(Role.ASSISTANT, "Earlier answer."),
            PromptMessage(Role.USER, text),
        ),
        max_tokens=900,
        effort=effort,
    )


# --- A fake Messages API ----------------------------------------------------------------


def sse(events: Iterable[dict[str, Any]]) -> bytes:  # Any: JSON event payloads
    return b"".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events
    )


def message_start(model: str = "claude-served-1", input_tokens: int = 42) -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 1},
        },
    }


def block_start(index: int, block: dict[str, Any]) -> dict[str, Any]:
    return {"type": "content_block_start", "index": index, "content_block": block}


def text_delta(index: int, text: str) -> dict[str, Any]:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }


def thinking_delta(index: int, text: str) -> dict[str, Any]:
    delta = {"type": "thinking_delta", "thinking": text}
    return {"type": "content_block_delta", "index": index, "delta": delta}


def block_stop(index: int) -> dict[str, Any]:
    return {"type": "content_block_stop", "index": index}


def message_delta(stop_reason: str | None, output_tokens: int = 7) -> dict[str, Any]:
    return {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }


MESSAGE_STOP = {"type": "message_stop"}
TEXT_BLOCK = {"type": "text", "text": ""}


def answer_events(
    pieces: Iterable[str] = ("Hello ", "world [S1]."), stop: str | None = "end_turn"
) -> list[dict[str, Any]]:
    return [
        message_start(),
        block_start(0, {"type": "thinking", "thinking": "", "signature": "sig"}),
        thinking_delta(0, "hidden reasoning"),
        block_stop(0),
        block_start(1, TEXT_BLOCK),
        *(text_delta(1, piece) for piece in pieces),
        block_stop(1),
        message_delta(stop),
        MESSAGE_STOP,
    ]


class Body(httpx2.AsyncByteStream):
    """A response body that can stall after some chunks and records being closed."""

    def __init__(self, chunks: list[bytes], stall_after: int | None = None) -> None:
        self.chunks, self.stall_after = chunks, stall_after
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index, chunk in enumerate(self.chunks):
            if index == self.stall_after:
                await asyncio.Event().wait()
            yield chunk
        if self.stall_after == len(self.chunks):
            await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


def ok(body: Body) -> httpx2.Response:
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=body)


def error_response(status: int, kind: str = "api_error") -> httpx2.Response:
    payload = {"type": "error", "error": {"type": kind, "message": UPSTREAM_TEXT}}
    return httpx2.Response(status, json=payload)


@dataclass
class FakeAPI:
    """Answers each call with the next response (the last one repeats)."""

    responses: list[httpx2.Response | Exception | Callable[[], Any]]
    requests: list[httpx2.Request] = field(default_factory=list)

    async def handle(self, call: httpx2.Request) -> httpx2.Response:
        self.requests.append(call)
        index = min(len(self.requests), len(self.responses)) - 1
        response = self.responses[index]
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return await response()  # type: ignore[no-any-return]  # test hooks
        return response

    def body(self, index: int = -1) -> dict[str, Any]:  # Any: the JSON request body
        loaded: dict[str, Any] = json.loads(self.requests[index].content)
        return loaded


def provider(api: FakeAPI, **overrides: Any) -> "Recording":  # Any: config fields
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(api.handle))
    return Recording(config(**overrides), http_client=client)


async def read(stream: GenerationStream) -> list[str]:
    return [piece async for piece in stream]


async def answer(
    api: FakeAPI, req: GenerationRequest | None = None, **overrides: Any
) -> tuple[list[str], GenerationStream]:  # Any: config fields
    async with provider(api, **overrides).stream(req or request()) as stream:
        pieces = await read(stream)
    return pieces, stream


class Recording(AnthropicProvider):
    """Keeps the last stream it opened, to read its usage after a failure."""

    last: GenerationStream

    @asynccontextmanager
    async def _open(self, request: GenerationRequest) -> AsyncIterator[GenerationStream]:
        async with super()._open(request) as stream:
            self.last = stream
            yield stream


async def consume(adapter: AnthropicProvider, seen: list[str]) -> None:
    async with adapter.stream(request()) as stream:
        async for piece in stream:
            seen.append(piece)


def streaming(*events: dict[str, Any]) -> FakeAPI:
    return FakeAPI([ok(Body([sse(events)]))])


# --- The request ------------------------------------------------------------------------


async def test_the_answer_request_sends_effort_and_fallbacks_and_no_sampling_or_thinking() -> None:
    api = streaming(*answer_events())

    await answer(api)

    body = api.body()
    assert body["model"] == "configured-model"
    assert body["max_tokens"] == 900
    assert body["system"] == "System prompt."
    assert body["messages"] == [
        {"role": "user", "content": "Earlier?"},
        {"role": "assistant", "content": "Earlier answer."},
        {"role": "user", "content": "Question?"},
    ]
    assert body["stream"] is True
    assert body["output_config"] == {"effort": "low"}
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA in api.requests[0].headers["anthropic-beta"].split(",")
    for absent in ("thinking", "temperature", "top_p", "top_k"):
        assert absent not in body


async def test_fallbacks_can_be_turned_off() -> None:
    api = streaming(*answer_events())

    await answer(api, refusal_fallback=False)

    assert "fallbacks" not in api.body()
    assert "anthropic-beta" not in api.requests[0].headers


async def test_no_effort_sends_no_output_config() -> None:
    api = streaming(*answer_events())

    await answer(api, request(effort=None))

    assert "output_config" not in api.body()


async def test_a_rewrite_disables_thinking_and_asks_for_no_fallback() -> None:
    """The rewrite model is small and fast (Haiku 5.5 allows disabled thinking at low
    effort and has no server-side fallback); a failed rewrite falls back to the raw
    question anyway.
    """
    api = streaming(*answer_events())

    await answer(api, request(Purpose.REWRITE))

    body = api.body()
    assert body["thinking"] == {"type": "disabled"}
    assert "fallbacks" not in body
    assert "anthropic-beta" not in api.requests[0].headers
    for absent in ("temperature", "top_p", "top_k"):
        assert absent not in body


async def test_stub_directives_are_sent_as_ordinary_text() -> None:
    api = streaming(*answer_events())
    text = "[stub:refuse] [stub:fail-before] [stub:fail-after] How?"

    pieces, _ = await answer(api, request(text=text))

    assert api.body()["messages"][-1]["content"] == text
    assert "".join(pieces) == "Hello world [S1]."


async def test_the_configured_key_and_the_public_url_are_used_whatever_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://ambient.invalid")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    api = streaming(*answer_events())

    await answer(api)

    sent = api.requests[0]
    assert str(sent.url).startswith(f"{API_URL}/v1/messages")
    # SDK 2.x sends the API key as a bearer credential.
    assert sent.headers["authorization"] == f"Bearer {KEY}"
    assert "x-api-key" not in sent.headers
    assert not [value for value in sent.headers.values() if "ambient" in value]


def test_the_provider_is_named_for_readiness() -> None:
    assert provider(FakeAPI([])).name == "anthropic"


def test_settings_become_the_client_config(make_settings: SettingsFactory) -> None:
    settings: Settings = make_settings(
        llm_provider="anthropic",
        anthropic_api_key=SecretStr(KEY),
        llm_connect_timeout_seconds=4.0,
        llm_first_token_timeout_seconds=25.0,
        llm_total_timeout_seconds=100.0,
        llm_max_retries=3,
        llm_refusal_fallback_enabled=False,
    )

    assert anthropic_config_from_settings(settings) == AnthropicConfig(
        api_key=SecretStr(KEY),
        connect_timeout_seconds=4.0,
        first_token_timeout_seconds=25.0,
        total_timeout_seconds=100.0,
        max_retries=3,
        refusal_fallback=False,
    )


def test_settings_without_a_key_are_not_an_anthropic_config(make_settings: SettingsFactory) -> None:
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        anthropic_config_from_settings(make_settings())


def test_the_client_uses_the_configured_retries_and_connect_timeout() -> None:
    client = AnthropicProvider(config(max_retries=2, connect_timeout_seconds=5.0)).client

    assert client.max_retries == 2
    assert isinstance(client.timeout, httpx2.Timeout)
    assert client.timeout.connect == 5.0


# --- The stream -------------------------------------------------------------------------


async def test_only_text_reaches_the_caller_with_model_usage_and_stop_reason() -> None:
    pieces, stream = await answer(streaming(*answer_events()))

    assert pieces == ["Hello ", "world [S1]."]
    assert stream.served_model == "claude-served-1"
    assert stream.usage == Usage(42, 7)
    assert stream.stop_reason is StopReason.END_TURN


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        ("end_turn", StopReason.END_TURN),
        ("max_tokens", StopReason.MAX_TOKENS),
        ("refusal", StopReason.REFUSAL),
        ("stop_sequence", StopReason.OTHER),
        ("pause_turn", StopReason.OTHER),
        ("model_context_window_exceeded", StopReason.OTHER),
    ],
)
async def test_stop_reasons_map_faithfully(wire: str, expected: StopReason) -> None:
    _, stream = await answer(streaming(*answer_events(stop=wire)))

    assert stream.stop_reason is expected


async def test_a_stream_that_never_reports_a_stop_reason_has_none() -> None:
    events = [message_start(), block_start(0, TEXT_BLOCK), text_delta(0, "Hi."), MESSAGE_STOP]

    _, stream = await answer(streaming(*events))

    assert stream.stop_reason is None


async def test_a_server_side_fallback_names_the_model_that_served() -> None:
    fallback = {
        "type": "fallback",
        "from": {"model": "configured-model"},
        "to": {"model": "claude-fallback-1"},
        "trigger": {"type": "refusal"},
    }
    events = [
        message_start(model="configured-model"),
        block_start(0, TEXT_BLOCK),
        text_delta(0, "Partial "),
        block_stop(0),
        block_start(1, fallback),
        block_stop(1),
        block_start(2, TEXT_BLOCK),
        text_delta(2, "continued."),
        block_stop(2),
        message_delta("end_turn"),
        MESSAGE_STOP,
    ]

    pieces, stream = await answer(streaming(*events))

    assert pieces == ["Partial ", "continued."]
    assert stream.served_model == "claude-fallback-1"


# --- Failures ---------------------------------------------------------------------------


def assert_fixed(error: LLMError, kind: type[LLMError]) -> None:
    assert type(error) is kind
    assert str(error) == kind.message
    assert error.__cause__ is None
    assert error.__suppress_context__ is True
    assert UPSTREAM_TEXT not in repr(error)


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (429, LLMUnavailableError),
        (500, LLMUnavailableError),
        (503, LLMUnavailableError),
        (529, LLMUnavailableError),
        (400, LLMRequestError),
        (401, LLMRequestError),
        (403, LLMRequestError),
        (404, LLMRequestError),
        (413, LLMRequestError),
    ],
)
async def test_http_errors_before_the_stream_map_to_fixed_errors(
    status: int, kind: type[LLMError]
) -> None:
    api = FakeAPI([error_response(status)])

    with pytest.raises(LLMError) as raised:
        await answer(api)

    assert_fixed(raised.value, kind)


async def test_a_connection_failure_is_unavailable() -> None:
    api = FakeAPI([httpx2.ConnectError(UPSTREAM_TEXT)])

    with pytest.raises(LLMError) as raised:
        await answer(api)

    assert_fixed(raised.value, LLMUnavailableError)


async def test_retryable_errors_are_retried_before_the_stream_opens() -> None:
    api = FakeAPI([error_response(529, "overloaded_error"), ok(Body([sse(answer_events())]))])

    pieces, _ = await answer(api, max_retries=1)

    assert len(api.requests) == 2
    assert pieces == ["Hello ", "world [S1]."]


async def test_an_error_event_mid_stream_is_unavailable_and_keeps_the_usage() -> None:
    overloaded = {
        "type": "error",
        "error": {"type": "overloaded_error", "message": UPSTREAM_TEXT},
    }
    api = streaming(message_start(), block_start(0, TEXT_BLOCK), text_delta(0, "Part"), overloaded)
    seen: list[str] = []

    adapter = provider(api)
    with pytest.raises(LLMError) as raised:
        await consume(adapter, seen)

    assert_fixed(raised.value, LLMUnavailableError)
    assert seen == ["Part"]
    assert adapter.last.usage == Usage(42, 1)
    assert adapter.last.served_model == "claude-served-1"


async def test_no_first_token_within_its_deadline_is_a_timeout() -> None:
    body = Body([sse([message_start(), block_start(0, TEXT_BLOCK)])], stall_after=1)

    with pytest.raises(LLMError) as raised:
        await answer(FakeAPI([ok(body)]), first_token_timeout_seconds=0.05)

    assert_fixed(raised.value, LLMTimeoutError)
    assert body.closed


async def test_a_response_that_never_starts_is_a_first_token_timeout() -> None:
    async def never() -> httpx2.Response:
        await asyncio.Event().wait()
        return httpx2.Response(200)

    with pytest.raises(LLMError) as raised:
        await answer(FakeAPI([never]), first_token_timeout_seconds=0.05)

    assert_fixed(raised.value, LLMTimeoutError)


async def test_after_the_first_token_only_the_total_deadline_applies() -> None:
    """A gap longer than the first-token deadline is fine once text arrived; a stall
    is cut at the total deadline.
    """
    first = sse([message_start(), block_start(0, TEXT_BLOCK), text_delta(0, "Early.")])

    class SlowThenStall(Body):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield first
            await asyncio.sleep(0.15)
            yield sse([text_delta(0, " Late.")])
            await asyncio.Event().wait()

    body = SlowThenStall([])
    seen: list[str] = []
    adapter = provider(
        FakeAPI([ok(body)]), first_token_timeout_seconds=0.1, total_timeout_seconds=0.4
    )
    with pytest.raises(LLMError) as raised:
        await consume(adapter, seen)

    assert_fixed(raised.value, LLMTimeoutError)
    assert seen == ["Early.", " Late."]
    assert body.closed


async def test_the_deadlines_hold_when_each_piece_is_read_from_another_task() -> None:
    """The chat endpoint reads each event in its own task, so a deadline cannot be
    tied to the task that opened the stream.
    """
    body = Body([sse([message_start(), block_start(0, TEXT_BLOCK), text_delta(0, "One.")])], 1)
    async with provider(
        FakeAPI([ok(body)]), first_token_timeout_seconds=0.1, total_timeout_seconds=0.2
    ).stream(request()) as stream:
        pieces = aiter(stream)
        assert await asyncio.ensure_future(anext(pieces)) == "One."
        with pytest.raises(LLMTimeoutError):
            await asyncio.ensure_future(anext(pieces))


# --- Closing ----------------------------------------------------------------------------


async def test_leaving_the_context_early_closes_the_upstream_at_once() -> None:
    body = Body([sse(answer_events())])

    async with provider(FakeAPI([ok(body)])).stream(request()) as stream:
        async for _ in stream:
            break
        assert not body.closed

    assert body.closed


async def test_a_cancelled_reader_closes_the_upstream() -> None:
    body = Body([sse([message_start(), block_start(0, TEXT_BLOCK)])], stall_after=1)
    opened = asyncio.Event()

    async def reader() -> None:
        async with provider(FakeAPI([ok(body)])).stream(request()) as stream:
            opened.set()
            await read(stream)

    task = asyncio.create_task(reader())
    async with asyncio.timeout(5):  # a reader that fails to open must fail, not hang
        await asyncio.wait(
            {task, asyncio.ensure_future(opened.wait())}, return_when=asyncio.FIRST_COMPLETED
        )
    assert opened.is_set()
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert body.closed


async def test_the_provider_closes_its_client() -> None:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(FakeAPI([]).handle))
    adapter = AnthropicProvider(config(), http_client=client)

    await adapter.aclose()

    assert client.is_closed


async def test_a_message_delta_without_a_stop_reason_leaves_it_unknown() -> None:
    events = [message_start(), block_start(0, TEXT_BLOCK), text_delta(0, "Hi.")]
    _, stream = await answer(streaming(*events, message_delta(None, 3), MESSAGE_STOP))

    assert stream.stop_reason is None
    assert stream.usage == Usage(42, 3)


async def test_an_unreadable_error_event_mid_stream_is_a_rejected_request() -> None:
    chunks = sse([message_start(), block_start(0, TEXT_BLOCK), text_delta(0, "Part")])
    body = Body([chunks + f"event: error\ndata: {UPSTREAM_TEXT}\n\n".encode()])

    with pytest.raises(LLMError) as raised:
        await answer(FakeAPI([ok(body)]))

    assert_fixed(raised.value, LLMRequestError)


def test_any_other_sdk_failure_is_a_generic_model_error() -> None:
    error = domain_error(anthropic.AnthropicError(UPSTREAM_TEXT))

    assert type(error) is LLMError
    assert str(error) == LLMError.message

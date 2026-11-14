"""The answer harness's HTTP client and per-item runner, against an httpx MockTransport
that serves the backend's own SSE frames and error bodies (`frame`, `error_body`).

Time is injected: a fake clock for latencies and a recording sleep for Retry-After, so
retries are checked without waiting.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

import httpx
import pytest

from app.api.messages import KEEPALIVE, frame
from app.core.errors import error_body
from app.domain.answer_dataset import EvalItem
from app.domain.chat import RefusalReason
from app.domain.evaluation import Split
from app.evaluation.answers.client import (
    MAX_RETRIES,
    MAX_RETRY_AFTER_SECONDS,
    Answered,
    ChatClient,
    Created,
    Failed,
    HarnessCode,
    Timeouts,
    run_item,
)

CONVERSATION = "c-1"
CLIENT_ID = "0b6f9a52-4a3e-4e56-9f0c-2d1f3c4b5a69"
MESSAGES = f"/v1/conversations/{CONVERSATION}/messages"


def done(**fields: object) -> dict[str, object]:
    data: dict[str, object] = {
        "conversation_id": CONVERSATION,
        "user_message_id": "u-1",
        "message_id": "m-1",
        "outcome": "completed",
        "refusal": False,
        "refusal_reason": None,
        "content": "Clear the tray [S1].",
        "citations": ["S1"],
        "flags": {
            "late_refusal_marker": False,
            "uncited": False,
            "invalid_citations": 0,
            "rewrite_fallback": None,
            "truncated": False,
        },
    }
    return {**data, **fields}


REFUSED = done(
    outcome="refused",
    refusal=True,
    refusal_reason="below_threshold",
    content="I cannot answer that from the manuals.",
    citations=[],
)


def answer_stream(final: dict[str, object] | None = None, tokens: int = 2) -> bytes:
    body = b"".join(frame("token", {"text": f"t{n} "}) for n in range(tokens))
    return body + KEEPALIVE + frame("sources", {"sources": []}) + frame("done", final or done())


def sse(body: bytes) -> httpx.Response:
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


def http_error(status: int, code: str, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status, json=error_body(code, "Fixed message.", "req-1"), headers=headers or {}
    )


Reply = httpx.Response | Exception


@dataclass
class Server:
    """Scripted replies per path, in order; every request is recorded."""

    replies: dict[str, list[Reply]] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        queue = self.replies[request.url.path]
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def bodies(self, path: str) -> list[object]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]


@dataclass
class Clock:
    """Each read advances by `step` seconds, so latencies are known exactly."""

    now: float = 0.0
    step: float = 0.010

    def __call__(self) -> float:
        self.now += self.step
        return self.now


@dataclass
class Sleeps:
    calls: list[float] = field(default_factory=list)

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@dataclass
class Rig:
    server: Server
    sleeps: Sleeps
    chat: ChatClient


@pytest.fixture
async def rig_factory() -> AsyncIterator[Callable[..., Rig]]:
    clients: list[httpx.AsyncClient] = []

    def make(timeouts: Timeouts | None = None, **replies: list[Reply]) -> Rig:
        server = Server(
            replies={"/" + path.replace("__", "/"): queue for path, queue in replies.items()}
        )
        http = httpx.AsyncClient(
            transport=httpx.MockTransport(server.handler), base_url="http://backend:8000"
        )
        clients.append(http)
        sleeps = Sleeps()
        chat = ChatClient(http, timeouts=timeouts or Timeouts(), clock=Clock(), sleep=sleeps)
        return Rig(server, sleeps, chat)

    yield make
    for client in clients:
        await client.aclose()


def turn_rig(make: Callable[..., Rig], *replies: Reply, timeouts: Timeouts | None = None) -> Rig:
    rig = make(timeouts=timeouts)
    rig.server.replies[MESSAGES] = list(replies)
    return rig


async def send(rig: Rig, content: str = "How do I clear a jam?") -> Answered | Failed:
    return await rig.chat.send_turn(CONVERSATION, CLIENT_ID, content)


# ------------------------------------------------------------------ one turn


async def test_an_answered_turn_returns_the_done_event_and_its_latencies(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = turn_rig(rig_factory, sse(answer_stream()))

    result = await send(rig)

    assert isinstance(result, Answered)
    assert result.done.message_id == "m-1"
    assert result.done.citations == ("S1",)
    assert result.retries == 0
    # Clock reads: start, first token, done -> 10 ms to the first token, 20 ms in all.
    assert result.ttft_ms == pytest.approx(10.0)
    assert result.total_ms == pytest.approx(20.0)


async def test_the_turn_is_posted_with_the_client_id_and_the_content(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = turn_rig(rig_factory, sse(answer_stream()))

    await send(rig, "Where is the tray?")

    (request,) = rig.server.requests
    assert request.method == "POST"
    assert request.headers["X-Client-ID"] == CLIENT_ID
    assert json.loads(request.content) == {"content": "Where is the tray?"}


async def test_a_refusal_without_tokens_has_no_time_to_first_token(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = turn_rig(rig_factory, sse(answer_stream(REFUSED, tokens=0)))

    result = await send(rig)

    assert isinstance(result, Answered)
    assert result.done.refusal_reason is RefusalReason.BELOW_THRESHOLD
    assert result.ttft_ms is None
    assert result.total_ms == pytest.approx(10.0)


@pytest.mark.parametrize(("status", "code"), [(429, "rate_limited"), (503, "llm_unavailable")])
async def test_a_retryable_status_is_retried_after_the_retry_after_seconds(
    rig_factory: Callable[..., Rig], status: int, code: str
) -> None:
    rig = turn_rig(
        rig_factory,
        http_error(status, code, {"Retry-After": "3"}),
        sse(answer_stream()),
    )

    result = await send(rig)

    assert isinstance(result, Answered)
    assert result.retries == 1
    assert rig.sleeps.calls == [3.0]
    assert len(rig.server.requests) == 2


async def test_latencies_are_measured_on_the_attempt_that_answered(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = turn_rig(rig_factory, http_error(503, "llm_unavailable"), sse(answer_stream()))

    result = await send(rig)

    assert isinstance(result, Answered)
    assert result.ttft_ms == pytest.approx(10.0)
    assert result.total_ms == pytest.approx(20.0)


async def test_retries_stop_after_the_maximum_and_return_the_last_error(
    rig_factory: Callable[..., Rig],
) -> None:
    replies = [http_error(429, "rate_limited", {"Retry-After": "1"})] * (MAX_RETRIES + 1)
    rig = turn_rig(rig_factory, *replies)

    result = await send(rig)

    assert result == Failed(code="rate_limited", http_status=429, retries=MAX_RETRIES)
    assert len(rig.server.requests) == MAX_RETRIES + 1
    assert len(rig.sleeps.calls) == MAX_RETRIES


@pytest.mark.parametrize(
    ("header", "seconds"),
    [
        (None, 1.0),
        ("soon", 1.0),
        ("-4", 1.0),
        ("0", 0.0),
        ("2.5", 2.5),
        ("100000", MAX_RETRY_AFTER_SECONDS),
    ],
)
async def test_retry_after_is_parsed_defensively(
    rig_factory: Callable[..., Rig], header: str | None, seconds: float
) -> None:
    headers = {} if header is None else {"Retry-After": header}
    rig = turn_rig(rig_factory, http_error(429, "rate_limited", headers), sse(answer_stream()))

    await send(rig)

    assert rig.sleeps.calls == [seconds]


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (400, "client_id_required"),
        (404, "conversation_not_found"),
        (409, "conversation_busy"),
        (422, "validation_error"),
        (500, "internal_error"),
        (502, "llm_error"),
    ],
)
async def test_other_error_statuses_fail_at_once_with_the_server_code(
    rig_factory: Callable[..., Rig], status: int, code: str
) -> None:
    rig = turn_rig(rig_factory, http_error(status, code))

    result = await send(rig)

    assert result == Failed(code=code, http_status=status, retries=0)
    assert rig.sleeps.calls == []


@pytest.mark.parametrize(
    "content",
    [b"<html>bad gateway</html>", b'{"error": "flat"}', b'{"error": {"code": "Not A Code"}}'],
)
async def test_an_error_body_without_a_usable_code_is_named_by_its_status(
    rig_factory: Callable[..., Rig], content: bytes
) -> None:
    rig = turn_rig(rig_factory, httpx.Response(502, content=content))

    assert await send(rig) == Failed(code="http_502", http_status=502, retries=0)


async def test_an_error_event_after_a_token_fails_without_a_retry(
    rig_factory: Callable[..., Rig],
) -> None:
    body = frame("token", {"text": "Clear "}) + frame(
        "error",
        {"code": "internal_error", "message": "x", "request_id": "r", "message_id": "m-1"},
    )
    rig = turn_rig(rig_factory, sse(body), sse(answer_stream()))

    result = await send(rig)

    assert result == Failed(code="internal_error", http_status=200, retries=0)
    assert len(rig.server.requests) == 1


async def test_a_malformed_done_event_is_a_failure_not_a_guess(
    rig_factory: Callable[..., Rig],
) -> None:
    broken = done()
    del broken["citations"]
    rig = turn_rig(rig_factory, sse(answer_stream(broken)))

    assert await send(rig) == Failed(HarnessCode.MALFORMED_EVENT, 200, 0)


async def test_a_done_event_for_another_conversation_is_malformed(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = turn_rig(rig_factory, sse(answer_stream(done(conversation_id="c-other"))))

    assert await send(rig) == Failed(HarnessCode.MALFORMED_EVENT, 200, 0)


async def test_a_malformed_error_event_is_malformed(rig_factory: Callable[..., Rig]) -> None:
    rig = turn_rig(rig_factory, sse(frame("token", {"text": "a"}) + frame("error", {"x": 1})))

    assert await send(rig) == Failed(HarnessCode.MALFORMED_EVENT, 200, 0)


async def test_a_stream_that_ends_without_done_is_incomplete(
    rig_factory: Callable[..., Rig],
) -> None:
    body = frame("token", {"text": "a"}) + frame("unknown", {"ignored": True})
    rig = turn_rig(rig_factory, sse(body))

    assert await send(rig) == Failed(HarnessCode.INCOMPLETE_STREAM, 200, 0)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (httpx.ReadTimeout("slow"), HarnessCode.READ_TIMEOUT),
        (httpx.ConnectError("refused"), HarnessCode.TRANSPORT_ERROR),
    ],
)
async def test_transport_failures_are_not_retried(
    rig_factory: Callable[..., Rig], error: Exception, code: HarnessCode
) -> None:
    rig = turn_rig(rig_factory, error, sse(answer_stream()))

    assert await send(rig) == Failed(code, None, 0)
    assert len(rig.server.requests) == 1


async def test_a_turn_over_the_total_deadline_fails(rig_factory: Callable[..., Rig]) -> None:
    async def slow() -> AsyncIterator[bytes]:
        yield frame("token", {"text": "a"})
        await asyncio.sleep(5)
        yield frame("done", done())

    response = httpx.Response(200, content=slow(), headers={"content-type": "text/event-stream"})
    rig = turn_rig(rig_factory, response, timeouts=Timeouts(read=1.0, total=0.05))

    assert await send(rig) == Failed(HarnessCode.TOTAL_TIMEOUT, 200, 0)


def test_the_default_deadlines_match_the_backend_limits() -> None:
    # 185 s is the backend's own longest answer (rewrite + total + one minute, ADR 0010).
    assert Timeouts() == Timeouts(read=60.0, total=185.0)


# ------------------------------------------------------------------ conversations


async def test_a_conversation_is_created_for_the_client(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory(v1__conversations=[httpx.Response(201, json={"id": CONVERSATION})])

    created = await rig.chat.create_conversation(CLIENT_ID)

    assert created == Created(CONVERSATION, retries=0)
    (request,) = rig.server.requests
    assert request.method == "POST"
    assert request.headers["X-Client-ID"] == CLIENT_ID


async def test_creation_is_retried_on_a_retryable_status(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory(
        v1__conversations=[
            http_error(429, "rate_limited", {"Retry-After": "2"}),
            httpx.Response(201, json={"id": CONVERSATION}),
        ]
    )

    assert await rig.chat.create_conversation(CLIENT_ID) == Created(CONVERSATION, retries=1)
    assert rig.sleeps.calls == [2.0]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (http_error(500, "internal_error"), Failed("internal_error", 500, 0)),
        (httpx.Response(201, json={"title": None}), Failed(HarnessCode.MALFORMED_EVENT, 201, 0)),
        (httpx.Response(201, content=b"[]"), Failed(HarnessCode.MALFORMED_EVENT, 201, 0)),
        (httpx.Response(201, content=b"not json"), Failed(HarnessCode.MALFORMED_EVENT, 201, 0)),
        (httpx.ConnectError("refused"), Failed(HarnessCode.TRANSPORT_ERROR, None, 0)),
    ],
)
async def test_a_failed_creation_says_why(
    rig_factory: Callable[..., Rig], reply: Reply, expected: Failed
) -> None:
    rig = rig_factory(v1__conversations=[reply])

    assert await rig.chat.create_conversation(CLIENT_ID) == expected


async def test_a_conversation_is_deleted_for_the_client(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()
    rig.server.replies[f"/v1/conversations/{CONVERSATION}"] = [httpx.Response(204)]

    assert await rig.chat.delete_conversation(CONVERSATION, CLIENT_ID) is None
    (request,) = rig.server.requests
    assert request.method == "DELETE"
    assert request.headers["X-Client-ID"] == CLIENT_ID


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (http_error(404, "conversation_not_found"), Failed("conversation_not_found", 404, 0)),
        (httpx.ConnectError("refused"), Failed(HarnessCode.TRANSPORT_ERROR, None, 0)),
    ],
)
async def test_a_failed_deletion_says_why(
    rig_factory: Callable[..., Rig], reply: Reply, expected: Failed
) -> None:
    rig = rig_factory()
    rig.server.replies[f"/v1/conversations/{CONVERSATION}"] = [reply]

    assert await rig.chat.delete_conversation(CONVERSATION, CLIENT_ID) == expected


# ------------------------------------------------------------------ one item


def item(*turns: str) -> EvalItem:
    return EvalItem(id="mt-1", split=Split.DEV, turns=turns, in_scope=True)


def item_rig(make: Callable[..., Rig], *replies: Reply) -> Rig:
    rig = make(v1__conversations=[httpx.Response(201, json={"id": CONVERSATION})])
    rig.server.replies[MESSAGES] = list(replies)
    return rig


async def test_a_single_turn_item_creates_a_conversation_and_asks_once(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = item_rig(rig_factory, sse(answer_stream()))

    result = await run_item(rig.chat, item("Where is the tray?"), CLIENT_ID)

    assert result.item_id == "mt-1"
    assert result.client_id == CLIENT_ID
    assert result.conversation_id == CONVERSATION
    assert isinstance(result.answer, Answered)
    assert result.answer.done.message_id == "m-1"
    assert result.error is None
    assert result.ok
    assert rig.server.bodies(MESSAGES) == [{"content": "Where is the tray?"}]


async def test_earlier_turns_are_replayed_in_order_and_the_last_is_scored(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = item_rig(
        rig_factory,
        sse(answer_stream(done(message_id="m-a"))),
        sse(answer_stream(done(message_id="m-b"))),
        sse(answer_stream(done(message_id="m-c"))),
    )

    result = await run_item(rig.chat, item("first", "second", "third"), CLIENT_ID)

    assert rig.server.bodies(MESSAGES) == [
        {"content": "first"},
        {"content": "second"},
        {"content": "third"},
    ]
    assert result.answer is not None
    assert result.answer.done.message_id == "m-c"
    assert result.setup_refused == 0


async def test_a_refused_earlier_turn_is_kept_and_counted(rig_factory: Callable[..., Rig]) -> None:
    rig = item_rig(rig_factory, sse(answer_stream(REFUSED, tokens=0)), sse(answer_stream()))

    result = await run_item(rig.chat, item("first", "second"), CLIENT_ID)

    assert result.ok
    assert result.setup_refused == 1


async def test_a_failed_earlier_turn_stops_the_item_as_a_setup_failure(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = item_rig(rig_factory, http_error(502, "llm_error"), sse(answer_stream()))

    result = await run_item(rig.chat, item("first", "second"), CLIENT_ID)

    assert result.error == HarnessCode.SETUP_TURN_FAILED
    assert result.cause == "llm_error"
    assert result.http_status == 502
    assert result.answer is None
    assert result.conversation_id == CONVERSATION
    assert not result.ok
    assert len(rig.server.bodies(MESSAGES)) == 1


async def test_a_failed_creation_sends_no_turn(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory(v1__conversations=[http_error(500, "internal_error")])

    result = await run_item(rig.chat, item("only"), CLIENT_ID)

    assert result.error == "internal_error"
    assert result.cause is None
    assert result.conversation_id is None
    assert rig.server.bodies(MESSAGES) == []


async def test_a_failed_last_turn_is_an_item_error(rig_factory: Callable[..., Rig]) -> None:
    rig = item_rig(rig_factory, sse(answer_stream()), http_error(409, "conversation_busy"))

    result = await run_item(rig.chat, item("first", "second"), CLIENT_ID)

    assert result.error == "conversation_busy"
    assert result.cause is None
    assert result.http_status == 409
    assert result.answer is None


async def test_retries_are_summed_over_creation_and_every_turn(
    rig_factory: Callable[..., Rig],
) -> None:
    rig = rig_factory(
        v1__conversations=[
            http_error(429, "rate_limited"),
            httpx.Response(201, json={"id": CONVERSATION}),
        ]
    )
    rig.server.replies[MESSAGES] = [
        http_error(503, "llm_unavailable"),
        sse(answer_stream()),
        http_error(429, "rate_limited"),
        http_error(429, "rate_limited"),
        sse(answer_stream()),
    ]

    result = await run_item(rig.chat, item("first", "second"), CLIENT_ID)

    assert result.ok
    assert result.retries == 4


async def test_retries_are_counted_on_a_failed_item_too(rig_factory: Callable[..., Rig]) -> None:
    rig = item_rig(rig_factory, *[http_error(503, "llm_unavailable")] * (MAX_RETRIES + 1))

    result = await run_item(rig.chat, item("only"), CLIENT_ID)

    assert result.error == "llm_unavailable"
    assert result.retries == MAX_RETRIES

"""Behavior of the pure-ASGI request ID and access log middleware."""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from starlette.types import Message, Receive, Scope, Send

from app.core.middleware import REQUEST_ID_HEADER, RequestIdMiddleware
from app.core.request_context import get_request_id
from tests.unit.conftest import LogReader

VALID_ID = "abcDEF12-3456-7890"


def build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    barrier = asyncio.Barrier(2)

    @app.get("/echo")
    async def echo() -> dict[str, str | None]:
        return {"request_id": get_request_id()}

    @app.get("/together")
    async def together() -> dict[str, str | None]:
        async with asyncio.timeout(2):
            await barrier.wait()
        return {"request_id": get_request_id()}

    @app.get("/teapot")
    async def teapot() -> None:
        raise HTTPException(status_code=418, detail="short and stout")

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("kaboom")

    @app.post("/submit")
    async def submit(payload: dict[str, str]) -> dict[str, int]:
        return {"fields": len(payload)}

    return app


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=build_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


def is_generated_uuid4(value: str) -> bool:
    return uuid.UUID(value).version == 4 and str(uuid.UUID(value)) == value


# ------------------------------------------------------------------ raw ASGI helpers


def http_scope(headers: list[tuple[bytes, bytes]]) -> Scope:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/raw",
        "raw_path": b"/raw",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("testclient", 1),
        "server": ("testserver", 80),
    }


async def empty_receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


async def echo_id_app(scope: Scope, receive: Receive, send: Send) -> None:  # noqa: ARG001
    body = json.dumps({"request_id": get_request_id()}).encode()
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": body})


async def call_raw(headers: list[tuple[bytes, bytes]]) -> list[Message]:
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    await RequestIdMiddleware(echo_id_app)(http_scope(headers), empty_receive, send)
    return sent


def response_request_id(messages: list[Message]) -> str:
    start = messages[0]
    headers: list[tuple[bytes, bytes]] = start["headers"]
    values = [v.decode() for k, v in headers if k == REQUEST_ID_HEADER.encode()]
    assert len(values) == 1
    return values[0]


# ------------------------------------------------------------------ X-Request-ID handling


@pytest.mark.parametrize("incoming", [VALID_ID, "a" * 8, "Z" * 64, "-" * 8])
async def test_valid_incoming_request_id_is_echoed(
    client: httpx.AsyncClient, incoming: str
) -> None:
    response = await client.get("/echo", headers={REQUEST_ID_HEADER: incoming})

    assert response.headers[REQUEST_ID_HEADER] == incoming
    assert response.json() == {"request_id": incoming}


async def test_missing_request_id_is_generated_as_uuid4(client: httpx.AsyncClient) -> None:
    response = await client.get("/echo")

    generated = response.headers[REQUEST_ID_HEADER]
    assert is_generated_uuid4(generated)
    assert response.json() == {"request_id": generated}


@pytest.mark.parametrize(
    "incoming",
    [
        pytest.param("a" * 7, id="too-short"),
        pytest.param("a" * 65, id="too-long"),
        pytest.param("abcd efgh", id="space"),
        pytest.param("abcd_efgh", id="underscore"),
        pytest.param("", id="empty"),
    ],
)
async def test_invalid_incoming_request_id_is_replaced(
    client: httpx.AsyncClient, incoming: str
) -> None:
    response = await client.get("/echo", headers={REQUEST_ID_HEADER: incoming})

    generated = response.headers[REQUEST_ID_HEADER]
    assert generated != incoming
    assert is_generated_uuid4(generated)


@pytest.mark.parametrize(
    "raw_value",
    [
        pytest.param(b"abcdefgh\r\nX-Injected: 1", id="crlf"),
        pytest.param(b"abcdefgh\n", id="trailing-lf"),
        pytest.param(b"abcdefgh\r", id="trailing-cr"),
        pytest.param("abcdéfgh".encode(), id="non-ascii-utf8"),
        pytest.param("abcdéfgh".encode("latin-1"), id="non-ascii-latin1"),
    ],
)
async def test_control_and_non_ascii_request_ids_are_replaced(raw_value: bytes) -> None:
    messages = await call_raw([(b"x-request-id", raw_value)])

    generated = response_request_id(messages)
    assert is_generated_uuid4(generated)
    assert json.loads(messages[1]["body"]) == {"request_id": generated}


async def test_existing_request_id_header_from_app_is_replaced() -> None:
    async def app_setting_header(scope: Scope, receive: Receive, send: Send) -> None:  # noqa: ARG001
        headers = [(b"x-request-id", b"from-the-app-itself")]
        await send({"type": "http.response.start", "status": 204, "headers": headers})
        await send({"type": "http.response.body", "body": b""})

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    scope = http_scope([(b"x-request-id", VALID_ID.encode())])
    await RequestIdMiddleware(app_setting_header)(scope, empty_receive, send)

    assert response_request_id(sent) == VALID_ID


async def test_request_id_header_present_on_http_error_response(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/teapot", headers={REQUEST_ID_HEADER: VALID_ID})

    assert response.status_code == 418
    assert response.headers[REQUEST_ID_HEADER] == VALID_ID


async def test_request_id_header_present_on_not_found(client: httpx.AsyncClient) -> None:
    response = await client.get("/nowhere", headers={REQUEST_ID_HEADER: VALID_ID})

    assert response.status_code == 404
    assert response.headers[REQUEST_ID_HEADER] == VALID_ID


async def test_unhandled_exception_returns_safe_500_with_request_id(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/boom", headers={REQUEST_ID_HEADER: VALID_ID})

    assert response.status_code == 500
    assert response.headers[REQUEST_ID_HEADER] == VALID_ID
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "error": {
            "code": "internal_error",
            "message": "Internal server error.",
            "request_id": VALID_ID,
            "details": [],
        }
    }
    assert "kaboom" not in response.text
    assert "Traceback" not in response.text


async def test_unhandled_exception_is_logged_once_with_request_id(
    client: httpx.AsyncClient, json_logs: LogReader
) -> None:
    await client.get("/boom", headers={REQUEST_ID_HEADER: VALID_ID})

    errors = [line for line in json_logs() if line["level"] == "ERROR"]
    assert len(errors) == 1
    assert errors[0]["request_id"] == VALID_ID
    assert errors[0]["exception_type"] == "RuntimeError"
    assert "kaboom" in errors[0]["exception"]


async def test_exception_after_response_started_is_reraised() -> None:
    async def breaks_mid_stream(scope: Scope, receive: Receive, send: Send) -> None:  # noqa: ARG001
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("stream broke")

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = RequestIdMiddleware(breaks_mid_stream)
    with pytest.raises(RuntimeError, match="stream broke"):
        await middleware(http_scope([]), empty_receive, send)

    assert len(sent) == 1
    assert is_generated_uuid4(response_request_id(sent))


# ------------------------------------------------------------------ request context


async def test_request_id_context_is_reset_after_request(client: httpx.AsyncClient) -> None:
    await client.get("/echo", headers={REQUEST_ID_HEADER: VALID_ID})
    assert get_request_id() is None

    await client.get("/boom", headers={REQUEST_ID_HEADER: VALID_ID})
    assert get_request_id() is None


async def test_concurrent_requests_keep_distinct_request_ids(client: httpx.AsyncClient) -> None:
    first, second = "first-request-aaaa", "second-request-bbbb"

    responses = await asyncio.gather(
        client.get("/together", headers={REQUEST_ID_HEADER: first}),
        client.get("/together", headers={REQUEST_ID_HEADER: second}),
    )

    assert [r.json()["request_id"] for r in responses] == [first, second]
    assert [r.headers[REQUEST_ID_HEADER] for r in responses] == [first, second]


async def test_non_http_scopes_pass_through_untouched() -> None:
    seen: list[Scope] = []

    async def lifespan_app(scope: Scope, receive: Receive, send: Send) -> None:  # noqa: ARG001
        seen.append(scope)

    async def never_send(message: Message) -> None:
        raise AssertionError(message)

    scope: Scope = {"type": "lifespan", "asgi": {"version": "3.0"}}
    await RequestIdMiddleware(lifespan_app)(scope, empty_receive, never_send)

    assert seen == [scope]
    assert get_request_id() is None


# ------------------------------------------------------------------ access log


async def test_access_log_records_method_path_status_and_duration(
    client: httpx.AsyncClient, json_logs: LogReader
) -> None:
    await client.get("/teapot?token=abc123", headers={REQUEST_ID_HEADER: VALID_ID})

    access = [line for line in json_logs() if line["logger"] == "app.access"]
    assert len(access) == 1
    entry = access[0]
    assert entry["method"] == "GET"
    assert entry["path"] == "/teapot"
    assert entry["status"] == 418
    assert isinstance(entry["duration_ms"], float)
    assert entry["duration_ms"] >= 0
    assert entry["request_id"] == VALID_ID


async def test_access_log_records_500_for_unhandled_exception(
    client: httpx.AsyncClient, json_logs: LogReader
) -> None:
    await client.get("/boom")

    access = [line for line in json_logs() if line["logger"] == "app.access"]
    assert [entry["status"] for entry in access] == [500]


async def test_access_log_never_contains_query_string_or_bodies(
    client: httpx.AsyncClient, json_logs: LogReader
) -> None:
    await client.post("/submit?token=query-secret", json={"question": "body-secret"})

    app_logs = [line for line in json_logs() if line["logger"].startswith("app")]
    assert app_logs
    raw = json.dumps(app_logs)
    assert "query-secret" not in raw
    assert "body-secret" not in raw
    assert '"fields"' not in raw


# ------------------------------------------------------------------ streaming responses

CHUNK_GAP_SECONDS = 0.05
POST_RESPONSE_WORK_SECONDS = 0.3


class EventRecorder(logging.Handler):
    """Records access log emissions into the same timeline as sent ASGI messages."""

    def __init__(self, timeline: list[str]) -> None:
        super().__init__()
        self.timeline = timeline
        self.durations: list[float] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.timeline.append("access-log")
        self.durations.append(record.__dict__["duration_ms"])


async def run_streaming(post_response_work: float) -> tuple[list[str], list[float]]:
    timeline: list[str] = []

    async def streaming_app(scope: Scope, receive: Receive, send: Send) -> None:  # noqa: ARG001
        await send({"type": "http.response.start", "status": 200, "headers": []})
        for chunk in (b"one", b"two"):
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await asyncio.sleep(CHUNK_GAP_SECONDS)
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        await asyncio.sleep(post_response_work)  # like Starlette BackgroundTasks

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start":
            timeline.append("start")
        else:
            timeline.append("final" if not message.get("more_body") else "chunk")

    recorder = EventRecorder(timeline)
    access_logger = logging.getLogger("app.access")
    previous_level = access_logger.level
    access_logger.addHandler(recorder)
    access_logger.setLevel(logging.INFO)
    try:
        await RequestIdMiddleware(streaming_app)(http_scope([]), empty_receive, send)
    finally:
        access_logger.removeHandler(recorder)
        access_logger.setLevel(previous_level)
    return timeline, recorder.durations


async def test_streaming_access_log_is_written_after_final_chunk() -> None:
    timeline, _ = await run_streaming(post_response_work=0)

    assert timeline == ["start", "chunk", "chunk", "final", "access-log"]


async def test_streaming_duration_covers_every_chunk() -> None:
    _, durations = await run_streaming(post_response_work=0)

    assert durations[0] >= 2 * CHUNK_GAP_SECONDS * 1000


async def test_streaming_duration_stops_at_final_chunk_not_post_response_work() -> None:
    _, durations = await run_streaming(post_response_work=POST_RESPONSE_WORK_SECONDS)

    assert durations[0] < POST_RESPONSE_WORK_SECONDS * 1000

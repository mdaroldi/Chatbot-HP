"""HttpChatApi against httpx.MockTransport: the backend's shapes in, codes out.

Every failure leaves as an ApiError with a code and no response text; the answer
stream yields typed events, stops at the first terminal one and always closes.
"""

import json
import logging
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime

import httpx
import pytest

from hpdocs_ui.api import (
    Conversation,
    ConversationDetail,
    ConversationPage,
    Done,
    Flags,
    Message,
    Pages,
    Source,
    SourcesEvent,
    StreamEvent,
    Token,
)
from hpdocs_ui.errors import BACKEND_UNREACHABLE, BAD_RESPONSE, STREAM_TIMEOUT, ApiError
from hpdocs_ui.http_api import HttpChatApi
from hpdocs_ui.settings import FrontendSettings
from hpdocs_ui.sse import MAX_PENDING_FRAME_BYTES

Handler = Callable[[httpx.Request], httpx.Response]
BASE = "http://backend:8000"
CID = "3f1c2a9e-5b7d-4e8f-9a0b-1c2d3e4f5a6b"
CONV = "8a1e5c3d-2b4f-4a6e-8c0d-9e1f2a3b4c5d"
RID = "req-12345678"
CANARY = "CANARY-7f3a"
STAMP = "2026-11-14T08:00:00Z"
WHEN = datetime(2026, 11, 14, 8, 0, tzinfo=UTC)
SETTINGS = FrontendSettings(
    backend_url=BASE + "/",
    ui_stream_read_timeout_seconds=60.0,
    ui_stream_total_timeout_seconds=185.0,
)

CONVERSATION_JSON = {"id": CONV, "title": None, "created_at": STAMP, "updated_at": STAMP}
CONVERSATION = Conversation(id=CONV, title=None, created_at=WHEN, updated_at=WHEN)
SOURCE_JSON = {
    "tag": "S1",
    "document_id": "envy-6000",
    "document_title": "HP ENVY 6000 All-in-One series User Guide",
    "section_path": ["Paper", "Load paper"],
    "pages": {"pdf_start": 7, "pdf_end": 8, "printed_start": 6, "printed_end": None},
    "has_warning": True,
    "snippet": "Pull out the input tray.",
}
SOURCE = Source(
    tag="S1",
    document_id="envy-6000",
    document_title="HP ENVY 6000 All-in-One series User Guide",
    section_path=("Paper", "Load paper"),
    pages=Pages(pdf_start=7, pdf_end=8, printed_start=6, printed_end=None),
    has_warning=True,
    snippet="Pull out the input tray.",
)
FLAGS_JSON = {
    "late_refusal_marker": False,
    "uncited": False,
    "invalid_citations": 0,
    "rewrite_fallback": None,
    "truncated": False,
}
DONE_JSON = {
    "conversation_id": CONV,
    "user_message_id": "11111111-1111-4111-8111-111111111111",
    "message_id": "22222222-2222-4222-8222-222222222222",
    "outcome": "completed",
    "refusal": False,
    "refusal_reason": None,
    "content": "Pull out the input tray [S1].",
    "citations": ["S1"],
    "flags": FLAGS_JSON,
}
DONE = Done(
    conversation_id=CONV,
    user_message_id="11111111-1111-4111-8111-111111111111",
    message_id="22222222-2222-4222-8222-222222222222",
    outcome="completed",
    refusal=False,
    refusal_reason=None,
    content="Pull out the input tray [S1].",
    citations=("S1",),
    flags=Flags(
        late_refusal_marker=False,
        uncited=False,
        invalid_citations=0,
        rewrite_fallback=None,
        truncated=False,
    ),
)


def frame(event: str, data: object) -> bytes:
    """As the backend writes it (backend/app/api/messages.py)."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def token(text: str) -> bytes:
    return frame("token", {"text": text})


SOURCES = frame("sources", {"sources": [SOURCE_JSON]})
DONE_FRAME = frame("done", DONE_JSON)


class RecordingStream(httpx.SyncByteStream):
    """A response body that records how far it was read and whether it was closed."""

    def __init__(self, chunks: Iterable[bytes | Exception]) -> None:
        self.chunks = chunks
        self.pulled = 0
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self.chunks:
            self.pulled += 1
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self) -> None:
        self.closed = True


def sse(body: RecordingStream, status: int = 200) -> Handler:
    def handler(_: httpx.Request) -> httpx.Response:
        headers = {"content-type": "text/event-stream", "x-request-id": RID}
        return httpx.Response(status, headers=headers, stream=body)

    return handler


def make_api(handler: Handler, clock: Callable[[], float] = time.monotonic) -> HttpChatApi:
    return HttpChatApi(SETTINGS, CID, transport=httpx.MockTransport(handler), clock=clock)


def collect(api: HttpChatApi, events: list[StreamEvent], content: str = "How?") -> None:
    with api.stream_answer(CONV, content) as stream:
        events.extend(stream)


def error_response(status: int, code: str, **headers: str) -> httpx.Response:
    body = {"error": {"code": code, "message": CANARY, "request_id": RID, "details": None}}
    return httpx.Response(status, json=body, headers={"x-request-id": RID, **headers})


def stream_failure(api: HttpChatApi) -> tuple[ApiError, list[StreamEvent]]:
    events: list[StreamEvent] = []
    with pytest.raises(ApiError) as raised:
        collect(api, events)
    return raised.value, events


def assert_unchained(error: BaseException) -> None:
    """A traceback of the error shows no other exception, whose text could be logged."""
    assert error.__cause__ is None
    assert error.__context__ is None or error.__suppress_context__


# ------------------------------------------------------------------ construction


@pytest.mark.parametrize("client_id", ["short", f"{CANARY}/../x", "", "a" * 65])
def test_a_client_id_outside_the_allowlist_is_refused_without_echoing_it(
    client_id: str,
) -> None:
    with pytest.raises(ValueError, match="client ID") as raised:
        HttpChatApi(SETTINGS, client_id)

    assert client_id not in str(raised.value) or client_id == ""


# ------------------------------------------------------------------ plain calls


def test_create_conversation_posts_and_parses_the_201() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert (request.method, str(request.url)) == ("POST", f"{BASE}/v1/conversations")
        return httpx.Response(201, json=CONVERSATION_JSON)

    assert make_api(handler).create_conversation() == CONVERSATION


def test_list_conversations_parses_the_page_and_sends_no_cursor_by_default() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert (request.method, str(request.url)) == ("GET", f"{BASE}/v1/conversations")
        titled = {**CONVERSATION_JSON, "title": "Load paper"}
        return httpx.Response(200, json={"items": [titled], "next_cursor": "abc"})

    page = make_api(handler).list_conversations()

    assert page == ConversationPage(
        items=(CONVERSATION.model_copy(update={"title": "Load paper"}),), next_cursor="abc"
    )


def test_list_conversations_sends_the_cursor_encoded_as_a_query_parameter() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["cursor"] == "a+b/c="
        assert request.url.query == b"cursor=a%2Bb%2Fc%3D"
        return httpx.Response(200, json={"items": [], "next_cursor": None})

    assert make_api(handler).list_conversations("a+b/c=") == ConversationPage(
        items=(), next_cursor=None
    )


def test_get_conversation_parses_messages_and_their_sources() -> None:
    user: dict[str, object] = {
        "id": "u1",
        "role": "user",
        "content": "How do I load paper?",
        "created_at": STAMP,
        "outcome": None,
        "refusal": False,
        "refusal_reason": None,
        "citations": [],
        "sources": [],
    }
    assistant = {
        **user,
        "id": "a1",
        "role": "assistant",
        "content": "Pull out the input tray [S1].",
        "outcome": "completed",
        "citations": ["S1"],
        "sources": [SOURCE_JSON],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert (request.method, str(request.url)) == ("GET", f"{BASE}/v1/conversations/{CONV}")
        return httpx.Response(200, json={**CONVERSATION_JSON, "messages": [user, assistant]})

    detail = make_api(handler).get_conversation(CONV)

    assert detail == ConversationDetail(
        **CONVERSATION.model_dump(),
        messages=(
            Message(
                id="u1",
                role="user",
                content="How do I load paper?",
                created_at=WHEN,
                outcome=None,
                refusal=False,
                refusal_reason=None,
                citations=(),
                sources=(),
            ),
            Message(
                id="a1",
                role="assistant",
                content="Pull out the input tray [S1].",
                created_at=WHEN,
                outcome="completed",
                refusal=False,
                refusal_reason=None,
                citations=("S1",),
                sources=(SOURCE,),
            ),
        ),
    )


def test_delete_conversation_accepts_the_204() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        return httpx.Response(204)

    make_api(handler).delete_conversation(CONV)

    assert calls == [("DELETE", f"{BASE}/v1/conversations/{CONV}")]


def test_fields_the_frontend_does_not_know_are_ignored() -> None:
    body = {**CONVERSATION_JSON, "owner": "someone", "extra": {"nested": 1}}

    api = make_api(lambda _: httpx.Response(201, json=body))

    assert api.create_conversation() == CONVERSATION


Call = Callable[[HttpChatApi], object]
CALLS: dict[str, Call] = {
    "create": lambda api: api.create_conversation(),
    "list": lambda api: api.list_conversations(),
    "get": lambda api: api.get_conversation(CONV),
    "delete": lambda api: api.delete_conversation(CONV),
}
STREAM_CALL: Call = lambda api: collect(api, [])  # noqa: E731
ALL_CALLS = {**CALLS, "stream": STREAM_CALL}


@pytest.mark.parametrize("call", ALL_CALLS.values(), ids=list(ALL_CALLS))
def test_every_call_sends_the_client_id_header(call: Call) -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("x-client-id"))
        return error_response(404, "conversation_not_found")

    with pytest.raises(ApiError):
        call(make_api(handler))

    assert seen == [CID]


@pytest.mark.parametrize(
    "call",
    [
        lambda api: api.get_conversation("not-a-uuid"),
        lambda api: api.delete_conversation("../conversations"),
        lambda api: collect_from(api, f"{CONV}/../x"),
    ],
    ids=["get", "delete", "stream"],
)
def test_a_conversation_id_that_is_not_a_uuid_never_reaches_the_network(call: Call) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    with pytest.raises(ValueError, match="conversation ID"):
        call(make_api(handler))


def collect_from(api: HttpChatApi, conversation_id: str) -> None:
    with api.stream_answer(conversation_id, "How?") as stream:
        list(stream)


def test_plain_calls_use_short_timeouts_and_the_stream_uses_the_configured_read() -> None:
    timeouts: list[dict[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"])
        return error_response(404, "conversation_not_found")

    for call in ALL_CALLS.values():
        with pytest.raises(ApiError):
            call(make_api(handler))

    plain = {"connect": 2.0, "read": 10.0, "write": 5.0, "pool": 2.0}
    assert timeouts == [plain] * 4 + [{**plain, "read": 60.0}]


# ------------------------------------------------------------------ error responses


@pytest.mark.parametrize("call", ALL_CALLS.values(), ids=list(ALL_CALLS))
@pytest.mark.parametrize(
    ("status", "code"),
    [
        (400, "client_id_required"),
        (404, "conversation_not_found"),
        (409, "conversation_busy"),
        (422, "validation_error"),
        (500, "internal_error"),
        (502, "llm_error"),
        (503, "models_not_ready"),
    ],
)
def test_an_error_response_becomes_its_code_status_and_request_id(
    call: Call, status: int, code: str
) -> None:
    with pytest.raises(ApiError) as raised:
        call(make_api(lambda _: error_response(status, code)))

    error = raised.value
    assert (error.code, error.status, error.request_id) == (code, status, RID)
    assert error.retry_after_seconds is None
    assert CANARY not in repr(error.args)


@pytest.mark.parametrize("call", ALL_CALLS.values(), ids=list(ALL_CALLS))
def test_a_429_carries_its_retry_after(call: Call) -> None:
    api = make_api(lambda _: error_response(429, "rate_limited", **{"retry-after": "7"}))

    with pytest.raises(ApiError) as raised:
        call(api)

    assert (raised.value.code, raised.value.retry_after_seconds) == ("rate_limited", 7)


def test_an_invalid_retry_after_is_dropped() -> None:
    api = make_api(lambda _: error_response(429, "rate_limited", **{"retry-after": "soon"}))

    with pytest.raises(ApiError) as raised:
        api.list_conversations()

    assert (raised.value.code, raised.value.retry_after_seconds) == ("rate_limited", None)


def test_the_request_id_comes_from_the_body_when_the_header_is_missing() -> None:
    body = {"error": {"code": "internal_error", "message": "x", "request_id": "body-id-1234"}}

    with pytest.raises(ApiError) as raised:
        make_api(lambda _: httpx.Response(500, json=body)).list_conversations()

    assert raised.value.request_id == "body-id-1234"


def test_a_request_id_outside_the_allowlist_is_dropped() -> None:
    body = {"error": {"code": "internal_error", "message": "x", "request_id": CANARY + " <b>"}}
    headers = {"x-request-id": "bad id with spaces"}

    with pytest.raises(ApiError) as raised:
        make_api(lambda _: httpx.Response(500, json=body, headers=headers)).list_conversations()

    assert raised.value.request_id is None


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(502, text=f"<html>{CANARY}</html>"),
        httpx.Response(500, json={"detail": CANARY}),
        httpx.Response(500, json={"error": {"code": 7}}),
        httpx.Response(500, json={"error": {"code": f"{CANARY} DROP"}}),
        httpx.Response(500, json={"error": {"code": "Upper_Case"}}),
        httpx.Response(500, json={"error": {"code": "x" * 65}}),
        httpx.Response(302, headers={"location": "http://elsewhere"}),
    ],
    ids=["html", "other-shape", "code-not-text", "code-hostile", "code-case", "code-long", "3xx"],
)
def test_an_unreadable_error_response_is_bad_response_with_its_status(
    response: httpx.Response,
) -> None:
    with pytest.raises(ApiError) as raised:
        make_api(lambda _: response).list_conversations()

    assert (raised.value.code, raised.value.status) == (BAD_RESPONSE, response.status_code)


@pytest.mark.parametrize(
    ("call", "response"),
    [
        (CALLS["create"], httpx.Response(201, text=f"{{{CANARY}")),
        (CALLS["create"], httpx.Response(201, json={"id": CONV, "title": CANARY})),
        (CALLS["list"], httpx.Response(200, json={"items": [{"id": CANARY}]})),
        (CALLS["list"], httpx.Response(200, json=[CANARY])),
        (CALLS["get"], httpx.Response(200, json={**CONVERSATION_JSON, "messages": CANARY})),
        (CALLS["get"], httpx.Response(200, content=b"\xff\xfe" + CANARY.encode())),
        (CALLS["delete"], httpx.Response(200, json={"deleted": CANARY})),
    ],
    ids=[
        "create-json",
        "create-shape",
        "list-shape",
        "list-array",
        "get-shape",
        "get-bytes",
        "delete-200",
    ],
)
def test_a_malformed_success_body_is_bad_response_and_nothing_of_it_is_logged(
    call: Call, response: httpx.Response, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    with pytest.raises(ApiError) as raised:
        call(make_api(lambda _: response))

    error = raised.value
    assert (error.code, error.status) == (BAD_RESPONSE, response.status_code)
    assert_unchained(error)
    assert CANARY not in caplog.text


# ------------------------------------------------------------------ transport failures


def transport_errors(*errors: type[httpx.TransportError]) -> tuple[Handler, list[int]]:
    """A handler raising the given errors in turn, then answering the list call."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) <= len(errors):
            raise errors[len(calls) - 1](CANARY, request=request)
        return httpx.Response(200, json={"items": [], "next_cursor": None})

    return handler, calls


@pytest.mark.parametrize(
    "error", [httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError]
)
def test_a_get_is_retried_once_after_a_transport_failure(
    error: type[httpx.TransportError], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    handler, calls = transport_errors(error)

    assert make_api(handler).list_conversations() == ConversationPage(items=(), next_cursor=None)
    assert len(calls) == 2
    assert error.__name__ in caplog.text
    assert CANARY not in caplog.text


@pytest.mark.parametrize("name", ["list", "get"])
def test_a_get_failing_twice_is_backend_unreachable(name: str) -> None:
    handler, calls = transport_errors(httpx.ConnectTimeout, httpx.ReadTimeout)

    with pytest.raises(ApiError) as raised:
        CALLS[name](make_api(handler))

    assert raised.value.code == BACKEND_UNREACHABLE
    assert_unchained(raised.value)
    assert len(calls) == 2


def test_an_error_status_on_a_get_is_not_retried() -> None:
    calls: list[int] = []

    def handler(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        return error_response(503, "retrieval_unavailable")

    with pytest.raises(ApiError):
        make_api(handler).list_conversations()

    assert len(calls) == 1


@pytest.mark.parametrize("name", ["create", "delete"])
@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ReadTimeout])
def test_posts_and_deletes_are_never_retried(name: str, error: type[httpx.TransportError]) -> None:
    handler, calls = transport_errors(error)

    with pytest.raises(ApiError) as raised:
        CALLS[name](make_api(handler))

    assert raised.value.code == BACKEND_UNREACHABLE
    assert len(calls) == 1


# ------------------------------------------------------------------ the answer stream


def test_a_streamed_answer_yields_tokens_sources_and_done_then_closes() -> None:
    body = RecordingStream([token("Pull out "), token("the input tray [S1]."), SOURCES, DONE_FRAME])
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return sse(body)(request)

    events: list[StreamEvent] = []
    collect(make_api(handler), events, content="  How do I load paper? ")

    assert events == [
        Token(text="Pull out "),
        Token(text="the input tray [S1]."),
        SourcesEvent(sources=(SOURCE,)),
        DONE,
    ]
    (request,) = requests
    assert (request.method, str(request.url)) == (
        "POST",
        f"{BASE}/v1/conversations/{CONV}/messages",
    )
    assert json.loads(request.content) == {"content": "  How do I load paper? "}
    assert request.headers["accept"] == "text/event-stream"
    assert body.closed


@pytest.mark.parametrize(
    ("invalid_citations", "rewrite_fallback"),
    [(0, None), (2, "llm_error")],
    ids=["no-fallback", "fallback-code"],
)
def test_done_flags_are_read_as_the_backend_writes_them(
    invalid_citations: int, rewrite_fallback: str | None
) -> None:
    # A count and a code (backend app/domain/chat.py AnswerFlags), not booleans.
    flags: dict[str, object] = {
        **FLAGS_JSON,
        "invalid_citations": invalid_citations,
        "rewrite_fallback": rewrite_fallback,
    }
    body = RecordingStream([frame("done", {**DONE_JSON, "flags": flags})])

    events: list[StreamEvent] = []
    collect(make_api(sse(body)), events)

    (done,) = events
    assert isinstance(done, Done)
    assert (done.flags.invalid_citations, done.flags.rewrite_fallback) == (
        invalid_citations,
        rewrite_fallback,
    )


def test_a_refusal_streams_its_text_then_empty_sources_and_a_refused_done() -> None:
    refused = {
        **DONE_JSON,
        "outcome": "refused",
        "refusal": True,
        "refusal_reason": "below_threshold",
        "content": "fixed text",
        "citations": [],
    }
    body = RecordingStream(
        [token("fixed text"), frame("sources", {"sources": []}), frame("done", refused)]
    )

    events: list[StreamEvent] = []
    collect(make_api(sse(body)), events)

    assert events[1:] == [
        SourcesEvent(sources=()),
        DONE.model_copy(
            update={
                "outcome": "refused",
                "refusal": True,
                "refusal_reason": "below_threshold",
                "content": "fixed text",
                "citations": (),
            }
        ),
    ]


def test_unknown_events_are_skipped() -> None:
    body = RecordingStream(
        [frame("future", {"x": CANARY}), token("a"), b": keepalive\n\n", SOURCES, DONE_FRAME]
    )

    events: list[StreamEvent] = []
    collect(make_api(sse(body)), events)

    assert events == [Token(text="a"), SourcesEvent(sources=(SOURCE,)), DONE]


def test_a_mid_stream_error_event_raises_its_code_after_the_partial_tokens() -> None:
    error_event = frame(
        "error",
        {"code": "llm_timeout", "message": CANARY, "request_id": "stream-id-99", "message_id": "m"},
    )
    body = RecordingStream([token("Pull out "), error_event])

    error, events = stream_failure(make_api(sse(body)))

    assert events == [Token(text="Pull out ")]
    assert (error.code, error.status, error.request_id) == ("llm_timeout", 200, "stream-id-99")
    assert CANARY not in repr(error.args)
    assert body.closed


def test_a_mid_stream_error_with_an_unreadable_code_is_bad_response() -> None:
    error_event = frame("error", {"code": CANARY + "!", "request_id": None, "message_id": None})

    error, _ = stream_failure(make_api(sse(RecordingStream([token("a"), error_event]))))

    assert (error.code, error.request_id) == (BAD_RESPONSE, RID)


@pytest.mark.parametrize(
    "bad",
    [
        f"event: token\ndata: {{{CANARY}\n\n".encode(),
        frame("token", {"txt": CANARY}),
        frame("token", [CANARY]),
        frame("sources", {"sources": [{"tag": CANARY}]}),
        frame("done", {**DONE_JSON, "outcome": CANARY}),
        frame("error", {"message": CANARY}),
    ],
    ids=["json", "token-shape", "token-array", "sources-shape", "done-outcome", "error-shape"],
)
def test_an_unreadable_event_is_bad_response_and_nothing_of_it_is_logged(
    bad: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    body = RecordingStream([token("a"), bad, SOURCES, DONE_FRAME])

    error, events = stream_failure(make_api(sse(body)))

    assert events == [Token(text="a")]
    assert (error.code, error.status, error.request_id) == (BAD_RESPONSE, 200, RID)
    assert_unchained(error)
    assert CANARY not in caplog.text
    assert body.closed


@pytest.mark.parametrize(
    "chunks",
    [
        [token("Pull out "), token("the tray")],
        [token("Pull out "), SOURCES],
        [],
        [b": keepalive\n\n"],
    ],
    ids=["tokens-only", "sources-without-done", "zero-events", "keepalive-only"],
)
def test_a_stream_ending_before_done_or_error_is_bad_response_after_its_events(
    chunks: list[bytes],
) -> None:
    body = RecordingStream(chunks)

    error, events = stream_failure(make_api(sse(body)))

    assert (error.code, error.status, error.request_id) == (BAD_RESPONSE, 200, RID)
    assert len(events) == sum(1 for chunk in chunks if b"event:" in chunk)
    assert body.closed


@pytest.mark.parametrize(
    "terminal",
    [DONE_FRAME, frame("error", {"code": "llm_error", "request_id": RID})],
    ids=["done", "error"],
)
def test_the_first_terminal_event_ends_reading_and_closes_the_response(terminal: bytes) -> None:
    body = RecordingStream([token("a"), terminal, token("late"), DONE_FRAME])

    events: list[StreamEvent] = []
    expected: AbstractContextManager[object] = (
        pytest.raises(ApiError) if b"error" in terminal else nullcontext()
    )
    with expected:
        collect(make_api(sse(body)), events)

    assert Token(text="late") not in events
    assert body.pulled == 2
    assert body.closed


def test_events_after_done_in_the_same_chunk_are_not_yielded() -> None:
    body = RecordingStream([token("a") + DONE_FRAME + token("late") + DONE_FRAME])

    events: list[StreamEvent] = []
    collect(make_api(sse(body)), events)

    assert events == [Token(text="a"), DONE]


class ScriptStopped(BaseException):
    """Stands in for Streamlit's StopException, which is not an Exception."""


def test_a_consumer_stopped_mid_answer_closes_the_response_without_waiting_for_gc() -> None:
    body = RecordingStream([token("a"), token("b"), SOURCES, DONE_FRAME])
    api = make_api(sse(body))

    def consume() -> None:
        with api.stream_answer(CONV, "How?") as stream:
            for _ in stream:
                raise ScriptStopped

    with pytest.raises(ScriptStopped) as stopped:
        consume()

    # The traceback in `stopped` still holds the stream's frames, so only an
    # explicit close can have closed the response by now.
    assert stopped.traceback is not None
    assert body.closed
    assert body.pulled == 1


def test_a_truncated_last_frame_with_invalid_json_is_bad_response() -> None:
    body = RecordingStream([token("a"), b'event: done\ndata: {"conversation_id": "8a1e'])

    error, events = stream_failure(make_api(sse(body)))

    assert (error.code, events) == (BAD_RESPONSE, [Token(text="a")])


def test_a_truncated_last_frame_that_is_not_terminal_is_bad_response_after_it() -> None:
    body = RecordingStream([token("a"), token("b").rstrip(b"\n")])

    error, events = stream_failure(make_api(sse(body)))

    assert (error.code, events) == (BAD_RESPONSE, [Token(text="a"), Token(text="b")])


def test_a_truncated_last_frame_holding_a_whole_done_completes_the_answer() -> None:
    body = RecordingStream([token("a"), SOURCES, DONE_FRAME.rstrip(b"\n")])

    events: list[StreamEvent] = []
    collect(make_api(sse(body)), events)

    assert events[-1] == DONE


def test_a_frame_over_the_cap_is_bad_response() -> None:
    huge = b"event: token\ndata: " + b"x" * MAX_PENDING_FRAME_BYTES
    body = RecordingStream([token("a"), huge, SOURCES, DONE_FRAME])

    error, events = stream_failure(make_api(sse(body)))

    assert (error.code, events) == (BAD_RESPONSE, [Token(text="a")])
    assert_unchained(error)
    assert body.closed


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"content": CANARY}, headers={"x-request-id": RID}),
        httpx.Response(201, text="event: done\n\n", headers={"content-type": "text/event-stream"}),
    ],
    ids=["json-200", "sse-201"],
)
def test_a_success_that_is_not_an_event_stream_is_bad_response(response: httpx.Response) -> None:
    error, events = stream_failure(make_api(lambda _: response))

    assert (error.code, error.status, events) == (BAD_RESPONSE, response.status_code, [])


def test_a_pre_token_rate_limit_raises_before_any_event() -> None:
    api = make_api(lambda _: error_response(429, "rate_limited", **{"retry-after": "3"}))

    error, events = stream_failure(api)

    assert (error.code, error.retry_after_seconds, events) == ("rate_limited", 3, [])


@pytest.mark.parametrize(
    ("error_type", "code"),
    [
        (httpx.ConnectError, BACKEND_UNREACHABLE),
        (httpx.ConnectTimeout, BACKEND_UNREACHABLE),
        (httpx.ReadTimeout, STREAM_TIMEOUT),
        # The question never fully left: the service is unreachable, not slow to answer.
        (httpx.WriteTimeout, BACKEND_UNREACHABLE),
        (httpx.PoolTimeout, BACKEND_UNREACHABLE),
        (httpx.RemoteProtocolError, BACKEND_UNREACHABLE),
    ],
)
def test_a_transport_failure_before_the_response_is_mapped_and_never_retried(
    error_type: type[httpx.TransportError], code: str
) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise error_type(CANARY, request=request)

    error, _ = stream_failure(make_api(handler))

    assert (error.code, error.status) == (code, None)
    assert_unchained(error)
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("error_type", "code"),
    [(httpx.ReadTimeout, STREAM_TIMEOUT), (httpx.RemoteProtocolError, BACKEND_UNREACHABLE)],
)
def test_a_transport_failure_mid_stream_is_mapped_after_the_partial_tokens(
    error_type: type[httpx.TransportError], code: str
) -> None:
    request = httpx.Request("POST", BASE)
    body = RecordingStream([token("a"), error_type(CANARY, request=request)])

    error, events = stream_failure(make_api(sse(body)))

    assert (error.code, error.status, events) == (code, 200, [Token(text="a")])
    assert body.closed


def test_the_total_deadline_stops_a_stream_that_keeps_sending() -> None:
    now = [0.0]

    def clock() -> float:
        return now[0]

    def ticking() -> Iterator[bytes]:
        while True:
            now[0] += 60.0
            yield b": keepalive\n\n"
            yield token("x")

    body = RecordingStream(_take(ticking(), 20))

    error, events = stream_failure(make_api(sse(body), clock=clock))

    assert (error.code, error.status) == (STREAM_TIMEOUT, 200)
    assert now[0] == 240.0
    assert len(events) < 4
    assert body.closed


def test_a_stream_finishing_inside_the_deadline_is_not_cut() -> None:
    ticks = iter([0.0, 61.0, 122.0, 184.9])  # the start, then one reading per chunk

    def clock() -> float:
        return next(ticks, 184.9)

    body = RecordingStream([token("a"), SOURCES, DONE_FRAME])
    events: list[StreamEvent] = []
    collect(make_api(sse(body), clock=clock), events)

    assert events[-1] == DONE


def _take(chunks: Iterator[bytes], count: int) -> Iterator[bytes]:
    for _, chunk in zip(range(count), chunks, strict=False):
        yield chunk

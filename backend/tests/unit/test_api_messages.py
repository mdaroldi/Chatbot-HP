"""POST /v1/conversations/{id}/messages: priming, the SSE events, heartbeats, client
disconnects, rate limits and the error mapping.

The real ChatService runs over fakes of its ports (tests.unit.chat_rig). Disconnects
are driven through the ASGI interface directly, since httpx's ASGI transport only
returns once the app has finished.
"""

import asyncio
import inspect
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from starlette.types import Message, Scope

from app import main as main_module
from app.api.dependencies import get_chat_service, get_chat_store
from app.core.middleware import REQUEST_ID_HEADER
from app.core.settings import Settings
from app.domain.chat import REFUSAL_MARKER, REFUSAL_TEXT, Outcome
from app.domain.chat import Message as ChatMessage
from app.domain.chat_store import ConversationBusyError, ConversationNotFoundError
from app.domain.citations import snippet
from app.domain.llm import (
    LLMEmptyAnswerError,
    LLMError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
    StopReason,
)
from app.infra.llm_stub import StubProvider
from app.main import create_app
from app.services.chat import AnswerEvent, ChatService
from tests.unit.chat_rig import ANSWER_PIECES, CLIENT, CONFIG, HITS, TITLES, Rig, retrieval_config
from tests.unit.conftest import LogReader, SettingsFactory
from tests.unit.llm_fakes import FakeProvider
from tests.unit.retrieval_fakes import make_hit

REQUEST_ID = "test-request-0010"
OTHER = "client-bbbb-0002"
CANARY = "CANARY-5e1d-sse"
MISSING = "00000000-0000-4000-8000-00000000dead"
ANSWER = "Remove the cover [S1]. Then the fan [S2]."
QUESTION = "How do I open it?"
GUARD_SECONDS = 5.0  # no test here should take longer; a hang fails instead
INTERNAL_ERROR = ("internal_error", "Internal server error.")

# --- Building the app ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def keep_log_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests build their apps after json_logs is installed; create_app() would
    replace its handler, so the app's logging setup is skipped here.
    """
    monkeypatch.setattr(main_module, "configure_logging", lambda *_, **__: None)


@pytest.fixture
def settings(make_settings: SettingsFactory) -> Settings:
    return make_settings(rate_limit_messages_per_minute=6_000, rate_limit_messages_burst=1_000)


def build(
    settings: Settings, rig: Rig, service: Callable[[], ChatService] | None = None
) -> FastAPI:
    app = create_app(settings)
    app.dependency_overrides[get_chat_store] = lambda: rig.store
    # A lambda, not the bound method: FastAPI would read its parameters as query fields.
    app.dependency_overrides[get_chat_service] = service or (lambda: rig.service())  # noqa: PLW0108
    return app


@asynccontextmanager
async def serve(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={REQUEST_ID_HEADER: REQUEST_ID},
    ) as client:
        yield client


def as_client(client_id: str = CLIENT) -> dict[str, str]:
    return {"X-Client-ID": client_id}


def path(conversation_id: str) -> str:
    return f"/v1/conversations/{conversation_id}/messages"


async def post(
    http: httpx.AsyncClient,
    conversation_id: str,
    body: object | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return await http.post(
        path(conversation_id),
        json={"content": QUESTION} if body is None else body,
        headers=as_client() if headers is None else headers,
    )


# --- Reading the stream ----------------------------------------------------------------


@dataclass
class Stream:
    events: list[tuple[str, dict[str, Any]]]  # Any: the JSON payload of each event
    keepalives: int

    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def data(self, name: str) -> dict[str, Any]:  # Any: one event's JSON payload
        (payload,) = [data for event, data in self.events if event == name]
        return payload

    def text(self) -> str:
        return "".join(data["text"] for name, data in self.events if name == "token")


def parse(raw: str) -> Stream:
    """The frames as a Phase 4 client reads them: a `:` line is a comment."""
    assert raw.endswith("\n\n"), raw
    events: list[tuple[str, dict[str, Any]]] = []  # Any: JSON payloads
    keepalives = 0
    for frame in raw.split("\n\n")[:-1]:
        if frame == ": keepalive":
            keepalives += 1
            continue
        name, data = frame.split("\n")
        assert name.startswith("event: "), frame
        assert data.startswith("data: "), frame
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return Stream(events, keepalives)


async def ask(settings: Settings, rig: Rig, content: str = QUESTION) -> tuple[str, Stream]:
    conversation_id = await rig.conversation()
    async with serve(build(settings, rig)) as http:
        response = await post(http, conversation_id, {"content": content})
    assert response.status_code == 200, response.text
    return conversation_id, parse(response.text)


def assert_error(response: httpx.Response, status: int, code: str, message: str) -> None:
    assert response.status_code == status, response.text
    assert response.json() == {
        "error": {"code": code, "message": message, "request_id": REQUEST_ID, "details": []}
    }


async def stored(rig: Rig, conversation_id: str) -> tuple[ChatMessage, ...]:
    thread = await rig.store.get_thread(CLIENT, conversation_id)
    assert thread is not None
    return thread.messages


# --- A completed answer ----------------------------------------------------------------


async def test_an_answer_streams_tokens_then_sources_then_done(settings: Settings) -> None:
    rig = Rig()
    conversation_id = await rig.conversation()
    async with serve(build(settings, rig)) as http:
        response = await post(http, conversation_id)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers[REQUEST_ID_HEADER] == REQUEST_ID
    stream = parse(response.text)
    assert stream.names()[-2:] == ["sources", "done"]
    assert set(stream.names()[:-2]) == {"token"}
    assert stream.text().strip() == "".join(ANSWER_PIECES).strip()


async def test_done_carries_the_validated_answer_and_the_stored_ids(settings: Settings) -> None:
    rig = Rig()
    conversation_id, stream = await ask(settings, rig)

    user, assistant = await stored(rig, conversation_id)
    assert stream.data("done") == {
        "conversation_id": conversation_id,
        "user_message_id": user.id,
        "message_id": assistant.id,
        "outcome": "completed",
        "refusal": False,
        "refusal_reason": None,
        "content": ANSWER,
        "citations": ["S1", "S2"],
        "flags": {
            "late_refusal_marker": False,
            "uncited": False,
            "invalid_citations": 1,
            "rewrite_fallback": None,
            "truncated": False,
        },
    }
    assert (assistant.outcome, assistant.content) == (Outcome.COMPLETED, ANSWER)
    assert user.content == QUESTION


async def test_sources_are_the_cited_chunks_in_tag_order(settings: Settings) -> None:
    _, stream = await ask(settings, Rig())

    by_id = {hit.chunk_id: hit for hit in HITS}
    expected = [
        {
            "tag": tag,
            "document_id": hit.doc_id,
            "document_title": TITLES[hit.doc_id],
            "section_path": list(hit.section_path),
            "pages": {
                "pdf_start": hit.pages.pdf_start,
                "pdf_end": hit.pages.pdf_end,
                "printed_start": hit.pages.printed_start,
                "printed_end": hit.pages.printed_end,
            },
            "has_warning": hit.has_warning,
            "snippet": snippet(hit.text, CONFIG.snippet_chars),
        }
        for tag, hit in (("S1", by_id["c2"]), ("S2", by_id["c3"]))
    ]
    assert stream.data("sources") == {"sources": expected}


async def test_the_service_comes_from_the_factory_the_lifespan_installs(
    settings: Settings,
) -> None:
    rig = Rig()
    app = create_app(settings)

    async def factory() -> ChatService:
        return rig.service()

    app.state.chat_service_factory = factory
    conversation_id = await rig.conversation()
    async with serve(app) as http:
        response = await post(http, conversation_id)

    assert parse(response.text).data("done")["outcome"] == "completed"


async def test_the_question_is_stripped_and_may_use_the_whole_limit(
    make_settings: SettingsFactory,
) -> None:
    rig = Rig()
    settings = make_settings(chat_message_max_chars=12)

    conversation_id, _ = await ask(settings, rig, "  " + "x" * 12 + "\n")

    user, _ = await stored(rig, conversation_id)
    assert user.content == "x" * 12


async def test_a_completed_stream_closes_the_answer_events_explicitly(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After `done` the service's generator is still paused at its last event; the
    response closes it instead of leaving it to the garbage collector.
    """
    made: list[AsyncGenerator[AnswerEvent, None]] = []
    original = ChatService.answer

    def recording(
        self: ChatService, client_id: str, conversation_id: str, question: str
    ) -> AsyncGenerator[AnswerEvent, None]:
        events = original(self, client_id, conversation_id, question)
        made.append(events)
        return events

    monkeypatch.setattr(ChatService, "answer", recording)

    await ask(settings, Rig())

    (events,) = made
    assert inspect.getasyncgenstate(events) == inspect.AGEN_CLOSED


# --- Refusals --------------------------------------------------------------------------


async def test_a_question_below_the_threshold_is_refused_without_calling_the_model(
    settings: Settings,
) -> None:
    rig = Rig()
    config = retrieval_config(relevance_threshold=1_000.0)
    conversation_id = await rig.conversation()
    async with serve(build(settings, rig, lambda: rig.service(config))) as http:
        stream = parse((await post(http, conversation_id)).text)

    assert stream.names() == ["token", "sources", "done"]
    assert stream.text() == REFUSAL_TEXT
    assert stream.data("sources") == {"sources": []}
    done = stream.data("done")
    assert (done["outcome"], done["refusal"], done["refusal_reason"]) == (
        "refused",
        True,
        "below_threshold",
    )
    assert (done["content"], done["citations"]) == (REFUSAL_TEXT, [])
    assert rig.answer.requests == []


async def test_a_model_refusal_shows_the_fixed_text_and_never_the_marker(
    settings: Settings,
) -> None:
    rig = Rig(answer=FakeProvider(("  ", REFUSAL_MARKER)))
    conversation_id = await rig.conversation()
    async with serve(build(settings, rig)) as http:
        response = await post(http, conversation_id)

    assert REFUSAL_MARKER not in response.text
    stream = parse(response.text)
    assert stream.names() == ["token", "sources", "done"]
    assert stream.text() == REFUSAL_TEXT
    assert stream.data("sources") == {"sources": []}
    done = stream.data("done")
    assert (done["refusal"], done["refusal_reason"]) == (True, "model_refused")


# --- Failures after the first token: an error event, then the end --------------------


async def test_a_model_failure_mid_stream_ends_with_an_error_event(settings: Settings) -> None:
    rig = Rig(answer=FakeProvider(ANSWER_PIECES, error=LLMUnavailableError(), fail_at=1))
    conversation_id, stream = await ask(settings, rig)

    _, assistant = await stored(rig, conversation_id)
    assert stream.names()[-1] == "error"
    assert set(stream.names()[:-1]) == {"token"}
    assert stream.data("error") == {
        "code": LLMUnavailableError.code,
        "message": LLMUnavailableError.message,
        "request_id": REQUEST_ID,
        "message_id": assistant.id,
    }
    assert assistant.outcome is Outcome.FAILED
    assert rig.store.answers[assistant.id].error_code == "llm_unavailable"


async def test_a_provider_refusal_after_text_withdraws_the_answer(settings: Settings) -> None:
    rig = Rig(answer=FakeProvider(ANSWER_PIECES, stop=StopReason.REFUSAL))
    conversation_id, stream = await ask(settings, rig)

    _, assistant = await stored(rig, conversation_id)
    assert "done" not in stream.names()
    assert "sources" not in stream.names()
    error = stream.data("error")
    assert (error["code"], error["message"]) == ("answer_withdrawn", "The answer was withdrawn.")
    assert error["message_id"] == assistant.id
    assert (assistant.outcome, assistant.content) == (Outcome.REFUSED, REFUSAL_TEXT)


async def test_an_unexpected_failure_mid_stream_is_a_fixed_internal_error(
    settings: Settings, json_logs: LogReader
) -> None:
    rig = Rig(answer=FakeProvider(ANSWER_PIECES, error=RuntimeError(CANARY), fail_at=1))
    conversation_id, stream = await ask(settings, rig)

    _, assistant = await stored(rig, conversation_id)
    assert stream.names()[-1] == "error"
    code, message = INTERNAL_ERROR
    assert stream.data("error") == {
        "code": code,
        "message": message,
        "request_id": REQUEST_ID,
        "message_id": None,
    }
    assert assistant.outcome is Outcome.FAILED
    assert rig.store.answers[assistant.id].error_code == "internal_error"
    logs = json_logs()
    (line,) = [r for r in logs if r["message"] == "answer stream failed"]
    assert (line["level"], line["exception_type"]) == ("ERROR", "RuntimeError")
    assert line["request_id"] == REQUEST_ID
    assert CANARY not in json.dumps(logs)


# --- Failures before the first token: HTTP errors, nothing stored --------------------


@pytest.mark.parametrize(
    ("provider", "status", "error"),
    [
        (FakeProvider(ANSWER_PIECES, error=LLMUnavailableError()), 503, LLMUnavailableError),
        (FakeProvider(ANSWER_PIECES, error=LLMTimeoutError()), 503, LLMTimeoutError),
        (FakeProvider(ANSWER_PIECES, error=LLMRequestError()), 502, LLMRequestError),
        (FakeProvider(()), 502, LLMEmptyAnswerError),
        (FakeProvider(ANSWER_PIECES, error=LLMError()), 502, LLMError),
    ],
    ids=["unavailable", "timeout", "rejected", "empty", "any-other-model-error"],
)
async def test_a_model_failure_before_the_first_token_is_an_http_error_and_stores_nothing(
    settings: Settings,
    json_logs: LogReader,
    provider: FakeProvider,
    status: int,
    error: type[LLMError],
) -> None:
    rig = Rig(answer=provider)
    conversation_id = await rig.conversation()
    async with serve(build(settings, rig)) as http:
        response = await post(http, conversation_id)

    assert_error(response, status, error.code, error.message)
    assert await stored(rig, conversation_id) == ()
    (line,) = [r for r in json_logs() if r["message"] == "turn discarded"]
    assert (line["error_code"], line["request_id"]) == (error.code, REQUEST_ID)


async def test_an_excerpt_of_an_unknown_document_is_a_500_with_its_own_code(
    settings: Settings,
) -> None:
    rig = Rig(hits=[make_hit("c9", doc_id="not-in-the-manifest", text="Crumb\n\nBody.")])
    conversation_id = await rig.conversation()
    async with serve(build(settings, rig)) as http:
        response = await post(http, conversation_id)

    assert_error(
        response,
        500,
        "unknown_document",
        "The index holds a document this service does not know.",
    )
    assert await stored(rig, conversation_id) == ()


async def test_an_unknown_or_foreign_conversation_is_not_found(settings: Settings) -> None:
    rig = Rig()
    foreign = (await rig.store.create_conversation(OTHER)).id
    async with serve(build(settings, rig)) as http:
        missing = await post(http, MISSING)
        theirs = await post(http, foreign)

    for response in (missing, theirs):
        assert_error(response, 404, "conversation_not_found", ConversationNotFoundError.message)
    assert rig.answer.requests == []


async def test_a_conversation_already_answering_is_busy(settings: Settings) -> None:
    rig = Rig()
    conversation_id = await rig.conversation()
    await rig.store.start_turn(CLIENT, conversation_id, "First?", title="First?")
    async with serve(build(settings, rig)) as http:
        response = await post(http, conversation_id)

    assert_error(response, 409, "conversation_busy", ConversationBusyError.message)


async def test_the_client_id_is_required(settings: Settings) -> None:
    rig = Rig()
    conversation_id = await rig.conversation()
    async with serve(build(settings, rig)) as http:
        response = await post(http, conversation_id, headers={})

    assert_error(response, 400, "client_id_required", "A valid X-Client-ID header is required.")


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"content": ""},
        {"content": " \n\t "},
        {"content": CANARY * 100},
        {"content": CANARY, "role": "system"},
        {"content": 42},
    ],
    ids=["missing", "empty", "blank", "too-long", "extra-field", "not-text"],
)
async def test_an_invalid_body_is_rejected_without_echoing_it(
    settings: Settings, body: dict[str, object]
) -> None:
    rig = Rig()
    conversation_id = await rig.conversation()
    async with serve(build(settings, rig)) as http:
        response = await post(http, conversation_id, body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert CANARY not in response.text
    assert "role" not in response.text
    assert await stored(rig, conversation_id) == ()


async def test_the_default_length_limit_is_the_chat_message_setting(
    make_settings: SettingsFactory,
) -> None:
    rig = Rig()
    settings = make_settings(chat_message_max_chars=5)
    conversation_id = await rig.conversation()
    async with serve(build(settings, rig)) as http:
        response = await post(http, conversation_id, {"content": "x" * 6})

    assert response.status_code == 422


async def test_a_malformed_conversation_id_is_rejected(settings: Settings) -> None:
    async with serve(build(settings, Rig())) as http:
        response = await post(http, "not-a-uuid")

    assert response.status_code == 422


# --- Rate limit ------------------------------------------------------------------------


async def test_messages_are_rate_limited_per_client_with_retry_after(
    make_settings: SettingsFactory,
) -> None:
    rig = Rig()
    settings = make_settings(rate_limit_messages_per_minute=60, rate_limit_messages_burst=2)
    conversation_id = await rig.conversation()
    theirs = (await rig.store.create_conversation(OTHER)).id
    async with serve(build(settings, rig)) as http:
        allowed = [(await post(http, conversation_id)).status_code for _ in range(2)]
        limited = await post(http, conversation_id)
        other = await post(http, theirs, headers=as_client(OTHER))
        listing = await http.get("/v1/conversations", headers=as_client())

    assert allowed == [200, 200]
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "1"
    assert limited.json()["error"]["code"] == "rate_limited"
    assert len(await stored(rig, conversation_id)) == 4  # the limited one stored nothing
    assert other.status_code == 200
    # Messages have their own bucket: the conversation routes are not spent.
    assert listing.status_code == 200


# --- Heartbeats ------------------------------------------------------------------------


async def test_a_slow_model_gets_keepalives_and_still_completes(
    make_settings: SettingsFactory,
) -> None:
    """The wait for the next event times out into a keepalive without cancelling it:
    a cancelled wait would cancel the answer, which would end `cancelled`.
    """
    rig = Rig()
    slow = StubProvider(honour_directives=False, token_delay_ms=40)
    settings = make_settings(sse_heartbeat_seconds=0.005)
    conversation_id = await rig.conversation()
    async with serve(build(settings, rig, lambda: rig.service(provider=slow))) as http:
        response = await post(http, conversation_id)

    stream = parse(response.text)
    assert stream.keepalives > 0
    assert stream.data("done")["outcome"] == "completed"
    _, assistant = await stored(rig, conversation_id)
    assert assistant.outcome is Outcome.COMPLETED


async def test_a_fast_answer_has_no_keepalives(settings: Settings) -> None:
    _, stream = await ask(settings, Rig())

    assert stream.keepalives == 0


# --- Client disconnects (raw ASGI) -----------------------------------------------------


@dataclass
class Peer:
    """An ASGI client: sends the body, then disconnects once `leave` is set.

    `before_leaving` are messages delivered after `leave` and before the disconnect.
    With `block_on_first_body`, the first body chunk is never acknowledged (a stalled
    socket); `leave` is then set by that chunk.
    """

    body: bytes
    leave: asyncio.Event
    before_leaving: list[Message]
    block_on_first_body: bool = False

    def __post_init__(self) -> None:
        self.sent: list[Message] = []
        self._request_sent = False

    async def receive(self) -> Message:
        if not self._request_sent:
            self._request_sent = True
            return {"type": "http.request", "body": self.body, "more_body": False}
        await self.leave.wait()
        if self.before_leaving:
            return self.before_leaving.pop(0)
        return {"type": "http.disconnect"}

    async def send(self, message: Message) -> None:
        self.sent.append(message)
        if self.block_on_first_body and message["type"] == "http.response.body":
            self.leave.set()
            await asyncio.Event().wait()

    def body_text(self) -> str:
        return b"".join(m.get("body", b"") for m in self.sent).decode()


def scope_for(conversation_id: str, body: bytes) -> Scope:
    url = path(conversation_id)
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": url,
        "raw_path": url.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"test"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"x-client-id", CLIENT.encode()),
            (REQUEST_ID_HEADER.encode(), REQUEST_ID.encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("test", 80),
    }


async def run_until_disconnect(
    settings: Settings,
    rig: Rig,
    leave: asyncio.Event,
    *,
    before_leaving: list[Message] | None = None,
    block_on_first_body: bool = False,
) -> tuple[str, Peer]:
    conversation_id = await rig.conversation()
    body = json.dumps({"content": QUESTION}).encode()
    peer = Peer(body, leave, list(before_leaving or []), block_on_first_body)
    async with asyncio.timeout(GUARD_SECONDS):
        await build(settings, rig)(scope_for(conversation_id, body), peer.receive, peer.send)
    return conversation_id, peer


@pytest.mark.parametrize(
    "before_leaving",
    [[], [{"type": "http.request", "body": b"", "more_body": False}]],
    ids=["disconnect", "other-message-first"],
)
async def test_a_disconnect_before_the_first_token_cancels_the_model_and_stores_nothing(
    settings: Settings, json_logs: LogReader, before_leaving: list[Message]
) -> None:
    rig = Rig(answer=FakeProvider(ANSWER_PIECES, stall=True))

    conversation_id, peer = await run_until_disconnect(
        settings, rig, rig.answer.stalled, before_leaving=before_leaving
    )

    assert peer.sent == []
    assert rig.answer.closed == 1
    assert await stored(rig, conversation_id) == ()
    logs = json_logs()
    (line,) = [r for r in logs if r["message"] == "turn discarded"]
    assert (line["error_code"], line["request_id"]) == ("client_disconnected", REQUEST_ID)
    (access,) = [r for r in logs if r["logger"] == "app.access"]
    assert access["status"] == 499


async def test_a_disconnect_while_waiting_for_the_next_token_stores_the_answer_cancelled(
    settings: Settings,
) -> None:
    """Checked as soon as the app returns: the open wait for the next event is
    cancelled and awaited, and the events closed, before the response ends.
    """
    rig = Rig(answer=FakeProvider(ANSWER_PIECES, stall_at=1))

    conversation_id, peer = await run_until_disconnect(settings, rig, rig.answer.stalled)

    assert peer.sent[0]["status"] == 200
    assert parse(peer.body_text()).names() == ["token"]
    assert rig.answer.closed == 1
    _, assistant = await stored(rig, conversation_id)
    assert assistant.outcome is Outcome.CANCELLED
    assert assistant.content == "Remove the cover [S1]."


async def test_a_disconnect_while_a_token_is_being_sent_stores_the_answer_cancelled(
    settings: Settings,
) -> None:
    """The response is cancelled inside `send`, with the events paused at a token:
    only the endpoint's explicit close stores the outcome before the app returns.
    """
    rig = Rig()

    conversation_id, _ = await run_until_disconnect(
        settings, rig, asyncio.Event(), block_on_first_body=True
    )

    assert rig.answer.closed == 1
    _, assistant = await stored(rig, conversation_id)
    assert assistant.outcome is Outcome.CANCELLED


async def test_canary_audit_no_log_line_ever_carries_user_or_model_text(
    settings: Settings, json_logs: LogReader
) -> None:
    """The real stub answers; questions, the title, the rewrite and the answers all hold
    the canary, through a completed, a rewritten, a refused and a failed turn.
    """
    rig = Rig(rewrite=FakeProvider((f"rewritten {CANARY}",)))
    stub = StubProvider(honour_directives=True)
    conversation_id = await rig.conversation()
    async with serve(build(settings, rig, lambda: rig.service(provider=stub))) as http:
        outcomes = []
        for content in (
            f"How do I clean the fan? {CANARY}",
            f"And then? {CANARY}",
            f"[stub:refuse] {CANARY}",
            f"[stub:fail-after] {CANARY}",
        ):
            response = await post(http, conversation_id, {"content": content})
            assert response.status_code == 200, response.text
            stream = parse(response.text)
            outcomes.append(stream.names()[-1])

    assert outcomes == ["done", "done", "done", "error"]
    rewrites = {answer.rewritten_query for answer in rig.store.answers.values()}
    assert f"rewritten {CANARY}" in rewrites  # stored with the answer, never logged
    logs = json_logs()
    assert len(logs) >= 8  # access lines and one answer line per turn at least
    assert CANARY not in json.dumps(logs)

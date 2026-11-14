"""A scripted ChatApi for page tests: canned results per call, and a record of calls."""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime

from hpdocs_ui.api import (
    Conversation,
    ConversationDetail,
    ConversationPage,
    Done,
    Flags,
    Message,
    Outcome,
    Pages,
    RefusalReason,
    Source,
    StreamEvent,
    Token,
)
from hpdocs_ui.errors import ApiError
from hpdocs_ui.status import BackendStatus

NOW = datetime(2026, 11, 14, 9, 0, tzinfo=UTC)
CID = "client-0001"
A = "6f1c2d3e-4b5a-4c6d-8e7f-90a1b2c3d4e5"
B = "0b7e7a0c-2f7d-4f43-9d3f-6f2b8f1d2c3a"
C = "1c9d8e7f-6a5b-4c3d-9e2f-1a0b9c8d7e6f"

EMPTY_PAGE = ConversationPage(items=(), next_cursor=None)


def conversation(conversation_id: str, title: str | None = "Title") -> Conversation:
    return Conversation(id=conversation_id, title=title, created_at=NOW, updated_at=NOW)


def page(*items: Conversation, next_cursor: str | None = None) -> ConversationPage:
    return ConversationPage(items=items, next_cursor=next_cursor)


def tokens(*texts: str) -> tuple[Token, ...]:
    return tuple(Token(text=text) for text in texts)


def done(
    content: str,
    conversation_id: str = A,
    *,
    uncited: bool = False,
    reason: RefusalReason | None = None,
) -> Done:
    flags = Flags(
        late_refusal_marker=False,
        uncited=uncited,
        invalid_citations=False,
        rewrite_fallback=False,
        truncated=False,
    )
    return Done(
        conversation_id=conversation_id,
        user_message_id="user-message",
        message_id="assistant-message",
        outcome="completed" if reason is None else "refused",
        refusal=reason is not None,
        refusal_reason=reason,
        content=content,
        citations=() if uncited or reason else ("S1",),
        flags=flags,
    )


def source(
    tag: str = "S1",
    *,
    title: str = "HP ENVY 6000 User Guide",
    section_path: tuple[str, ...] = ("Print", "Print documents"),
    has_warning: bool = False,
    snippet: str = "Load paper, then press OK.",
) -> Source:
    return Source(
        tag=tag,
        document_id="envy-6000",
        document_title=title,
        section_path=section_path,
        pages=Pages(pdf_start=40, pdf_end=40, printed_start=38, printed_end=38),
        has_warning=has_warning,
        snippet=snippet,
    )


def asked(content: str) -> Message:
    return Message(
        id="user-message",
        role="user",
        content=content,
        created_at=NOW,
        outcome=None,
        refusal=False,
        refusal_reason=None,
        citations=(),
        sources=(),
    )


def answered(
    content: str,
    outcome: Outcome = "completed",
    *,
    reason: RefusalReason | None = None,
    citations: tuple[str, ...] = (),
    sources: tuple[Source, ...] = (),
) -> Message:
    return Message(
        id="assistant-message",
        role="assistant",
        content=content,
        created_at=NOW,
        outcome=outcome,
        refusal=outcome == "refused",
        refusal_reason=reason,
        citations=citations,
        sources=sources,
    )


def detail(conversation_id: str, *messages: Message) -> ConversationDetail:
    return ConversationDetail(
        id=conversation_id, title="Title", created_at=NOW, updated_at=NOW, messages=messages
    )


# The events one POST /messages yields, then the error it raises (if any). Errors
# other than ApiError stand for bugs, which only the page's guard catches.
StreamScript = tuple[tuple[StreamEvent, ...], Exception | None]


class FakeChatApi:
    """Implements the ChatApi Protocol. A call nobody scripted fails the test."""

    def __init__(self) -> None:
        self.pages: dict[str | None, ConversationPage | Exception] = {None: EMPTY_PAGE}
        self.delete_errors: dict[str, Exception] = {}
        self.created: list[Conversation | ApiError] = []
        self.details: dict[str, ConversationDetail | ApiError] = {}
        self.streams: list[StreamScript] = []
        # Called inside the page run when the page starts reading an answer, and when
        # it closes the stream.
        self.on_stream_start: Callable[[], None] | None = None
        self.on_stream_closed: Callable[[], None] | None = None
        self.questions: list[str] = []
        self.streams_closed = 0
        self.calls: list[tuple[str, str | None]] = []
        self.client_ids: list[str] = []

    def count(self, name: str) -> int:
        return sum(1 for call, _ in self.calls if call == name)

    def create_conversation(self) -> Conversation:
        self.calls.append(("create", None))
        if not self.created:
            raise AssertionError("create_conversation was not scripted")
        result = self.created.pop(0)
        if isinstance(result, ApiError):
            raise result
        return result

    def list_conversations(self, cursor: str | None = None) -> ConversationPage:
        self.calls.append(("list", cursor))
        result = self.pages[cursor]
        if isinstance(result, Exception):
            raise result
        return result

    def get_conversation(self, conversation_id: str) -> ConversationDetail:
        self.calls.append(("get", conversation_id))
        if conversation_id not in self.details:
            raise AssertionError(f"get_conversation({conversation_id}) was not scripted")
        result = self.details[conversation_id]
        if isinstance(result, ApiError):
            raise result
        return result

    def delete_conversation(self, conversation_id: str) -> None:
        self.calls.append(("delete", conversation_id))
        if conversation_id in self.delete_errors:
            raise self.delete_errors[conversation_id]

    def stream_answer(
        self, conversation_id: str, content: str
    ) -> AbstractContextManager[Iterator[StreamEvent]]:
        self.calls.append(("stream", conversation_id))
        self.questions.append(content)
        if not self.streams:
            raise AssertionError(f"stream_answer({conversation_id}) was not scripted")
        events, error = self.streams.pop(0)
        return self._open(events, error)

    @contextmanager
    def _open(
        self, events: tuple[StreamEvent, ...], error: Exception | None
    ) -> Iterator[Iterator[StreamEvent]]:
        try:
            yield self._events(events, error)
        finally:
            self.streams_closed += 1
            if self.on_stream_closed is not None:
                self.on_stream_closed()

    def _events(
        self, events: tuple[StreamEvent, ...], error: Exception | None
    ) -> Iterator[StreamEvent]:
        if self.on_stream_start is not None:
            self.on_stream_start()
        yield from events
        if error is not None:
            raise error


class FakeFetch:
    """Stands in for status.fetch_status: one canned result per path."""

    def __init__(self, results: dict[str, BackendStatus]) -> None:
        self.results = results
        self.calls: list[tuple[str, str]] = []

    def __call__(self, base_url: str, path: str) -> BackendStatus:
        self.calls.append((base_url, path))
        return self.results[path]


UP = {
    "/health": BackendStatus(True, 200, {"status": "ok"}, "rid-12345678"),
    "/ready": BackendStatus(
        True, 200, {"status": "ready", "checks": {"postgres": {"status": "ok"}}}, "rid-12345678"
    ),
}
NOT_READY = {
    "/health": UP["/health"],
    "/ready": BackendStatus(
        True, 503, {"status": "not_ready", "checks": {"qdrant": {"status": "down"}}}, None
    ),
}
DOWN_STATUS = BackendStatus(reachable=False, http_status=None, body=None, request_id=None)
DOWN = {"/health": DOWN_STATUS, "/ready": DOWN_STATUS}


def up(llm: object) -> dict[str, BackendStatus]:
    """The backend up and ready, its /ready naming the language model provider."""
    ready = UP["/ready"]
    body = {**(ready.body or {}), "llm": llm}
    return {"/health": UP["/health"], "/ready": BackendStatus(True, 200, body, ready.request_id)}

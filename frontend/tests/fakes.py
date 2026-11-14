"""A scripted ChatApi for page tests: canned results per call, and a record of calls."""

from collections.abc import Iterator
from contextlib import AbstractContextManager
from datetime import UTC, datetime

from hpdocs_ui.api import Conversation, ConversationDetail, ConversationPage, StreamEvent
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


class FakeChatApi:
    """Implements the ChatApi Protocol. A call nobody scripted fails the test."""

    def __init__(self) -> None:
        self.pages: dict[str | None, ConversationPage | ApiError] = {None: EMPTY_PAGE}
        self.delete_errors: dict[str, ApiError] = {}
        self.calls: list[tuple[str, str | None]] = []
        self.client_ids: list[str] = []

    def count(self, name: str) -> int:
        return sum(1 for call, _ in self.calls if call == name)

    def create_conversation(self) -> Conversation:
        raise AssertionError("create_conversation was not scripted")

    def list_conversations(self, cursor: str | None = None) -> ConversationPage:
        self.calls.append(("list", cursor))
        result = self.pages[cursor]
        if isinstance(result, ApiError):
            raise result
        return result

    def get_conversation(self, conversation_id: str) -> ConversationDetail:
        raise AssertionError(f"get_conversation({conversation_id}) was not scripted")

    def delete_conversation(self, conversation_id: str) -> None:
        self.calls.append(("delete", conversation_id))
        if conversation_id in self.delete_errors:
            raise self.delete_errors[conversation_id]

    def stream_answer(
        self, conversation_id: str, content: str
    ) -> AbstractContextManager[Iterator[StreamEvent]]:
        raise AssertionError(f"stream_answer({conversation_id}, {content!r}) was not scripted")


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

"""What the page needs from the chat API: the calls, and the shapes they return.

The models mirror the backend's responses (ADR 0010) but ignore fields they do not
know, so a backend that adds a field does not break the page. They are frozen:
views and state hold them without copying.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

Outcome = Literal["pending", "completed", "refused", "failed", "cancelled"]
RefusalReason = Literal["below_threshold", "model_refused", "provider_refused"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class Pages(_Model):
    pdf_start: int
    pdf_end: int
    printed_start: int | None
    printed_end: int | None


class Source(_Model):
    tag: str
    document_id: str
    document_title: str
    section_path: tuple[str, ...]
    pages: Pages
    has_warning: bool
    snippet: str


class Conversation(_Model):
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


class ConversationPage(_Model):
    items: tuple[Conversation, ...]
    next_cursor: str | None


class Message(_Model):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    outcome: Outcome | None
    refusal: bool
    refusal_reason: RefusalReason | None
    citations: tuple[str, ...]
    sources: tuple[Source, ...]


class ConversationDetail(Conversation):
    messages: tuple[Message, ...]


class Token(_Model):
    text: str


class SourcesEvent(_Model):
    sources: tuple[Source, ...]


class Flags(_Model):
    late_refusal_marker: bool
    uncited: bool
    invalid_citations: bool
    rewrite_fallback: bool
    truncated: bool


class Done(_Model):
    conversation_id: str
    user_message_id: str
    message_id: str
    outcome: Outcome
    refusal: bool
    refusal_reason: RefusalReason | None
    content: str
    citations: tuple[str, ...]
    flags: Flags


StreamEvent = Token | SourcesEvent | Done


class ChatApi(Protocol):
    """The backend as the page sees it. Every failure is raised as an ApiError."""

    def create_conversation(self) -> Conversation: ...

    def list_conversations(self, cursor: str | None = None) -> ConversationPage: ...

    def get_conversation(self, conversation_id: str) -> ConversationDetail: ...

    def delete_conversation(self, conversation_id: str) -> None: ...

    def stream_answer(
        self, conversation_id: str, content: str
    ) -> AbstractContextManager[Iterator[StreamEvent]]:
        """The answer's events, ending with Done; a failure raises after the events
        already yielded, so the caller keeps the partial text it has drawn.

        Leaving the context closes the connection, also on an early exit, which the
        backend records as a cancelled answer.
        """
        ...

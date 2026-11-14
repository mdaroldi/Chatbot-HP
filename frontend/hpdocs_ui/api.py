"""What the page needs from the chat API: the calls, and the shapes they return.

The models are tolerant readers of the backend's responses (ADR 0010, ADR 0011):
they declare only the fields the page reads and ignore every other field, whatever
it holds, so a backend change to a field the page never shows cannot break it.
They are frozen: views and state hold them without copying.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, StrictBool

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
    document_title: str
    section_path: tuple[str, ...]
    pages: Pages
    has_warning: bool
    snippet: str


class Conversation(_Model):
    id: str
    title: str | None


class ConversationPage(_Model):
    items: tuple[Conversation, ...]
    next_cursor: str | None


class Message(_Model):
    role: Literal["user", "assistant"]
    content: str
    outcome: Outcome | None
    refusal_reason: RefusalReason | None
    citations: tuple[str, ...]
    sources: tuple[Source, ...]


class ConversationDetail(_Model):
    messages: tuple[Message, ...]


class Token(_Model):
    text: str


class SourcesEvent(_Model):
    sources: tuple[Source, ...]


class Flags(_Model):
    # Strict: "true" or 1 would otherwise pass as True and hide a changed contract.
    uncited: StrictBool


class Done(_Model):
    outcome: Outcome
    refusal_reason: RefusalReason | None
    content: str
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

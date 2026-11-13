"""The chat store port: conversations and their messages, always scoped to one client.

Every method takes the client ID first and the implementation applies it inside each
statement, so a conversation of another client behaves exactly like one that does not
exist (the API answers 404, never 403).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from app.domain.chat import AnswerRecord, Conversation, Message, Outcome, Role


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ConversationCursor:
    """Keyset position in the newest-first list: the last item seen."""

    updated_at: datetime
    id: str


@dataclass(frozen=True)
class ConversationPage:
    items: tuple[Conversation, ...]
    next_cursor: ConversationCursor | None


@dataclass(frozen=True)
class ConversationThread:
    conversation: Conversation
    messages: tuple[Message, ...]  # in sequence order


@dataclass(frozen=True)
class Turn:
    """A question and the pending answer row reserved for it, committed together."""

    user: Message
    assistant: Message

    def __post_init__(self) -> None:
        _require(
            self.user.role is Role.USER
            and self.assistant.role is Role.ASSISTANT
            and self.assistant.outcome is Outcome.PENDING
            and self.assistant.seq == self.user.seq + 1
            and self.assistant.conversation_id == self.user.conversation_id,
            "a turn is a user message and its pending answer",
        )


class ChatStoreError(Exception):
    """Base of the store's expected failures; code and message are fixed and safe to send."""

    code = "chat_store_error"
    message = "The conversation store failed."

    def __init__(self) -> None:
        super().__init__(self.message)


class ConversationNotFoundError(ChatStoreError):
    """Absent, or owned by another client: the two are deliberately indistinguishable."""

    code = "conversation_not_found"
    message = "Conversation not found."


class ConversationBusyError(ChatStoreError):
    """The conversation already has an answer in progress (one at a time)."""

    code = "conversation_busy"
    message = "The conversation is still answering the previous message."


class ChatStore(Protocol):
    async def create_conversation(self, client_id: str) -> Conversation: ...

    async def list_conversations(
        self, client_id: str, *, limit: int, after: ConversationCursor | None = None
    ) -> ConversationPage:
        """Newest first by last update; next_cursor is None on the last page."""
        ...

    async def get_thread(
        self, client_id: str, conversation_id: str
    ) -> ConversationThread | None: ...

    async def delete_conversation(self, client_id: str, conversation_id: str) -> bool:
        """True when it existed for this client; its messages go with it."""
        ...

    async def start_turn(
        self, client_id: str, conversation_id: str, content: str, *, title: str | None
    ) -> Turn:
        """Commit the question and a pending answer row before any slow work.

        The title is stored only if the conversation has none yet. A pending answer
        older than the store's abandon window is first marked failed ("abandoned").
        Raises ConversationNotFoundError or ConversationBusyError.
        """
        ...

    async def discard_turn(self, client_id: str, turn: Turn) -> None:
        """Remove a turn that failed before its first token, as if never sent."""
        ...

    async def finish_answer(
        self, client_id: str, message_id: str, answer: AnswerRecord
    ) -> Message | None:
        """Fill the pending answer; None when there is no such pending row any more."""
        ...

"""An in-memory ChatStore: the port's observable behaviour without a database.

It scopes everything by client ID exactly like SqlChatStore, so API and service tests
can rely on ownership without SQL. `calls` records which methods ran.
"""

import dataclasses
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from app.domain.chat import AnswerRecord, Conversation, Message, Outcome, Role
from app.domain.chat_store import (
    ConversationBusyError,
    ConversationCursor,
    ConversationNotFoundError,
    ConversationPage,
    ConversationThread,
    Turn,
)

T0 = datetime(2026, 11, 14, 9, 0, tzinfo=UTC)


class InMemoryChatStore:
    def __init__(self, clock: Callable[[], datetime] = lambda: T0) -> None:
        self._clock = clock
        self._owners: dict[str, str] = {}
        self._conversations: dict[str, Conversation] = {}
        self._messages: dict[str, list[Message]] = {}
        self.calls: list[str] = []
        self.answers: dict[str, AnswerRecord] = {}  # what finish_answer stored, by message

    def _owned(self, client_id: str, conversation_id: str) -> Conversation | None:
        if self._owners.get(conversation_id) != client_id:
            return None
        return self._conversations[conversation_id]

    def _touch(self, conversation_id: str, **changes: object) -> None:
        current = self._conversations[conversation_id]
        changes.setdefault("updated_at", max(self._clock(), current.updated_at))
        self._conversations[conversation_id] = dataclasses.replace(current, **changes)  # type: ignore[arg-type]

    async def create_conversation(self, client_id: str) -> Conversation:
        self.calls.append("create_conversation")
        now = self._clock()
        conversation = Conversation(str(uuid.uuid4()), None, now, now)
        self._owners[conversation.id] = client_id
        self._conversations[conversation.id] = conversation
        self._messages[conversation.id] = []
        return conversation

    async def list_conversations(
        self, client_id: str, *, limit: int, after: ConversationCursor | None = None
    ) -> ConversationPage:
        self.calls.append("list_conversations")
        owned = [c for c in self._conversations.values() if self._owners[c.id] == client_id]
        owned.sort(key=lambda c: (c.updated_at, c.id), reverse=True)
        if after is not None:
            owned = [c for c in owned if (c.updated_at, c.id) < (after.updated_at, after.id)]
        items = tuple(owned[:limit])
        more = len(owned) > limit
        cursor = ConversationCursor(items[-1].updated_at, items[-1].id) if more else None
        return ConversationPage(items, cursor)

    async def get_thread(self, client_id: str, conversation_id: str) -> ConversationThread | None:
        self.calls.append("get_thread")
        conversation = self._owned(client_id, conversation_id)
        if conversation is None:
            return None
        return ConversationThread(conversation, tuple(self._messages[conversation_id]))

    async def delete_conversation(self, client_id: str, conversation_id: str) -> bool:
        self.calls.append("delete_conversation")
        if self._owned(client_id, conversation_id) is None:
            return False
        del self._owners[conversation_id], self._conversations[conversation_id]
        del self._messages[conversation_id]
        return True

    async def start_turn(
        self, client_id: str, conversation_id: str, content: str, *, title: str | None
    ) -> Turn:
        self.calls.append("start_turn")
        conversation = self._owned(client_id, conversation_id)
        if conversation is None:
            raise ConversationNotFoundError
        messages = self._messages[conversation_id]
        if any(m.outcome is Outcome.PENDING for m in messages):
            raise ConversationBusyError
        now = self._clock()
        seq = len(messages) + 1
        turn = Turn(
            Message(str(uuid.uuid4()), conversation_id, seq, Role.USER, content, now),
            Message(
                str(uuid.uuid4()),
                conversation_id,
                seq + 1,
                Role.ASSISTANT,
                "",
                now,
                outcome=Outcome.PENDING,
            ),
        )
        messages.extend((turn.user, turn.assistant))
        self._touch(conversation_id, title=conversation.title or title)
        return turn

    async def discard_turn(self, client_id: str, turn: Turn) -> None:
        self.calls.append("discard_turn")
        conversation_id = turn.user.conversation_id
        if self._owned(client_id, conversation_id) is None:
            return
        ids = {turn.user.id, turn.assistant.id}
        remaining = [m for m in self._messages[conversation_id] if m.id not in ids]
        self._messages[conversation_id] = remaining
        if not remaining:
            self._touch(conversation_id, title=None)

    async def finish_answer(
        self, client_id: str, message_id: str, answer: AnswerRecord
    ) -> Message | None:
        self.calls.append("finish_answer")
        for conversation_id, messages in self._messages.items():
            for index, message in enumerate(messages):
                if message.id != message_id:
                    continue
                if self._owners[conversation_id] != client_id:
                    return None
                if message.outcome is not Outcome.PENDING:
                    return None
                finished = dataclasses.replace(
                    message,
                    content=answer.content,
                    outcome=answer.outcome,
                    refusal_reason=answer.refusal_reason,
                    citations=answer.citations,
                    sources=answer.sources,
                )
                messages[index] = finished
                self.answers[message_id] = answer
                self._touch(conversation_id)
                return finished
        return None

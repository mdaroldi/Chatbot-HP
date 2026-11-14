"""The page's state, and every change to it as a pure function.

Streamlit reruns the whole script on every click, so the state must say on its own
what to fetch and what to show. Each transition returns a new UiState; none does I/O,
so the page's behaviour across reruns is tested without Streamlit.

A question goes through the outbox: only a submission writes it, and the run that
sends it pops it first, so no rerun can send it twice.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

from hpdocs_ui.api import Conversation, ConversationPage, StreamEvent
from hpdocs_ui.chat_view import ErrorView, MessageView, answer_from_stream, error_view
from hpdocs_ui.errors import ERROR_TEXT, ApiError
from hpdocs_ui.url_state import UrlState

Notice = Literal["not_found", "deleted"]
NOTICE_TEXT: Mapping[Notice, str] = {
    "not_found": ERROR_TEXT["conversation_not_found"],
    "deleted": "The conversation was deleted.",
}


@dataclass(frozen=True)
class UiState:
    client_id: str
    conversation_id: str | None
    # The loaded thread, and the conversation it belongs to.
    thread: tuple[MessageView, ...] | None = None
    thread_for: str | None = None
    conversations: tuple[Conversation, ...] = ()
    next_cursor: str | None = None
    list_stale: bool = True
    outbox: str | None = None
    # One-shot: cleared once drawn.
    notice: Notice | None = None
    # The last failed send, shown until the next send or selection.
    send_error: ErrorView | None = None
    confirm_delete: str | None = None

    @property
    def needs_thread(self) -> bool:
        return self.conversation_id is not None and self.thread_for != self.conversation_id


def initial(url: UrlState) -> UiState:
    return UiState(client_id=url.client_id, conversation_id=url.conversation_id)


def url_state(state: UiState) -> UrlState:
    return UrlState(state.client_id, state.conversation_id)


def queue(state: UiState, text: str) -> UiState:
    """Stripped as the backend stores it, so the reloaded thread shows the same text."""
    question = text.strip()
    if not question or state.outbox is not None:
        return state
    return replace(state, outbox=question, send_error=None)


def take_outbox(state: UiState) -> tuple[UiState, str | None]:
    return replace(state, outbox=None), state.outbox


def conversation_created(state: UiState, conversation: Conversation) -> UiState:
    return replace(
        state,
        conversation_id=conversation.id,
        thread=(),
        thread_for=conversation.id,
        list_stale=True,
    )


def send_finished(
    state: UiState, question: str, events: Sequence[StreamEvent], error: ApiError | None
) -> UiState:
    """The answer joins the thread if the backend stored the turn; if it stored
    nothing, the question is kept only in the error view, to be resent.
    """
    answer = answer_from_stream(events, error)
    # Title and order change after any send, and a new conversation must appear.
    state = replace(state, list_stale=True)
    if answer is None:
        # answer_from_stream returns None only for an error before the stream started.
        return replace(state, send_error=error_view(cast("ApiError", error), question=question))
    asked = MessageView(role="user", text=question, state=None, sources=())
    return replace(
        state,
        thread=(*(state.thread or ()), asked, answer),
        thread_for=state.conversation_id,
        send_error=None if error is None else error_view(error),
    )


def select(state: UiState, conversation_id: str) -> UiState:
    if conversation_id == state.conversation_id:
        return state
    return replace(
        state,
        conversation_id=conversation_id,
        thread=None,
        thread_for=None,
        send_error=None,
        confirm_delete=None,
    )


def new_conversation(state: UiState) -> UiState:
    """Creates nothing: the conversation is created with its first question."""
    return replace(
        state,
        conversation_id=None,
        thread=None,
        thread_for=None,
        send_error=None,
        confirm_delete=None,
    )


def thread_loaded(state: UiState, conversation_id: str, thread: tuple[MessageView, ...]) -> UiState:
    return replace(state, thread=thread, thread_for=conversation_id)


def thread_missing(state: UiState) -> UiState:
    return replace(new_conversation(state), notice="not_found", list_stale=True)


def refresh_thread(state: UiState) -> UiState:
    return replace(state, thread_for=None)


def notice_shown(state: UiState) -> UiState:
    return replace(state, notice=None)


def list_loaded(state: UiState, page: ConversationPage, *, more: bool) -> UiState:
    items = (*state.conversations, *page.items) if more else page.items
    return replace(state, conversations=items, next_cursor=page.next_cursor, list_stale=False)


def ask_delete(state: UiState, conversation_id: str) -> UiState:
    return replace(state, confirm_delete=conversation_id)


def cancel_delete(state: UiState) -> UiState:
    return replace(state, confirm_delete=None)


def deleted(state: UiState, conversation_id: str) -> UiState:
    if conversation_id == state.conversation_id:
        state = new_conversation(state)
    remaining = tuple(c for c in state.conversations if c.id != conversation_id)
    return replace(
        state, conversations=remaining, confirm_delete=None, notice="deleted", list_stale=True
    )

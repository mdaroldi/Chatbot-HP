"""Page state transitions: pure functions from one UiState to the next."""

from datetime import UTC, datetime

from hpdocs_ui.api import Conversation, ConversationPage, Done, Flags, StreamEvent, Token
from hpdocs_ui.chat_view import REFUSAL_TEXT, ErrorView, MessageView
from hpdocs_ui.errors import ApiError
from hpdocs_ui.state import (
    NOTICE_TEXT,
    UiState,
    ask_delete,
    cancel_delete,
    conversation_created,
    deleted,
    initial,
    list_loaded,
    new_conversation,
    notice_shown,
    queue,
    refresh_thread,
    select,
    send_finished,
    take_outbox,
    thread_loaded,
    thread_missing,
    url_state,
)
from hpdocs_ui.url_state import UrlState

NOW = datetime(2026, 11, 14, 9, 0, tzinfo=UTC)
CID = "client-0001"
A = "6f1c2d3e-4b5a-4c6d-8e7f-90a1b2c3d4e5"
B = "0b7e7a0c-2f7d-4f43-9d3f-6f2b8f1d2c3a"
ANSWER = MessageView(role="assistant", text="Old answer.", state="uncited", sources=())


def conversation(conversation_id: str, title: str | None = "Title") -> Conversation:
    return Conversation(id=conversation_id, title=title, created_at=NOW, updated_at=NOW)


def selected(conversation_id: str = A) -> UiState:
    """A state with a loaded, one-answer thread for the conversation."""
    return thread_loaded(initial(UrlState(CID, conversation_id)), conversation_id, (ANSWER,))


def done(content: str = "New answer [S1].") -> Done:
    flags = Flags(
        late_refusal_marker=False,
        uncited=False,
        invalid_citations=False,
        rewrite_fallback=False,
        truncated=False,
    )
    return Done(
        conversation_id=A,
        user_message_id="u",
        message_id="m",
        outcome="completed",
        refusal=False,
        refusal_reason=None,
        content=content,
        citations=("S1",),
        flags=flags,
    )


def user(text: str) -> MessageView:
    return MessageView(role="user", text=text, state=None, sources=())


# --- start and URL ---------------------------------------------------------------


def test_the_initial_state_follows_the_url_and_needs_the_list_and_thread() -> None:
    state = initial(UrlState(CID, A))

    assert (state.client_id, state.conversation_id, state.thread, state.thread_for) == (
        CID,
        A,
        None,
        None,
    )
    assert state.list_stale
    assert (state.conversations, state.next_cursor, state.outbox) == ((), None, None)
    assert (state.notice, state.send_error, state.confirm_delete) == (None, None, None)
    assert state.needs_thread


def test_the_url_follows_the_state() -> None:
    assert url_state(initial(UrlState(CID, A))) == UrlState(CID, A)
    assert url_state(new_conversation(selected())) == UrlState(CID, None)


def test_no_thread_is_needed_without_a_selection_or_once_it_is_loaded() -> None:
    assert not initial(UrlState(CID, None)).needs_thread
    assert not selected().needs_thread


# --- single send -----------------------------------------------------------------


def test_queue_stores_the_stripped_question_and_clears_the_last_error() -> None:
    sending, question = take_outbox(queue(selected(), "Q?"))
    assert question is not None
    failed = send_finished(sending, question, [], ApiError("llm_error", status=502))
    assert failed.send_error is not None

    state = queue(failed, "  How do I scan?  ")

    assert (state.outbox, state.send_error) == ("How do I scan?", None)


def test_queue_ignores_a_blank_question() -> None:
    state = selected()

    assert queue(state, "   ") == state


def test_a_second_question_while_one_is_queued_is_ignored() -> None:
    state = queue(selected(), "first")

    assert queue(state, "second").outbox == "first"


def test_take_outbox_pops_the_question_once() -> None:
    state, question = take_outbox(queue(selected(), "Q?"))
    again, second = take_outbox(state)

    assert (question, state.outbox, second) == ("Q?", None, None)
    assert again == state


def test_a_created_conversation_is_selected_with_an_empty_thread() -> None:
    state = conversation_created(initial(UrlState(CID, None)), conversation(A, None))

    assert (state.conversation_id, state.thread, state.thread_for) == (A, (), A)
    assert state.list_stale
    assert not state.needs_thread


def test_a_finished_answer_is_appended_after_its_question() -> None:
    events: list[StreamEvent] = [Token(text="New answer"), done()]
    listed = list_loaded(selected(), ConversationPage(items=(), next_cursor=None), more=False)

    state = send_finished(listed, "Q?", events, None)

    assert state.thread is not None
    assert [(m.role, m.text) for m in state.thread] == [
        ("assistant", "Old answer."),
        ("user", "Q?"),
        ("assistant", "New answer [S1]."),
    ]
    assert (state.send_error, state.list_stale) == (None, True)


def test_a_failure_before_the_stream_keeps_the_question_only_in_the_error_view() -> None:
    error = ApiError("llm_unavailable", status=503, request_id="req-12345678")

    state = send_finished(selected(), "Q?", [], error)

    assert state.thread == (ANSWER,)
    assert state.send_error == ErrorView(
        text="The language model is unavailable right now. Try again in a moment.",
        request_id="req-12345678",
        question="Q?",
    )
    assert state.list_stale


def test_a_failure_mid_stream_keeps_the_partial_answer_and_the_error_without_the_question() -> None:
    state = send_finished(selected(), "Q?", [Token(text="Part")], ApiError("llm_error", status=200))

    assert state.thread is not None
    assert [(m.role, m.text, m.state) for m in state.thread[1:]] == [
        ("user", "Q?", None),
        ("assistant", "Part", "failed"),
    ]
    assert state.send_error == ErrorView(
        text="The language model couldn't answer this question.", request_id=None, question=None
    )


def test_a_withdrawn_answer_shows_the_refusal_and_the_withdrawn_error() -> None:
    state = send_finished(
        selected(), "Q?", [Token(text="Part")], ApiError("answer_withdrawn", status=200)
    )

    assert state.thread is not None
    assert state.thread[-1].text == REFUSAL_TEXT["provider_refused"]
    assert state.send_error is not None
    assert state.send_error.text == "The model withdrew this answer."


def test_an_answer_to_a_conversation_with_no_loaded_thread_starts_one() -> None:
    state = send_finished(initial(UrlState(CID, A)), "Q?", [done()], None)

    assert state.thread is not None
    assert [m.role for m in state.thread] == ["user", "assistant"]
    assert state.thread_for == A


# --- selection and threads -------------------------------------------------------


def test_selecting_another_conversation_drops_the_thread_and_transient_views() -> None:
    state = ask_delete(send_finished(selected(), "Q?", [], ApiError("x", status=500)), A)

    moved = select(state, B)

    assert (moved.conversation_id, moved.thread, moved.thread_for) == (B, None, None)
    assert (moved.send_error, moved.confirm_delete) == (None, None)
    assert moved.needs_thread


def test_selecting_the_current_conversation_keeps_its_thread() -> None:
    state = selected()

    assert select(state, A).thread == (ANSWER,)


def test_new_conversation_clears_the_selection_and_creates_nothing() -> None:
    page = ConversationPage(items=(conversation(A),), next_cursor="c1")
    failed = send_finished(selected(), "Q?", [], ApiError("x", status=500))

    state = new_conversation(list_loaded(failed, page, more=False))

    assert (state.conversation_id, state.thread, state.thread_for, state.send_error) == (
        None,
        None,
        None,
        None,
    )
    # The list is untouched: nothing was created, so nothing is refetched.
    assert ([c.id for c in state.conversations], state.next_cursor, state.list_stale) == (
        [A],
        "c1",
        False,
    )


def test_a_missing_thread_clears_the_selection_with_the_not_found_notice() -> None:
    state = thread_missing(initial(UrlState(CID, A)))

    assert (state.conversation_id, state.thread, state.thread_for) == (None, None, None)
    assert state.notice == "not_found"
    assert NOTICE_TEXT["not_found"] == "This conversation doesn't exist or was deleted."
    assert state.list_stale


def test_refresh_thread_reloads_the_selected_thread() -> None:
    state = refresh_thread(selected())

    assert (state.conversation_id, state.thread_for) == (A, None)
    assert state.needs_thread


def test_a_notice_is_shown_once() -> None:
    assert notice_shown(thread_missing(selected())).notice is None


# --- the conversation list -------------------------------------------------------


def test_a_first_page_replaces_the_list_and_load_more_appends() -> None:
    first = list_loaded(
        initial(UrlState(CID, None)),
        ConversationPage(items=(conversation(A),), next_cursor="c1"),
        more=False,
    )
    more = list_loaded(
        first, ConversationPage(items=(conversation(B),), next_cursor=None), more=True
    )
    again = list_loaded(
        more, ConversationPage(items=(conversation(B),), next_cursor="c2"), more=False
    )

    assert ([c.id for c in first.conversations], first.next_cursor, first.list_stale) == (
        [A],
        "c1",
        False,
    )
    assert ([c.id for c in more.conversations], more.next_cursor) == ([A, B], None)
    assert ([c.id for c in again.conversations], again.next_cursor) == ([B], "c2")


# --- delete ----------------------------------------------------------------------


def test_delete_asks_for_confirmation_and_can_be_cancelled() -> None:
    state = ask_delete(selected(), B)

    assert state.confirm_delete == B
    assert cancel_delete(state).confirm_delete is None


def test_deleting_the_selected_conversation_clears_it() -> None:
    listed = list_loaded(
        selected(),
        ConversationPage(items=(conversation(A), conversation(B)), next_cursor=None),
        more=False,
    )

    state = deleted(ask_delete(listed, A), A)

    assert (state.conversation_id, state.thread, state.confirm_delete) == (None, None, None)
    assert [c.id for c in state.conversations] == [B]
    assert (state.notice, state.list_stale) == ("deleted", True)
    assert NOTICE_TEXT["deleted"] == "The conversation was deleted."


def test_deleting_another_conversation_keeps_the_selection() -> None:
    listed = list_loaded(
        selected(),
        ConversationPage(items=(conversation(A), conversation(B)), next_cursor=None),
        more=False,
    )

    state = deleted(ask_delete(listed, B), B)

    assert (state.conversation_id, state.thread) == (A, (ANSWER,))
    assert [c.id for c in state.conversations] == [A]
    assert (state.notice, state.confirm_delete, state.list_stale) == ("deleted", None, True)

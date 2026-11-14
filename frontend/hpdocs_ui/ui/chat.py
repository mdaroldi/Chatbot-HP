"""The main area: the question box, the answer being streamed, the last failed send.

A question is sent only in the run that a submission started (the callback queues
it in the outbox), so that run knows before drawing anything that it will stream and
draws every widget disabled: no click can interrupt an answer. The run pops the
outbox before any request, streams, stores the outcome and reruns, so the final page
is drawn from state like any other.
"""

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from hpdocs_ui import state as ui
from hpdocs_ui.api import ChatApi, StreamEvent, Token
from hpdocs_ui.chat_view import (
    NOTICE_TEXT,
    PRIMING_TEXT,
    WARNING_BADGE,
    ErrorView,
    MessageView,
    error_view,
    source_heading,
    thread_view,
)
from hpdocs_ui.errors import ApiError
from hpdocs_ui.sanitize import escape_markdown, sanitize_markdown
from hpdocs_ui.settings import FrontendSettings
from hpdocs_ui.ui import session

QUESTION_KEY = "question"
_NOT_FOUND = "conversation_not_found"


def render_answer_text(slot: DeltaGenerator, text: str) -> None:
    """The only Markdown on the page: answer text, sanitized as a whole every time, so
    a link or image split across tokens never renders, even for one update.
    """
    slot.markdown(sanitize_markdown(text))


def load_thread(api: ChatApi) -> ErrorView | None:
    """The selected conversation's thread, read once; a failed read is shown and the
    thread stays unloaded, so the next run reads it again.
    """
    state = session.get()
    conversation_id = state.conversation_id
    if conversation_id is None or not state.needs_thread:
        return None
    try:
        detail = api.get_conversation(conversation_id)
    except ApiError as error:
        if error.code == _NOT_FOUND:
            gone = ui.thread_missing(state)
            session.put(gone)
            session.sync_url(gone)  # this run's URL was written before the read
            return None
        return error_view(error)
    session.put(ui.thread_loaded(state, conversation_id, thread_view(detail)))
    return None


def render_thread(load_error: ErrorView | None) -> None:
    if load_error is not None:
        _error(load_error)
    for message in session.get().thread or ():
        if message.role == "user":
            with st.chat_message("user"):
                st.text(message.text)
        else:
            with st.chat_message("assistant"):
                _answer(message)


def _answer(message: MessageView) -> None:
    render_answer_text(st.empty(), message.text)
    state = message.state
    if state == "pending":
        st.info(NOTICE_TEXT[state])
    elif state is not None and state in NOTICE_TEXT:
        st.warning(NOTICE_TEXT[state])
    if not message.sources:
        return
    with st.expander("Sources"):
        for source in message.sources:
            st.caption(escape_markdown(source_heading(source)))
            if source.has_warning:
                st.badge(WARNING_BADGE, icon=":material/warning:", color="orange")
            st.text(source.snippet)


def render_send_error() -> None:
    error = session.get().send_error
    if error is not None:
        _error(error)


def _error(error: ErrorView) -> None:
    st.error(error.text)
    if error.request_id is not None:
        st.caption(escape_markdown(f"Request ID: {error.request_id}"))
    if error.question is not None:
        # A code block shows the text as typed, with a copy button.
        st.code(error.question, language=None, wrap_lines=True)


def render_input(settings: FrontendSettings, *, sending: bool) -> None:
    st.chat_input(
        "Ask about the HP ENVY 6000 or the OMEN 17",
        key=QUESTION_KEY,
        max_chars=settings.ui_message_max_chars,
        disabled=sending,
        on_submit=_submit,
    )


def _submit() -> None:
    # On submit, the input's value is the text just sent.
    text: str = st.session_state[QUESTION_KEY]
    session.apply(ui.queue, text)


def send(api: ChatApi) -> None:
    state, question = ui.take_outbox(session.get())
    if question is None:
        return
    # Popped before any request: a rerun from here on finds nothing to send.
    session.put(state)
    with st.chat_message("user"):
        st.text(question)
    conversation_id = state.conversation_id
    if conversation_id is None:
        try:
            created = api.create_conversation()
        except ApiError as error:
            session.put(ui.not_sent(state, question, error))
            st.rerun()
        state = ui.conversation_created(state, created)
        session.put(state)
        # Before streaming: a reload mid-answer must find the conversation.
        session.sync_url(state)
        conversation_id = created.id
    with st.chat_message("assistant"):
        slot = st.empty()
        slot.caption(PRIMING_TEXT)
        with st.spinner("", show_time=True):
            events, failure = _read(api, conversation_id, question, slot)
    session.put(ui.send_finished(session.get(), question, events, failure))
    st.rerun()


def _read(
    api: ChatApi, conversation_id: str, question: str, slot: DeltaGenerator
) -> tuple[list[StreamEvent], ApiError | None]:
    """The events read and how reading ended; the text so far is redrawn per token.

    Leaving the stream's context closes the connection, also when Streamlit stops the
    script because the tab went away; the backend then stores the answer cancelled.
    """
    events: list[StreamEvent] = []
    text = ""
    try:
        with api.stream_answer(conversation_id, question) as answer:
            for event in answer:
                events.append(event)
                if isinstance(event, Token):
                    text += event.text
                    render_answer_text(slot, text)
    except ApiError as error:
        return events, error
    return events, None

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
from hpdocs_ui.chat_view import PRIMING_TEXT
from hpdocs_ui.errors import ApiError
from hpdocs_ui.sanitize import escape_markdown, sanitize_markdown
from hpdocs_ui.settings import FrontendSettings
from hpdocs_ui.ui import session

QUESTION_KEY = "question"


def render_answer_text(slot: DeltaGenerator, text: str) -> None:
    """The only Markdown on the page: answer text, sanitized as a whole every time, so
    a link or image split across tokens never renders, even for one update.
    """
    slot.markdown(sanitize_markdown(text))


def render_send_error() -> None:
    error = session.get().send_error
    if error is None:
        return
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

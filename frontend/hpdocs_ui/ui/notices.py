"""Standing notices: where questions go, the private link, the backend being down, and
the empty state with example questions.

Every text here is fixed wording; the provider named by /ready only picks which.
"""

import streamlit as st

from hpdocs_ui import state as ui
from hpdocs_ui.chat_view import EXAMPLE_QUESTIONS, EXAMPLES_INTRO, LINK_TEXT, PRIVACY_TEXT
from hpdocs_ui.errors import BACKEND_UNREACHABLE, ERROR_TEXT
from hpdocs_ui.ui import session
from hpdocs_ui.view import Provider, StatusView


def privacy_and_link(provider: Provider) -> None:
    st.caption(PRIVACY_TEXT[provider])
    st.caption(LINK_TEXT)


def backend_down(view: StatusView, *, disabled: bool) -> None:
    """Retry checks the status again; that run also reads the list and the thread
    again, since a failed read leaves them to be read.
    """
    if view.level != "error":
        return
    st.error(ERROR_TEXT[BACKEND_UNREACHABLE])
    st.button(
        "Retry",
        key="retry-backend",
        disabled=disabled,
        on_click=session.apply,
        args=(ui.refresh_status,),
    )


def empty_state(provider: Provider, *, disabled: bool) -> None:
    """An example is queued like a typed question, so it is sent the same way."""
    st.caption(EXAMPLES_INTRO)
    for index in range(len(EXAMPLE_QUESTIONS)):
        st.button(
            EXAMPLE_QUESTIONS[index],
            key=f"example-{index}",
            disabled=disabled,
            on_click=session.apply,
            args=(ui.queue, EXAMPLE_QUESTIONS[index]),
        )
    st.caption(PRIVACY_TEXT[provider])

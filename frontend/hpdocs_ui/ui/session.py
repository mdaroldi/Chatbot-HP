"""The UiState in Streamlit's session, and the URL kept in step with it."""

from collections.abc import Callable
from typing import Concatenate

import streamlit as st

from hpdocs_ui import url_state
from hpdocs_ui.state import UiState, initial
from hpdocs_ui.state import url_state as state_url
from hpdocs_ui.ui import guard

_KEY = "ui"


def start() -> UiState:
    """The session's state, created from the URL on the first run; then the URL is
    rewritten from the state, so it always holds a valid client ID and the open
    conversation (and never a value the page did not accept).
    """
    if _KEY not in st.session_state:
        st.session_state[_KEY] = initial(url_state.resolve(st.query_params.to_dict()))
    state = get()
    sync_url(state)
    return state


def get() -> UiState:
    state: UiState = st.session_state[_KEY]
    return state


def put(state: UiState) -> None:
    st.session_state[_KEY] = state


def sync_url(state: UiState) -> None:
    params = url_state.query_params(state_url(state))
    if st.query_params.to_dict() != params:
        st.query_params.from_dict(params)


@guard.callback
def apply[**P](
    transition: Callable[Concatenate[UiState, P], UiState], *args: P.args, **kwargs: P.kwargs
) -> None:
    """A widget callback: one pure transition on the session's state."""
    put(transition(get(), *args, **kwargs))

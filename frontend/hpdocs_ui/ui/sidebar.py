"""The sidebar: one-shot notices, the conversation list, delete, backend status.

Every click is a callback that applies one transition from hpdocs_ui.state; the
calls with I/O (load more, delete) catch every ApiError and turn it into state, so
nothing escapes to Streamlit's error page.
"""

import streamlit as st

from hpdocs_ui import state as ui
from hpdocs_ui import status
from hpdocs_ui.api import ChatApi, Conversation
from hpdocs_ui.chat_view import conversation_label
from hpdocs_ui.errors import ApiError
from hpdocs_ui.settings import FrontendSettings
from hpdocs_ui.state import NOTICE_TEXT, UiState
from hpdocs_ui.ui import guard, notices, session
from hpdocs_ui.view import StatusView, describe

_NOT_FOUND = "conversation_not_found"


def load_list(api: ChatApi) -> None:
    """The first page, whenever something made the list stale."""
    state = session.get()
    if not state.list_stale:
        return
    try:
        page = api.list_conversations()
    except ApiError as error:
        session.put(ui.sidebar_failed(state, error))
        return
    session.put(ui.list_loaded(state, page, more=False))


def check_status(settings: FrontendSettings) -> StatusView:
    """Once per session, and again after Refresh."""
    state = session.get()
    if state.backend is not None:
        return state.backend
    base_url = str(settings.backend_url)
    view = describe(
        status.fetch_status(base_url, "/health"), status.fetch_status(base_url, "/ready")
    )
    session.put(ui.status_checked(state, view))
    return view


def render(api: ChatApi, backend: StatusView, *, disabled: bool) -> None:
    """Every button is disabled while an answer streams: a click would stop it."""
    state = session.get()
    with st.sidebar:
        notices.privacy_and_link(backend.provider)
        _notice(state)
        st.button(
            "New conversation",
            key="new-conversation",
            icon=":material/add:",
            disabled=disabled,
            on_click=session.apply,
            args=(ui.new_conversation,),
        )
        for conversation in state.conversations:
            _conversation_row(api, state, conversation, disabled=disabled)
        if state.next_cursor is not None:
            st.button(
                "Load more",
                key="load-more",
                disabled=disabled,
                on_click=_load_more,
                args=(api,),
            )
        if state.sidebar_error is not None:
            error = state.sidebar_error
            st.error(error.text)
        _status(backend, disabled=disabled)


def _notice(state: UiState) -> None:
    if state.notice is None:
        return
    if state.notice == "deleted":
        st.info(NOTICE_TEXT[state.notice])
    else:
        st.warning(NOTICE_TEXT[state.notice])
    session.put(ui.notice_shown(state))


def _conversation_row(
    api: ChatApi, state: UiState, conversation: Conversation, *, disabled: bool
) -> None:
    conversation_id = conversation.id
    current = conversation_id == state.conversation_id
    title, delete = st.columns([5, 1], vertical_alignment="center")
    title.button(
        conversation_label(conversation),
        key=f"open-{conversation_id}",
        type="primary" if current else "tertiary",
        width="stretch",
        disabled=disabled,
        on_click=session.apply,
        args=(ui.select, conversation_id),
    )
    delete.button(
        "",
        key=f"delete-{conversation_id}",
        icon=":material/delete:",
        help="Delete this conversation",
        type="tertiary",
        disabled=disabled,
        on_click=session.apply,
        args=(ui.ask_delete, conversation_id),
    )
    if state.confirm_delete != conversation_id:
        return
    st.warning("Delete this conversation? This can't be undone.")
    confirm, cancel = st.columns(2)
    confirm.button(
        "Delete",
        key=f"confirm-delete-{conversation_id}",
        type="primary",
        disabled=disabled,
        on_click=_delete,
        args=(api, conversation_id),
    )
    cancel.button(
        "Cancel",
        key=f"cancel-delete-{conversation_id}",
        disabled=disabled,
        on_click=session.apply,
        args=(ui.cancel_delete,),
    )


@guard.callback
def _load_more(api: ChatApi) -> None:
    state = session.get()
    try:
        page = api.list_conversations(state.next_cursor)
    except ApiError as error:
        session.put(ui.sidebar_failed(state, error))
        return
    session.put(ui.list_loaded(state, page, more=True))


@guard.callback
def _delete(api: ChatApi, conversation_id: str) -> None:
    state = session.get()
    try:
        api.delete_conversation(conversation_id)
    except ApiError as error:
        if error.code == _NOT_FOUND:
            session.put(ui.already_deleted(state, conversation_id))
        else:
            session.put(ui.sidebar_failed(state, error))
        return
    session.put(ui.deleted(state, conversation_id))


def _status(view: StatusView, *, disabled: bool) -> None:
    with st.expander("Backend status", expanded=False):
        if view.level == "success":
            st.success(view.headline)
        elif view.level == "warning":
            st.warning(view.headline)
        else:
            st.error(view.headline)
        for line in view.lines:
            st.text(line)  # plain text: backend values are never read as Markdown
        st.button(
            "Refresh status",
            key="refresh-status",
            disabled=disabled,
            on_click=session.apply,
            args=(ui.refresh_status,),
        )

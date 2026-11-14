"""Streamlit entry point: the run order of one page run. Logic lives in hpdocs_ui."""

import streamlit as st

from hpdocs_ui import wiring
from hpdocs_ui.settings import FrontendSettings
from hpdocs_ui.ui import chat, session, sidebar

st.set_page_config(page_title="HP Docs Chatbot", initial_sidebar_state="expanded")
wiring.quiet_http_logs()

settings = FrontendSettings()
state = session.start()
# A submission queued a question: this run streams it, with every widget disabled.
sending = state.outbox is not None
api = wiring.build_api(settings, state.client_id)

sidebar.load_list(api)
backend = sidebar.check_status(settings)
sidebar.render(api, backend, disabled=sending)

st.title("HP Docs Chatbot")
chat.render_send_error()
chat.render_input(settings, sending=sending)
chat.send(api)

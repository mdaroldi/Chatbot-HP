"""Streamlit entry point. Rendering only; logic lives in hpdocs_ui."""

import streamlit as st

from hpdocs_ui import status
from hpdocs_ui.settings import FrontendSettings
from hpdocs_ui.view import describe

RENDER_LEVEL = {"success": st.success, "warning": st.warning, "error": st.error}

st.set_page_config(page_title="HP Docs Chatbot")
st.title("HP Docs Chatbot")
st.caption("Phase 0 placeholder: backend status.")

# Any click reruns the script, which fetches status again (nothing is cached).
st.button("Refresh status")

backend_url = str(FrontendSettings().backend_url)
view = describe(
    status.fetch_status(backend_url, "/health"),
    status.fetch_status(backend_url, "/ready"),
)
RENDER_LEVEL[view.level](view.headline)
for line in view.lines:
    st.text(line)  # plain text: backend values are never interpreted as markdown

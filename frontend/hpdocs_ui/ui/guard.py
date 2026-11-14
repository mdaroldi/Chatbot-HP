"""The last line of defence: a bug becomes the generic wording and a log line naming
only its type.

Streamlit logs an uncaught script exception with its message and source lines, and
either could hold a question or an answer; catching every Exception here is what keeps
user content out of the logs (ADR 0011). Streamlit's control flow (stop, rerun) and
interrupts derive from BaseException and pass through.

Widget callbacks run before the script, outside the page run, so each is guarded on
its own. A failure is shown in the next page run, once: a page that failed is rerun a
single time, so a run that drew every widget disabled (an answer streaming) is
replaced by an enabled one.
"""

import functools
import logging
from collections.abc import Callable

import streamlit as st

from hpdocs_ui.errors import GENERIC_ERROR_TEXT

_FAILED = "page-failed"
_logger = logging.getLogger(__name__)


def run_page(page: Callable[[], None]) -> None:
    failed_before = bool(st.session_state.pop(_FAILED, False))
    if failed_before:
        st.error(GENERIC_ERROR_TEXT)
    try:
        page()
    except Exception as error:
        _log("page run failed", error)
        if failed_before:
            return  # already shown in this run; rerunning again could loop
        st.session_state[_FAILED] = True
        st.rerun()


def callback[**P](action: Callable[P, None]) -> Callable[P, None]:
    @functools.wraps(action)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            action(*args, **kwargs)
        except Exception as error:
            _log("page action failed", error)
            st.session_state[_FAILED] = True

    return guarded


def _log(what: str, error: Exception) -> None:
    # The type only: the message, the traceback and its source lines may hold user text.
    _logger.error("%s: %s", what, type(error).__name__)

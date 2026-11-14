"""The page guard: bugs become the generic wording and a log line with the type only;
Streamlit's own control flow passes through untouched.
"""

import logging

import pytest
import streamlit as st
from streamlit.runtime.scriptrunner_utils.exceptions import RerunException, StopException
from streamlit.runtime.scriptrunner_utils.script_requests import RerunData

from hpdocs_ui.ui import guard

CANARY = "CANARY-guard-5f1e"


def raising(error: BaseException) -> None:
    raise error


@pytest.mark.parametrize(
    "control",
    [StopException(), RerunException(RerunData()), KeyboardInterrupt(), SystemExit(0)],
    ids=["stop", "rerun", "keyboard-interrupt", "system-exit"],
)
def test_streamlit_control_flow_and_interrupts_pass_through(control: BaseException) -> None:
    with pytest.raises(type(control)):
        guard.run_page(lambda: raising(control))
    with pytest.raises(type(control)):
        guard.callback(raising)(control)


def test_a_failed_callback_logs_only_the_type_and_shows_nothing_itself(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown: list[str] = []
    monkeypatch.setattr(st, "error", shown.append)
    caplog.set_level(logging.DEBUG)

    guard.callback(raising)(ValueError(CANARY))

    records = [r for r in caplog.records if r.name == guard.__name__]
    assert [(r.levelno, r.getMessage(), r.exc_info) for r in records] == [
        (logging.ERROR, "page action failed: ValueError", None)
    ]
    # The page run that follows the callback shows the generic wording.
    assert shown == []

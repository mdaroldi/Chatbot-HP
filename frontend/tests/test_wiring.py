"""The page's one seam to the backend, and the HTTP logs it keeps quiet."""

import logging

import pytest

from hpdocs_ui.http_api import HttpChatApi
from hpdocs_ui.settings import FrontendSettings
from hpdocs_ui.wiring import build_api, quiet_http_logs


def test_build_api_returns_the_http_client_for_the_client_id() -> None:
    api = build_api(FrontendSettings(), "client-0001")

    assert isinstance(api, HttpChatApi)


def test_build_api_refuses_an_invalid_client_id() -> None:
    with pytest.raises(ValueError, match="client ID"):
        build_api(FrontendSettings(), "bad id")


@pytest.mark.parametrize("name", ["httpx", "httpcore"])
def test_http_libraries_log_warnings_only(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Their INFO and DEBUG lines carry URLs, which hold conversation IDs.
    logger = logging.getLogger(name)
    monkeypatch.setattr(logger, "level", logging.NOTSET)

    quiet_http_logs()

    assert logger.level == logging.WARNING

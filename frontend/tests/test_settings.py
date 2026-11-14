"""Behavior of the frontend settings."""

import pytest
from pydantic import ValidationError

from hpdocs_ui.settings import FrontendSettings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKEND_URL", raising=False)


def test_backend_url_defaults_to_the_compose_service() -> None:
    assert str(FrontendSettings().backend_url) == "http://backend:8000/"


@pytest.mark.parametrize("url", ["http://localhost:8000", "https://api.example.com/base"])
def test_backend_url_accepts_http_and_https(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("BACKEND_URL", url)

    parsed = FrontendSettings().backend_url

    assert parsed.scheme in {"http", "https"}
    assert str(parsed).startswith(url)


@pytest.mark.parametrize("url", ["ftp://backend:21", "backend:8000", "not a url", ""])
def test_backend_url_must_be_http_or_https(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("BACKEND_URL", url)

    with pytest.raises(ValidationError, match="backend_url"):
        FrontendSettings()


STREAM_FIELDS = {
    "UI_STREAM_READ_TIMEOUT_SECONDS": 60.0,
    "UI_STREAM_TOTAL_TIMEOUT_SECONDS": 185.0,
}


@pytest.fixture
def no_ui_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (*STREAM_FIELDS, "UI_MESSAGE_MAX_CHARS"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.usefixtures("no_ui_env")
def test_stream_timeouts_default_to_the_values_derived_from_the_backend() -> None:
    settings = FrontendSettings()

    assert settings.ui_stream_read_timeout_seconds == 60.0
    assert settings.ui_stream_total_timeout_seconds == 185.0


@pytest.mark.usefixtures("no_ui_env")
def test_message_limit_defaults_to_the_backend_limit() -> None:
    assert FrontendSettings().ui_message_max_chars == 1000


@pytest.mark.usefixtures("no_ui_env")
def test_ui_settings_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UI_STREAM_READ_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("UI_STREAM_TOTAL_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("UI_MESSAGE_MAX_CHARS", "500")

    settings = FrontendSettings()

    assert settings.ui_stream_read_timeout_seconds == 90.0
    assert settings.ui_stream_total_timeout_seconds == 300.0
    assert settings.ui_message_max_chars == 500


@pytest.mark.usefixtures("no_ui_env")
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("UI_STREAM_READ_TIMEOUT_SECONDS", "0"),
        ("UI_STREAM_READ_TIMEOUT_SECONDS", "-1"),
        ("UI_STREAM_READ_TIMEOUT_SECONDS", "inf"),
        ("UI_STREAM_READ_TIMEOUT_SECONDS", "nan"),
        ("UI_STREAM_READ_TIMEOUT_SECONDS", "601"),
        ("UI_STREAM_TOTAL_TIMEOUT_SECONDS", "0"),
        ("UI_STREAM_TOTAL_TIMEOUT_SECONDS", "3601"),
        ("UI_MESSAGE_MAX_CHARS", "0"),
        ("UI_MESSAGE_MAX_CHARS", "4001"),
        ("UI_MESSAGE_MAX_CHARS", "ten"),
    ],
)
def test_ui_settings_out_of_range_fail_at_startup(
    monkeypatch: pytest.MonkeyPatch, key: str, value: str
) -> None:
    monkeypatch.setenv(key, value)

    with pytest.raises(ValidationError, match=key.lower()):
        FrontendSettings()


@pytest.mark.usefixtures("no_ui_env")
def test_total_stream_timeout_must_not_be_shorter_than_the_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UI_STREAM_READ_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("UI_STREAM_TOTAL_TIMEOUT_SECONDS", "100")

    with pytest.raises(ValidationError, match="ui_stream_total_timeout_seconds"):
        FrontendSettings()

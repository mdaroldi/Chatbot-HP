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

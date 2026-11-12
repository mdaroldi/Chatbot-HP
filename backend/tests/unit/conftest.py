"""Shared isolation for unit tests: environment, cached settings, logging state."""

import io
import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from pydantic import SecretStr

from app.core.logging import JsonFormatter
from app.core.settings import Settings, get_settings

TEST_PASSWORD = "test-password"  # noqa: S105 - placeholder for unit tests only

LogReader = Callable[[], list[dict[str, Any]]]
SettingsFactory = Callable[..., Settings]

_LOGGERS_TOUCHED_BY_CONFIGURE = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "httpx",
    "httpcore",
    "sqlalchemy",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate every test from the developer's shell; covers every settings field."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", TEST_PASSWORD)


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    """Keep the cached settings instance from leaking between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[logging.Logger]:
    """create_app() reconfigures global logging; undo it after each test."""
    root = logging.getLogger()
    root_state = (root.handlers[:], root.level)
    others = {
        name: (lg.handlers[:], lg.level, lg.propagate)
        for name in _LOGGERS_TOUCHED_BY_CONFIGURE
        for lg in [logging.getLogger(name)]
    }
    yield root
    root.handlers[:], level = root_state
    root.setLevel(level)
    for name, (handlers, lvl, propagate) in others.items():
        lg = logging.getLogger(name)
        lg.handlers[:] = handlers
        lg.setLevel(lvl)
        lg.propagate = propagate


@pytest.fixture
def make_settings() -> SettingsFactory:
    def factory(**overrides: Any) -> Settings:  # Any: arbitrary field overrides
        return Settings(postgres_password=SecretStr(TEST_PASSWORD), **overrides)

    return factory


@pytest.fixture
def json_logs() -> Iterator[LogReader]:
    """Capture real formatter output, formatted at emit time like in production.

    Request it after any fixture that calls create_app(), which replaces root handlers.
    """
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    def read() -> list[dict[str, Any]]:
        return [json.loads(line) for line in stream.getvalue().splitlines()]

    yield read
    root.removeHandler(handler)

"""Integration fixtures: real Postgres and Qdrant from docker-compose.test.yml."""

import logging
import socket
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from pydantic import HttpUrl, SecretStr

from app.core.settings import Settings, get_settings

# Must match docker-compose.test.yml. Throwaway values for a local test container.
PG_HOST, PG_PORT = "127.0.0.1", 55432
PG_DB, PG_USER = "hpdocs_it", "it_user"
PG_PASSWORD = "it-throwaway-password-not-a-secret"  # noqa: S105 - test container only
QDRANT_URL = "http://127.0.0.1:56333"

START_HINT = "Start them with `make test-int` (uses docker-compose.test.yml)."

SettingsFactory = Callable[..., Settings]


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session", autouse=True)
def require_services() -> None:
    """Fail loudly (never skip) when the integration services are not running."""
    missing = [
        f"{name} on {host}:{port}"
        for name, host, port in (("Postgres", PG_HOST, PG_PORT), ("Qdrant", "127.0.0.1", 56333))
        if not _port_open(host, port)
    ]
    if missing:
        pytest.fail(f"Integration services not reachable: {', '.join(missing)}. {START_HINT}")


@pytest.fixture
def closed_port() -> int:
    """A loopback port with nothing listening on it."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


@pytest.fixture
def make_settings(monkeypatch: pytest.MonkeyPatch) -> SettingsFactory:
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)

    def factory(**overrides: Any) -> Settings:  # Any: arbitrary field overrides
        values: dict[str, Any] = {
            "postgres_host": PG_HOST,
            "postgres_port": PG_PORT,
            "postgres_db": PG_DB,
            "postgres_user": PG_USER,
            "postgres_password": SecretStr(PG_PASSWORD),
            "qdrant_url": HttpUrl(QDRANT_URL),
            "readiness_timeout_seconds": 5.0,
        }
        return Settings(**(values | overrides))

    return factory


@pytest.fixture(autouse=True)
def isolate_globals() -> Iterator[None]:
    """create_app() reconfigures logging and settings are cached; undo both."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    root.handlers[:] = handlers
    root.setLevel(level)

"""Integration fixtures: real Postgres and Qdrant from docker-compose.test.yml."""

import logging
import os
import pathlib
import socket
import subprocess  # nosec B404 - runs the Alembic CLI under test with fixed arguments
import sys
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import pytest
from pydantic import HttpUrl, SecretStr
from sqlalchemy import text

from app.core.settings import Settings, get_settings
from app.infra.db import create_engine

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
            # Apps under test purge only when a test asks (tests/integration/test_retention.py
            # runs the purge itself).
            "retention_purge_interval_minutes": 0,
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


BACKEND = pathlib.Path(__file__).resolve().parents[2]
ALEMBIC_INI = BACKEND / "alembic.ini"


@pytest.fixture
async def scratch_db(make_settings: SettingsFactory) -> AsyncIterator[str]:
    """A fresh, uniquely named database, dropped (forcefully) after the test."""
    name = f"it_scratch_{uuid.uuid4().hex[:12]}"
    admin = create_engine(make_settings(postgres_db=PG_DB)).execution_options(
        isolation_level="AUTOCOMMIT"
    )
    quoted = admin.dialect.identifier_preparer.quote(name)
    try:
        async with admin.connect() as connection:
            await connection.execute(text(f"CREATE DATABASE {quoted}"))
        yield name
        async with admin.connect() as connection:
            await connection.execute(text(f"DROP DATABASE IF EXISTS {quoted} WITH (FORCE)"))
    finally:
        await admin.dispose()


def run_alembic(
    database: str, *args: str, cwd: pathlib.Path, log_level: str = "INFO"
) -> subprocess.CompletedProcess[str]:
    """Run the real CLI from an unrelated directory with only env-var configuration."""
    env = {k: v for k, v in os.environ.items() if k.lower() not in Settings.model_fields}
    env |= {
        "POSTGRES_HOST": PG_HOST,
        "POSTGRES_PORT": str(PG_PORT),
        "POSTGRES_DB": database,
        "POSTGRES_USER": PG_USER,
        "POSTGRES_PASSWORD": PG_PASSWORD,
        "LOG_LEVEL": log_level,
    }
    command = [sys.executable, "-m", "alembic", "-c", str(ALEMBIC_INI), *args]
    result = subprocess.run(  # noqa: S603 - fixed command, no shell  # nosec B603
        command, cwd=cwd, env=env, capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, f"alembic {' '.join(args)} failed:\n{result.stderr}"
    return result

"""Alembic migrations against a scratch database per test on the real Postgres."""

import json
import os
import pathlib
import subprocess  # nosec B404 - runs the Alembic CLI under test with fixed arguments
import sys
import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text

from app.core.settings import Settings
from app.infra.db import create_engine
from tests.integration.conftest import (
    PG_DB,
    PG_HOST,
    PG_PASSWORD,
    PG_PORT,
    PG_USER,
    SettingsFactory,
)

pytestmark = pytest.mark.integration

BACKEND = pathlib.Path(__file__).resolve().parents[2]
ALEMBIC_INI = BACKEND / "alembic.ini"
BASELINE_REVISION = "0001"


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


async def versions(settings: Settings) -> list[str]:
    engine = create_engine(settings)
    try:
        async with engine.connect() as connection:
            rows = await connection.execute(text("SELECT version_num FROM alembic_version"))
            return [row[0] for row in rows]
    finally:
        await engine.dispose()


async def test_upgrade_is_idempotent_and_downgrade_returns_to_base(
    scratch_db: str, make_settings: SettingsFactory, tmp_path: pathlib.Path
) -> None:
    settings = make_settings(postgres_db=scratch_db)

    run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path)
    assert await versions(settings) == [BASELINE_REVISION]

    run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path)
    assert await versions(settings) == [BASELINE_REVISION]

    run_alembic(scratch_db, "downgrade", "base", cwd=tmp_path)
    assert await versions(settings) == []


async def test_models_and_migrations_have_no_drift(scratch_db: str, tmp_path: pathlib.Path) -> None:
    run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path)

    result = run_alembic(scratch_db, "check", cwd=tmp_path)

    assert "No new upgrade operations detected" in result.stdout + result.stderr


async def test_migration_output_is_json_lines_only(scratch_db: str, tmp_path: pathlib.Path) -> None:
    result = run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path)

    lines = [line for line in (result.stdout + result.stderr).splitlines() if line.strip()]
    parsed = [json.loads(line) for line in lines]
    assert any("Running upgrade" in entry["message"] for entry in parsed)
    assert all(entry["logger"].startswith("alembic") for entry in parsed)


async def test_verbose_migration_output_never_contains_the_password(
    scratch_db: str, tmp_path: pathlib.Path
) -> None:
    result = run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path, log_level="DEBUG")

    output = result.stdout + result.stderr
    assert output.strip(), "expected log output at DEBUG"
    assert PG_PASSWORD not in output

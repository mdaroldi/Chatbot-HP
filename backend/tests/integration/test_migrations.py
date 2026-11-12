"""Alembic migrations against a scratch database per test on the real Postgres."""

import json
import pathlib

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.core.settings import Settings
from app.infra.db import create_engine
from tests.integration.conftest import PG_PASSWORD, SettingsFactory, run_alembic

pytestmark = pytest.mark.integration

HEAD_REVISION = "0002"


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
    assert await versions(settings) == [HEAD_REVISION]

    run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path)
    assert await versions(settings) == [HEAD_REVISION]

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


INDEX_BUILDS_COLUMNS = {
    "build_id": "character varying",
    "created_at": "timestamp with time zone",
    "status": "character varying",
    "collection": "character varying",
    "document_hashes": "json",
    "chunking_version": "character varying",
    "dense_model": "character varying",
    "sparse_model": "character varying",
    "models": "json",
    "chunk_count": "integer",
    "chunk_counts": "json",
    "drop_stats": "json",
    "warnings": "json",
}


async def columns(settings: Settings, table: str) -> dict[str, tuple[str, str]]:
    query = text(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = :table"
    )
    engine = create_engine(settings)
    try:
        async with engine.connect() as connection:
            rows = await connection.execute(query, {"table": table})
            return {name: (data_type, nullable) for name, data_type, nullable in rows}
    finally:
        await engine.dispose()


async def test_migration_0002_creates_index_builds_with_required_columns(
    scratch_db: str, make_settings: SettingsFactory, tmp_path: pathlib.Path
) -> None:
    settings = make_settings(postgres_db=scratch_db)

    run_alembic(scratch_db, "upgrade", "0002", cwd=tmp_path)

    found = await columns(settings, "index_builds")
    assert {name: data_type for name, (data_type, _) in found.items()} == INDEX_BUILDS_COLUMNS
    assert {nullable for _, nullable in found.values()} == {"NO"}


async def test_downgrade_to_baseline_drops_index_builds(
    scratch_db: str, make_settings: SettingsFactory, tmp_path: pathlib.Path
) -> None:
    settings = make_settings(postgres_db=scratch_db)
    run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path)
    assert await columns(settings, "index_builds"), "head must create index_builds"

    run_alembic(scratch_db, "downgrade", "0001", cwd=tmp_path)

    assert await columns(settings, "index_builds") == {}
    assert await versions(settings) == ["0001"]


ROW = (
    "INSERT INTO index_builds VALUES (:build_id, now(), :status, :collection, '{}', 'v1', "
    "'d', 's', '{}', :chunk_count, '{}', '{}', '[]')"
)


async def insert_row(settings: Settings, **values: object) -> None:
    params = {"build_id": "b1", "status": "succeeded", "collection": "c1", "chunk_count": 1}
    engine = create_engine(settings)
    try:
        async with engine.begin() as connection:
            await connection.execute(text(ROW), params | values)
    finally:
        await engine.dispose()


async def test_index_builds_constraints_hold_in_the_database(
    scratch_db: str, make_settings: SettingsFactory, tmp_path: pathlib.Path
) -> None:
    settings = make_settings(postgres_db=scratch_db)
    run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path)
    await insert_row(settings)  # a valid row is accepted

    for values, constraint in (
        ({"build_id": "b2", "collection": "c2", "status": "running"}, "ck_index_builds_status"),
        (
            {"build_id": "b2", "collection": "c2", "chunk_count": -1},
            "ck_index_builds_chunk_count_non_negative",
        ),
        ({"build_id": "b2"}, "uq_index_builds_collection"),
        ({"collection": "c2"}, "pk_index_builds"),
    ):
        with pytest.raises(IntegrityError, match=constraint):
            await insert_row(settings, **values)

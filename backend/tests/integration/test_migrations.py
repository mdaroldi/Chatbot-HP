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

HEAD_REVISION = "0004"


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


ROW_0002 = (
    "INSERT INTO index_builds (build_id, created_at, status, collection, document_hashes, "
    "chunking_version, dense_model, sparse_model, models, chunk_count, chunk_counts, "
    "drop_stats, warnings) VALUES (:build_id, now(), :status, :collection, '{}', 'v1', "
    "'d', 's', '{}', :chunk_count, '{}', '{}', '[]')"
)
ROW = (
    "INSERT INTO index_builds (build_id, created_at, status, collection, document_hashes, "
    "chunking_version, dense_model, sparse_model, models, chunk_count, chunk_counts, "
    "drop_stats, warnings, error_type) VALUES (:build_id, now(), :status, :collection, "
    "'{}', 'v1', 'd', 's', '{}', :chunk_count, '{}', '{}', '[]', :error_type)"
)


async def insert_row(settings: Settings, statement: str = ROW, **values: object) -> None:
    params = {
        "build_id": "b1",
        "status": "succeeded",
        "collection": "c1",
        "chunk_count": 1,
        "error_type": None,
    }
    engine = create_engine(settings)
    try:
        async with engine.begin() as connection:
            await connection.execute(text(statement), params | values)
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


ERROR_TYPE_CHECK = "ck_index_builds_error_type_status"


async def test_migration_0003_adds_a_nullable_error_type(
    scratch_db: str, make_settings: SettingsFactory, tmp_path: pathlib.Path
) -> None:
    settings = make_settings(postgres_db=scratch_db)
    run_alembic(scratch_db, "upgrade", "0002", cwd=tmp_path)
    assert "error_type" not in await columns(settings, "index_builds")

    run_alembic(scratch_db, "upgrade", "0003", cwd=tmp_path)

    found = await columns(settings, "index_builds")
    assert found["error_type"] == ("character varying", "YES")
    assert {name for name, (_, nullable) in found.items() if nullable == "YES"} == {"error_type"}


async def test_error_type_must_match_the_status_in_the_database(
    scratch_db: str, make_settings: SettingsFactory, tmp_path: pathlib.Path
) -> None:
    settings = make_settings(postgres_db=scratch_db)
    run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path)
    await insert_row(settings)  # succeeded without an error type is accepted
    await insert_row(settings, build_id="b2", collection="c2", status="failed", error_type="E")

    for values in (
        {"status": "failed", "error_type": None},
        {"status": "succeeded", "error_type": "OSError"},
    ):
        with pytest.raises(IntegrityError, match=ERROR_TYPE_CHECK):
            await insert_row(settings, build_id="b3", collection="c3", **values)


async def test_downgrade_to_0002_drops_error_type_and_its_check(
    scratch_db: str, make_settings: SettingsFactory, tmp_path: pathlib.Path
) -> None:
    settings = make_settings(postgres_db=scratch_db)
    run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path)
    assert "error_type" in await columns(settings, "index_builds")

    run_alembic(scratch_db, "downgrade", "0002", cwd=tmp_path)

    assert await versions(settings) == ["0002"]
    assert set(await columns(settings, "index_builds")) == set(INDEX_BUILDS_COLUMNS)
    # Without the check, a failed row with no error type fits the 0002 schema again.
    await insert_row(settings, ROW_0002, status="failed")


async def test_migration_0004_adds_nullable_sparse_params_and_keeps_old_rows_null(
    scratch_db: str, make_settings: SettingsFactory, tmp_path: pathlib.Path
) -> None:
    """NULL on a row built before 0004 means fastembed's BM25 defaults (ADR 0009)."""
    settings = make_settings(postgres_db=scratch_db)
    run_alembic(scratch_db, "upgrade", "0003", cwd=tmp_path)
    await insert_row(settings)

    run_alembic(scratch_db, "upgrade", "0004", cwd=tmp_path)

    found = await columns(settings, "index_builds")
    assert found["sparse_params"] == ("json", "YES")
    assert await sparse_params(settings) == [("b1", None)]


async def test_downgrade_to_0003_drops_sparse_params(
    scratch_db: str, make_settings: SettingsFactory, tmp_path: pathlib.Path
) -> None:
    settings = make_settings(postgres_db=scratch_db)
    run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path)
    await insert_row(settings)

    run_alembic(scratch_db, "downgrade", "0003", cwd=tmp_path)

    assert await versions(settings) == ["0003"]
    assert "sparse_params" not in await columns(settings, "index_builds")
    assert set(await columns(settings, "index_builds")) == {*INDEX_BUILDS_COLUMNS, "error_type"}


async def sparse_params(settings: Settings) -> list[tuple[str, object]]:
    engine = create_engine(settings)
    try:
        async with engine.connect() as connection:
            rows = await connection.execute(
                text("SELECT build_id, sparse_params FROM index_builds")
            )
            return [(build_id, params) for build_id, params in rows]
    finally:
        await engine.dispose()

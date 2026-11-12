"""SqlManifestStore on SQLite (aiosqlite); the Postgres integration tests repeat the core cases.

The table comes from the ORM metadata here. Migration 0002 builds the same table on
Postgres, and `alembic check` proves the two agree.
"""

import pathlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.infra.db import create_session_factory
from app.infra.manifest_repository import SqlManifestStore
from app.infra.orm import Base, IndexBuildRow
from tests.manifest_data import make_build, make_failed_build


@pytest.fixture
async def sessions(tmp_path: pathlib.Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'manifest.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield create_session_factory(engine)
    await engine.dispose()


@pytest.fixture
def store(sessions: async_sessionmaker[AsyncSession]) -> SqlManifestStore:
    return SqlManifestStore(sessions)


async def test_recorded_build_reads_back_unchanged(store: SqlManifestStore) -> None:
    build = make_build()

    await store.record(build)

    loaded = await store.latest_successful(build.collection)
    assert loaded == build
    assert loaded is not None
    assert loaded.created_at.tzinfo is UTC


async def test_unknown_collection_has_no_build(store: SqlManifestStore) -> None:
    build = make_build()
    await store.record(build)

    assert await store.latest_successful("hpdocs_chunks_v1_20261111t000000z") is None
    assert await store.latest_successful(build.collection) == build


async def test_failed_build_is_never_returned_as_successful(store: SqlManifestStore) -> None:
    succeeded = make_build("20261112t143005z")
    failed = make_failed_build("20261112t150000z")

    await store.record(succeeded)
    await store.record(failed)

    assert await store.latest_successful(failed.collection) is None
    assert await store.latest_successful(succeeded.collection) == succeeded


async def test_a_build_is_recorded_once(store: SqlManifestStore) -> None:
    build = make_build()
    await store.record(build)

    with pytest.raises(IntegrityError):
        await store.record(make_failed_build(make_build().build_id))

    assert await store.latest_successful(build.collection) == build


async def test_a_collection_belongs_to_one_build(store: SqlManifestStore) -> None:
    first = make_build("20261112t143005z")
    await store.record(first)

    with pytest.raises(IntegrityError):
        await store.record(make_build("20261112t150000z", collection=first.collection))


async def test_row_stores_names_and_total_for_queries(
    store: SqlManifestStore, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await store.record(make_build())

    async with sessions() as session:
        row = (await session.execute(select(IndexBuildRow))).scalar_one()
    assert (row.dense_model, row.sparse_model, row.chunk_count) == ("demo/dense", "demo/sparse", 10)
    assert row.models["dense"]["revision"] == make_build().dense.revision
    assert row.warnings == [
        {"kind": "prune_failed", "collection": "hpdocs_chunks_v1_old", "error_type": "OSError"}
    ]


async def test_failed_build_stores_its_error_type(
    store: SqlManifestStore, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await store.record(make_build())
    await store.record(make_failed_build(error_type="UnexpectedResponse"))

    async with sessions() as session:
        rows = (await session.execute(select(IndexBuildRow.status, IndexBuildRow.error_type))).all()
    assert sorted(rows) == [("failed", "UnexpectedResponse"), ("succeeded", None)]


def raw_row(build_id: str, status: str, error_type: str | None) -> IndexBuildRow:
    """A row that bypasses the domain checks, so only the database constraint can stop it."""
    model = {"name": "d", "source": "d", "revision": "r", "files": {}}
    return IndexBuildRow(
        build_id=build_id,
        created_at=datetime(2026, 11, 12, tzinfo=UTC),
        status=status,
        collection=f"c_{build_id}",
        document_hashes={},
        chunking_version="v1",
        dense_model="d",
        sparse_model="s",
        models={"dense": model, "sparse": model},
        chunk_count=0,
        chunk_counts={},
        drop_stats={},
        warnings=[],
        error_type=error_type,
    )


@pytest.mark.parametrize(
    ("status", "error_type"),
    [("failed", None), ("succeeded", "OSError")],
)
async def test_database_rejects_an_error_type_that_contradicts_the_status(
    sessions: async_sessionmaker[AsyncSession], status: str, error_type: str | None
) -> None:
    async with sessions() as session, session.begin():  # the valid pair is accepted
        session.add(raw_row("b1", "failed", "OSError"))
        session.add(raw_row("b2", "succeeded", None))

    with pytest.raises(IntegrityError, match="ck_index_builds_error_type_status"):
        async with sessions() as session, session.begin():
            session.add(raw_row("b3", status, error_type))

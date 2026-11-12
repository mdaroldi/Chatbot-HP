"""SqlManifestStore on the real Postgres, on a scratch database built by migration 0002."""

import pathlib
from collections.abc import AsyncIterator
from datetime import UTC

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.infra.db import create_engine, create_session_factory
from app.infra.manifest_repository import SqlManifestStore
from app.infra.orm import IndexBuildRow
from tests.integration.conftest import SettingsFactory, run_alembic
from tests.manifest_data import make_build, make_failed_build

pytestmark = pytest.mark.integration


@pytest.fixture
async def sessions(
    scratch_db: str, make_settings: SettingsFactory, tmp_path: pathlib.Path
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path)
    engine = create_engine(make_settings(postgres_db=scratch_db))
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
    assert loaded.created_at.utcoffset() is not None
    assert loaded.created_at.tzinfo is UTC


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


async def test_failed_build_keeps_its_error_type(
    store: SqlManifestStore, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await store.record(make_build())
    await store.record(make_failed_build(error_type="UnexpectedResponse"))

    async with sessions() as session:
        rows = (await session.execute(select(IndexBuildRow.status, IndexBuildRow.error_type))).all()
    assert sorted(rows) == [("failed", "UnexpectedResponse"), ("succeeded", None)]

"""SqlManifestStore on the real Postgres, on a scratch database built by migration 0002."""

import pathlib
from collections.abc import AsyncIterator
from datetime import UTC

import pytest
from sqlalchemy.exc import IntegrityError

from app.domain.ingestion import BuildStatus
from app.infra.db import create_engine, create_session_factory
from app.infra.manifest_repository import SqlManifestStore
from tests.integration.conftest import SettingsFactory, run_alembic
from tests.manifest_data import make_build

pytestmark = pytest.mark.integration


@pytest.fixture
async def store(
    scratch_db: str, make_settings: SettingsFactory, tmp_path: pathlib.Path
) -> AsyncIterator[SqlManifestStore]:
    run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path)
    engine = create_engine(make_settings(postgres_db=scratch_db))
    yield SqlManifestStore(create_session_factory(engine))
    await engine.dispose()


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
    failed = make_build("20261112t150000z", status=BuildStatus.FAILED)

    await store.record(succeeded)
    await store.record(failed)

    assert await store.latest_successful(failed.collection) is None
    assert await store.latest_successful(succeeded.collection) == succeeded


async def test_a_build_is_recorded_once(store: SqlManifestStore) -> None:
    build = make_build()
    await store.record(build)

    with pytest.raises(IntegrityError):
        await store.record(make_build(status=BuildStatus.FAILED))

    assert await store.latest_successful(build.collection) == build

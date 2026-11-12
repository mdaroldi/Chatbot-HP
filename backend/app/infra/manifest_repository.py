"""Index build manifest in the relational database (ADR 0006)."""

from datetime import UTC

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.ingestion import (
    BuildStatus,
    BuildWarning,
    BuildWarningKind,
    IndexBuild,
    ModelFiles,
)
from app.infra.orm import IndexBuildRow, ModelFilesJson


def _model_json(files: ModelFiles) -> ModelFilesJson:
    return {
        "name": files.name,
        "source": files.source,
        "revision": files.revision,
        "files": dict(files.files),
    }


def _model_files(data: ModelFilesJson) -> ModelFiles:
    return ModelFiles(data["name"], data["source"], data["revision"], data["files"])


def _to_row(build: IndexBuild) -> IndexBuildRow:
    return IndexBuildRow(
        build_id=build.build_id,
        created_at=build.created_at,
        status=build.status.value,
        collection=build.collection,
        document_hashes=dict(build.document_hashes),
        chunking_version=build.chunking_version,
        dense_model=build.dense_model,
        sparse_model=build.sparse_model,
        models={"dense": _model_json(build.dense), "sparse": _model_json(build.sparse)},
        chunk_count=build.chunk_count,
        chunk_counts=dict(build.chunk_counts),
        drop_stats={doc: dict(stats) for doc, stats in build.drop_stats.items()},
        warnings=[
            {"kind": w.kind.value, "collection": w.collection, "error_type": w.error_type}
            for w in build.warnings
        ],
        error_type=build.error_type,
    )


def _to_build(row: IndexBuildRow) -> IndexBuild:
    return IndexBuild(
        build_id=row.build_id,
        # Stored in UTC (the domain requires it). asyncpg returns it as UTC; SQLite, used
        # by the unit tests, drops the zone, so setting UTC is right for both.
        created_at=row.created_at.replace(tzinfo=UTC),
        collection=row.collection,
        status=BuildStatus(row.status),
        document_hashes=row.document_hashes,
        chunking_version=row.chunking_version,
        dense=_model_files(row.models["dense"]),
        sparse=_model_files(row.models["sparse"]),
        chunk_counts=row.chunk_counts,
        drop_stats=row.drop_stats,
        warnings=tuple(
            BuildWarning(BuildWarningKind(w["kind"]), w["collection"], w["error_type"])
            for w in row.warnings
        ),
        error_type=row.error_type,
    )


class SqlManifestStore:
    """ManifestStore over SQLAlchemy: one row per build that started, never updated.

    A build ID or collection that is already recorded raises IntegrityError, so a
    manifest entry can never be overwritten.
    """

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def record(self, build: IndexBuild) -> None:
        async with self._sessions() as session, session.begin():
            session.add(_to_row(build))

    async def succeeded_build(self, collection: str) -> IndexBuild | None:
        query = select(IndexBuildRow).where(
            IndexBuildRow.collection == collection,
            IndexBuildRow.status == BuildStatus.SUCCEEDED.value,
        )
        async with self._sessions() as session:
            row = (await session.execute(query)).scalar_one_or_none()
        return None if row is None else _to_build(row)

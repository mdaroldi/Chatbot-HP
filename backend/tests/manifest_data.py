"""Synthetic index builds shared by the manifest repository's unit and integration tests."""

from dataclasses import replace
from datetime import UTC, datetime

from app.domain.ingestion import (
    BuildStatus,
    BuildWarning,
    BuildWarningKind,
    IndexBuild,
    ModelFiles,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"


def make_build(build_id: str = "20261112t143005z", **overrides: object) -> IndexBuild:
    build = IndexBuild(
        build_id=build_id,
        created_at=datetime(2026, 11, 12, 14, 30, 5, tzinfo=UTC),
        collection=f"hpdocs_chunks_v1_{build_id}",
        status=BuildStatus.SUCCEEDED,
        document_hashes={"demo-guide": "a" * 64, "demo-manual": "b" * 64},
        chunking_version="v1",
        dense=ModelFiles(
            name="demo/dense",
            source="demo/dense-onnx",
            revision=REVISION,
            files={"model.onnx": "c" * 64, "tokenizer.json": "d" * 64},
        ),
        sparse=ModelFiles(
            name="demo/sparse", source="demo/sparse", revision=REVISION, files={"x.txt": "e" * 64}
        ),
        chunk_counts={"demo-guide": 7, "demo-manual": 3},
        drop_stats={"demo-guide": {"footer": 3, "toc": 1}, "demo-manual": {}},
        warnings=(BuildWarning(BuildWarningKind.PRUNE_FAILED, "hpdocs_chunks_v1_old", "OSError"),),
    )
    return replace(build, **overrides)  # type: ignore[arg-type]  # test data

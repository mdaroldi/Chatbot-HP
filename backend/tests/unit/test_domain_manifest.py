"""Behavior of the index manifest types: builds, model files and build warnings."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.domain.ingestion import (
    BuildStatus,
    BuildWarning,
    BuildWarningKind,
    IndexBuild,
    ModelDownloadError,
    ModelError,
    ModelFiles,
    ModelIntegrityError,
    ModelNotAllowedError,
    ModelPinsError,
)

SHA = "a" * 64
REVISION = "0123456789abcdef0123456789abcdef01234567"
CREATED = datetime(2026, 11, 12, 14, 30, 5, tzinfo=UTC)


def model_files(**overrides: object) -> ModelFiles:
    values: dict[str, object] = {
        "name": "demo/dense",
        "source": "demo/dense-onnx",
        "revision": REVISION,
        "files": {"model.onnx": SHA, "nested/tokenizer.json": "b" * 64},
    }
    return ModelFiles(**(values | overrides))  # type: ignore[arg-type]  # test data


def test_model_files_record_name_source_revision_and_file_digests() -> None:
    files = model_files()

    assert (files.name, files.source, files.revision) == ("demo/dense", "demo/dense-onnx", REVISION)
    assert files.files["nested/tokenizer.json"] == "b" * 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", " ", "name must not be blank"),
        ("source", "", "source must not be blank"),
        ("revision", "0123456", "revision must be a full commit SHA"),
        ("revision", REVISION.upper(), "revision must be a full commit SHA"),
        ("files", {}, "files must not be empty"),
        ("files", {"model.onnx": "not-a-hash"}, "files must hold SHA-256 digests"),
        ("files", {"/abs/model.onnx": SHA}, "file paths must be relative"),
        ("files", {"../model.onnx": SHA}, "file paths must be relative"),
        ("files", {"": SHA}, "file paths must be relative"),
    ],
)
def test_model_files_reject_invalid_values(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        model_files(**{field: value})


def test_build_warning_names_the_collection_and_error_type_only() -> None:
    warning = BuildWarning(
        kind=BuildWarningKind.PRUNE_FAILED,
        collection="hpdocs_chunks_v1_20261111t090000z",
        error_type="UnexpectedResponse",
    )

    assert warning.kind is BuildWarningKind.PRUNE_FAILED
    assert warning.error_type == "UnexpectedResponse"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("collection", " ", "collection must not be blank"),
        ("error_type", "", "error_type must be an exception type name"),
        # a message, not a type name: warnings never carry exception text
        ("error_type", "connection refused by host", "error_type must be an exception type name"),
    ],
)
def test_build_warning_rejects_invalid_values(field: str, value: str, message: str) -> None:
    values = {"collection": "hpdocs_chunks_v1_x", "error_type": "OSError"} | {field: value}
    with pytest.raises(ValueError, match=message):
        BuildWarning(kind=BuildWarningKind.PRUNE_FAILED, **values)


def index_build(**overrides: object) -> IndexBuild:
    values: dict[str, object] = {
        "build_id": "20261112t143005z",
        "created_at": CREATED,
        "collection": "hpdocs_chunks_v1_20261112t143005z",
        "status": BuildStatus.SUCCEEDED,
        "document_hashes": {"demo-guide": SHA, "demo-manual": "c" * 64},
        "chunking_version": "v1",
        "dense": model_files(),
        "sparse": model_files(name="demo/sparse", source="demo/sparse"),
        "chunk_counts": {"demo-guide": 7, "demo-manual": 3},
        "drop_stats": {"demo-guide": {"footer": 3}},
    }
    return IndexBuild(**(values | overrides))  # type: ignore[arg-type]  # test data


def test_index_build_records_what_produced_a_collection() -> None:
    build = index_build()

    assert build.status is BuildStatus.SUCCEEDED
    assert build.document_hashes["demo-guide"] == SHA
    assert (build.chunk_count, build.dense_model, build.sparse_model) == (
        10,
        "demo/dense",
        "demo/sparse",
    )
    assert build.warnings == ()


def test_index_build_keeps_its_warnings() -> None:
    warning = BuildWarning(BuildWarningKind.PRUNE_FAILED, "hpdocs_chunks_v1_old", "OSError")

    assert index_build(warnings=(warning,)).warnings == (warning,)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("build_id", "20261112T143005Z", "build_id must look like 20261112t143005z"),
        ("build_id", "build-1", "build_id must look like 20261112t143005z"),
        ("build_id", "", "build_id must look like 20261112t143005z"),
        ("created_at", CREATED.replace(tzinfo=None), "created_at must be in UTC"),
        ("created_at", CREATED.astimezone(timezone(timedelta(hours=1))), "created_at must be in"),
        ("collection", "", "collection must not be blank"),
        ("document_hashes", {}, "document_hashes must not be empty"),
        ("document_hashes", {"demo-guide": "not-a-hash"}, "document_hashes must hold SHA-256"),
        ("chunking_version", " ", "chunking_version must not be blank"),
        ("chunk_counts", {"demo-guide": -1, "demo-manual": 3}, "chunk counts must be >= 0"),
        ("chunk_counts", {"demo-guide": 7, "other": 3}, "chunk counts must name indexed docu"),
        ("chunk_counts", {"demo-guide": 7}, "a succeeded build counts every document"),
        ("chunk_counts", {"demo-guide": 0, "demo-manual": 0}, "a succeeded build has chunks"),
    ],
)
def test_index_build_rejects_invalid_values(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        index_build(**{field: value})


def test_failed_build_may_stop_before_every_document_is_chunked() -> None:
    # The same counts are rejected for a succeeded build (above).
    build = index_build(status=BuildStatus.FAILED, chunk_counts={"demo-guide": 7})

    assert build.chunk_count == 7


def test_failed_build_may_have_no_chunks() -> None:
    build = index_build(status=BuildStatus.FAILED, chunk_counts={})

    assert build.chunk_count == 0


def test_integrity_error_names_the_model_the_file_and_both_digests() -> None:
    error = ModelIntegrityError("demo/dense", "model.onnx", expected=SHA, actual="b" * 64)

    assert (error.model, error.file, error.expected, error.actual) == (
        "demo/dense",
        "model.onnx",
        SHA,
        "b" * 64,
    )
    assert str(error) == f"demo/dense: model.onnx has SHA-256 {'b' * 64}, pinned {SHA}"
    assert isinstance(error, ModelError)


def test_integrity_error_reports_a_missing_file() -> None:
    error = ModelIntegrityError("demo/dense", "model.onnx", expected=SHA, actual=None)

    assert str(error) == f"demo/dense: model.onnx is missing, pinned {SHA}"


def test_model_errors_are_distinct() -> None:
    kinds = {ModelNotAllowedError, ModelPinsError, ModelDownloadError, ModelIntegrityError}

    assert all(issubclass(kind, ModelError) for kind in kinds)
    assert not issubclass(ModelDownloadError, ModelIntegrityError)
    assert not issubclass(ModelIntegrityError, ModelDownloadError)

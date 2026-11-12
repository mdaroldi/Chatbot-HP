"""The `--if-needed` rule: when the live build may be reused instead of rebuilt.

The live build is reused only if every input a build would record now is what it
recorded then, and its collection still holds every point. The comparison reads
`corpus.toml` and `models.toml` only, so a skipped run never downloads, hashes or
loads anything (ADR 0006). The code is not compared: a chunker change must bump the
chunking version, which a corpus test of the chunk ids enforces.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from app.domain.ingestion import IndexBuild, ModelFiles


class StaleReason(StrEnum):
    """Why a run rebuilds: logged, so an unexpected rebuild can be explained."""

    NO_LIVE_INDEX = "no_live_index"
    NO_MANIFEST = "no_manifest"  # the alias points to a collection with no succeeded build
    DOCUMENTS = "documents"
    CHUNKING_VERSION = "chunking_version"
    DENSE_MODEL = "dense_model"
    SPARSE_MODEL = "sparse_model"
    DENSE_PIN = "dense_pin"
    SPARSE_PIN = "sparse_pin"
    POINT_COUNT = "point_count"


@dataclass(frozen=True)
class Expected:
    """What a build made now would record. A pin is None when the model has none."""

    document_hashes: Mapping[str, str]
    chunking_version: str
    dense_model: str
    sparse_model: str
    dense_pin: ModelFiles | None
    sparse_pin: ModelFiles | None


def _same_files(recorded: ModelFiles, pin: ModelFiles | None) -> bool:
    if pin is None:
        return False
    return (recorded.source, recorded.revision, dict(recorded.files)) == (
        pin.source,
        pin.revision,
        dict(pin.files),
    )


def stale_reason(live: IndexBuild, expected: Expected, point_count: int) -> StaleReason | None:
    """The first input that differs from the live build, or None when it is current."""
    checks = (
        (dict(live.document_hashes) == dict(expected.document_hashes), StaleReason.DOCUMENTS),
        (live.chunking_version == expected.chunking_version, StaleReason.CHUNKING_VERSION),
        (live.dense_model == expected.dense_model, StaleReason.DENSE_MODEL),
        (live.sparse_model == expected.sparse_model, StaleReason.SPARSE_MODEL),
        (_same_files(live.dense, expected.dense_pin), StaleReason.DENSE_PIN),
        (_same_files(live.sparse, expected.sparse_pin), StaleReason.SPARSE_PIN),
        (point_count == live.chunk_count, StaleReason.POINT_COUNT),
    )
    return next((reason for holds, reason in checks if not holds), None)

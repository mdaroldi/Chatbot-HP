"""The ingest pipeline: --if-needed check, model verification, build, publish, manifest.

Ports are in-memory fakes, except the vector index, which is QdrantIndex on
qdrant-client local mode (as in the publish tests).
"""

import logging
import pathlib
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import pytest
from qdrant_client import AsyncQdrantClient

from app.domain.ingestion import (
    BuildStatus,
    BuildWarning,
    BuildWarningKind,
    CorpusDownloadError,
    IndexPublishError,
    ModelDownloadError,
    ModelIntegrityError,
)
from app.infra.qdrant_index import QdrantIndex
from app.ingestion.freshness import StaleReason
from app.ingestion.pipeline import IngestConfig, IngestPorts, IngestResult, ingest
from tests.index_data import make_batch
from tests.unit.ingest_fakes import (
    CLOCK,
    DENSE,
    DOCUMENTS,
    PINS,
    SPARSE,
    Calls,
    FakeCorpus,
    FakeEmbedder,
    FakeModels,
    FakeProcessor,
    InMemoryManifest,
)
from tests.unit.test_ingestion_publish import FailingDelete

pytestmark = pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect in the local Qdrant:UserWarning"
)

ALIAS = "hpdocs_chunks"
CONFIG = IngestConfig(
    alias=ALIAS,
    keep_previous=True,
    dense_model=DENSE.name,
    sparse_model=SPARSE.name,
    chunking_version="v1",
)
COUNTS = {"demo-guide": 3, "demo-manual": 2}


@pytest.fixture
async def client() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(location=":memory:")
    yield client
    await client.close()


class World:
    """One set of ports that survives several ingest runs, like the compose volumes."""

    def __init__(self, client: AsyncQdrantClient, folder: pathlib.Path) -> None:
        self.calls = Calls()
        self.folder = folder
        self.index: QdrantIndex = QdrantIndex(client)
        self.manifest = InMemoryManifest(self.calls)
        self.corpus = FakeCorpus(self.calls, folder)
        self.models = FakeModels(self.calls, folder)
        self.embedder = FakeEmbedder(self.calls)
        self.counts: Mapping[str, int] = COUNTS
        self.now = CLOCK

    def _processor(self, dense: object) -> FakeProcessor:
        self.calls.log.append(("build", "processor"))
        assert dense is not None
        return FakeProcessor(self.calls, self.counts)

    def _embedder(self, dense: object, sparse: object) -> FakeEmbedder:
        self.calls.log.append(("build", "embedder"))
        assert dense is not None
        assert sparse is not None
        return self.embedder

    def ports(self) -> IngestPorts:
        return IngestPorts(
            corpus=self.corpus,
            models=self.models,
            index=self.index,
            manifest=self.manifest,
            processor=self._processor,
            embedder=self._embedder,
            clock=lambda: self.now,
        )

    async def run(self, *, force: bool = False, **config: object) -> IngestResult:
        self.calls.log.clear()
        settings = replace(CONFIG, **config)  # type: ignore[arg-type]  # test data
        return await ingest(DOCUMENTS, PINS, settings, self.ports(), force=force)

    def later(self, seconds: int = 60) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def world(client: AsyncQdrantClient, tmp_path: pathlib.Path) -> World:
    return World(client, tmp_path)


# ------------------------------------------------------------------ first build


async def test_the_first_run_builds_publishes_and_records_the_build(world: World) -> None:
    result = await world.run()

    assert (result.skipped, result.reason) == (False, StaleReason.NO_LIVE_INDEX)
    assert result.build_id == "20261112t143005z"
    assert result.collection == "hpdocs_chunks_v1_20261112t143005z"
    assert await world.index.alias_target(ALIAS) == result.collection
    assert await world.index.count(ALIAS) == 5
    [build] = world.manifest.builds
    assert build.status is BuildStatus.SUCCEEDED
    assert (build.build_id, build.collection, build.created_at) == (
        result.build_id,
        result.collection,
        CLOCK,
    )
    assert build.document_hashes == {"demo-guide": "a" * 64, "demo-manual": "b" * 64}
    assert dict(build.chunk_counts) == COUNTS
    assert (build.chunking_version, build.dense, build.sparse) == ("v1", DENSE, SPARSE)
    assert set(build.drop_stats) == {"demo-guide", "demo-manual"}
    assert (build.warnings, build.error_type) == ((), None)


async def test_every_chunk_text_is_embedded_once_in_order(world: World) -> None:
    await world.run()

    assert len(world.embedder.texts) == 5
    assert world.embedder.texts[0].startswith("Demo Guide > Setup > Step 0")


async def test_the_build_id_is_the_lowercase_utc_start_time(world: World) -> None:
    world.now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

    result = await world.run()

    assert result.build_id == "20260102t030405z"


async def test_a_clock_outside_utc_is_converted_to_utc(world: World) -> None:
    world.now = datetime(2026, 1, 2, 4, 4, 5, tzinfo=timezone(timedelta(hours=1)))

    result = await world.run()

    assert result.build_id == "20260102t030405z"
    assert world.manifest.builds[0].created_at.utcoffset() == timedelta(0)


async def test_the_collection_prefix_follows_the_chunking_version(world: World) -> None:
    world.counts = COUNTS

    def v2_processor(dense: object) -> FakeProcessor:
        del dense
        return FakeProcessor(world.calls, COUNTS, version="v2")

    ports = replace(world.ports(), processor=v2_processor)
    result = await ingest(
        DOCUMENTS, PINS, replace(CONFIG, chunking_version="v2"), ports, force=True
    )

    assert result.collection == "hpdocs_chunks_v2_20261112t143005z"


# ------------------------------------------------------------------ --if-needed


async def test_an_unchanged_index_is_skipped_without_touching_corpus_or_models(
    world: World,
) -> None:
    first = await world.run()
    world.later()

    second = await world.run()

    assert (second.skipped, second.reason) == (True, None)
    assert (second.collection, second.build_id) == (first.collection, first.build_id)
    # Only the alias and manifest were read: no download, no hashing, no model load.
    kinds = {kind for kind, _ in world.calls.log}
    assert kinds == {"manifest"}
    assert len(world.manifest.builds) == 1
    assert await world.index.list_collections("hpdocs_chunks_") == [first.collection]


async def test_force_rebuilds_an_unchanged_index_with_the_same_chunk_ids(world: World) -> None:
    first = await world.run()
    world.later()

    second = await world.run(force=True)

    assert (second.skipped, second.reason) == (False, None)
    assert second.collection != first.collection
    assert await world.index.alias_target(ALIAS) == second.collection
    assert await world.index.count(first.collection) == await world.index.count(ALIAS) == 5
    assert [b.status for b in world.manifest.builds] == [BuildStatus.SUCCEEDED] * 2
    assert "manifest" not in {kind for kind, _ in world.calls.log}


async def test_a_changed_document_hash_triggers_a_rebuild(world: World) -> None:
    await world.run()
    world.later()
    changed = (DOCUMENTS[0], replace(DOCUMENTS[1], sha256="9" * 64))

    result = await ingest(changed, PINS, CONFIG, world.ports(), force=False)

    assert (result.skipped, result.reason) == (False, StaleReason.DOCUMENTS)
    assert len(world.manifest.builds) == 2


async def test_a_changed_pin_triggers_a_rebuild(world: World) -> None:
    await world.run()
    world.later()
    pins = {**PINS, DENSE.name: replace(DENSE, revision="1" * 40)}

    result = await ingest(DOCUMENTS, pins, CONFIG, world.ports(), force=False)

    assert result.reason is StaleReason.DENSE_PIN


async def test_a_live_alias_without_a_succeeded_manifest_row_triggers_a_rebuild(
    world: World,
) -> None:
    await world.run()
    world.manifest.builds.clear()
    world.later()

    result = await world.run()

    assert result.reason is StaleReason.NO_MANIFEST


async def test_a_live_collection_with_missing_points_triggers_a_rebuild(
    world: World, client: AsyncQdrantClient
) -> None:
    first = await world.run()
    one_point = make_batch(1, doc_id="demo-guide").chunks[0].chunk_id
    await client.delete(first.collection, points_selector=[one_point])
    world.later()

    result = await world.run()

    assert result.reason is StaleReason.POINT_COUNT


# ------------------------------------------------------------------ model failures


@pytest.mark.parametrize(
    "error",
    [
        ModelIntegrityError("demo/dense", "model.onnx", expected="c" * 64, actual="d" * 64),
        ModelDownloadError("demo/dense: cannot download", model="demo/dense"),
    ],
)
async def test_a_model_failure_stops_before_any_build_exists(
    world: World, error: Exception
) -> None:
    world.models.error = error

    with pytest.raises(type(error)):
        await world.run()

    assert world.calls.of("models") == ["demo/dense"]
    assert {kind for kind, _ in world.calls.log} <= {"models", "manifest"}
    assert world.manifest.builds == []
    assert await world.index.list_collections("") == []


async def test_models_are_verified_before_the_corpus_is_fetched(world: World) -> None:
    await world.run()

    log = world.calls.log
    assert log.index(("models", SPARSE.name)) < log.index(("corpus", "demo-guide"))
    assert log.index(("build", "processor")) < log.index(("corpus", "demo-guide"))
    assert log.index(("process", "demo-manual")) < log.index(("build", "embedder"))


# ------------------------------------------------------------------ failed builds


async def test_a_failure_after_the_build_started_is_recorded_as_failed(world: World) -> None:
    live = await world.run()
    world.later()
    world.corpus.failing = {"demo-manual": CorpusDownloadError("demo-manual: HTTP 503")}

    with pytest.raises(CorpusDownloadError):
        await world.run(force=True)

    failed = world.manifest.builds[-1]
    assert failed.status is BuildStatus.FAILED
    assert failed.error_type == "CorpusDownloadError"
    assert failed.build_id == "20261112t143105z"
    assert dict(failed.chunk_counts) == {"demo-guide": 3}  # stopped at the second document
    assert await world.index.alias_target(ALIAS) == live.collection
    assert await world.index.list_collections("hpdocs_chunks_") == [live.collection]


class LosesPoints(QdrantIndex):
    """Reports no points for collections created after `live` (a failed upload)."""

    def __init__(self, client: AsyncQdrantClient, live: str) -> None:
        super().__init__(client)
        self.live = live

    async def count(self, name: str) -> int:
        return await super().count(name) if name in (ALIAS, self.live) else 0


async def test_a_failed_publish_is_recorded_and_the_live_index_stays(
    world: World, client: AsyncQdrantClient
) -> None:
    live = await world.run()
    world.later()
    world.index = LosesPoints(client, live.collection)

    with pytest.raises(IndexPublishError):
        await world.run(force=True)

    failed = world.manifest.builds[-1]
    assert (failed.status, failed.error_type) == (BuildStatus.FAILED, "IndexPublishError")
    assert dict(failed.chunk_counts) == COUNTS
    assert await world.index.alias_target(ALIAS) == live.collection


async def test_an_embedder_that_returns_too_few_vectors_fails_the_build(world: World) -> None:
    world.embedder.short_by = 1

    with pytest.raises(ValueError, match="one dense and one sparse vector per chunk"):
        await world.run()

    assert world.manifest.builds[-1].error_type == "ValueError"
    assert await world.index.alias_target(ALIAS) is None


# ------------------------------------------------------------------ pruning


async def test_a_prune_failure_is_a_warning_and_the_run_still_succeeds(
    world: World, client: AsyncQdrantClient
) -> None:
    first = await world.run()
    world.later()
    await world.run(force=True)
    world.index = FailingDelete(client, failing={first.collection})
    world.later()

    result = await world.run(force=True, keep_previous=False)

    warning = BuildWarning(BuildWarningKind.PRUNE_FAILED, first.collection, "ConnectionError")
    assert result.warnings == (warning,)
    latest = world.manifest.builds[-1]
    assert (latest.status, latest.warnings) == (BuildStatus.SUCCEEDED, (warning,))
    assert await world.index.alias_target(ALIAS) == result.collection


# ------------------------------------------------------------------ logging


async def test_logs_name_documents_sizes_and_outcomes_never_content(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="app.ingestion")

    await world.run()
    await world.run()

    messages = [r.getMessage() for r in caplog.records]
    assert messages == [
        "index build needed",
        "index build started",
        "document chunked",
        "document chunked",
        "index build published",
        "index is current; build skipped",
    ]
    chunked = [r.__dict__ for r in caplog.records if r.getMessage() == "document chunked"]
    assert [(r["doc_id"], r["pages"], r["chunks"]) for r in chunked] == [
        ("demo-guide", 4, 3),
        ("demo-manual", 4, 2),
    ]
    assert chunked[0]["bytes"] == len(b"%PDF-1.7 demo-guide")
    assert caplog.records[0].__dict__["reason"] == "no_live_index"
    logged = " ".join(str(value) for r in caplog.records for value in r.__dict__.values())
    assert "demo button" not in logged
    assert "Step 0" not in logged

"""The benchmark's real runtime: index resolution, the build check, models and chunks.

Qdrant is qdrant-client's local mode, the manifest is SQLite and the models are
stand-ins, so the wiring is checked without servers or ONNX.
"""

import pathlib
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Executor
from contextlib import AbstractAsyncContextManager
from dataclasses import replace

import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.settings import Settings
from app.domain.ingestion import Bm25Params, IndexBuild, ModelFiles
from app.evaluation.cli import EvaluationError, Failure, Runtime, open_runtime
from app.infra.db import create_session_factory
from app.infra.manifest_repository import SqlManifestStore
from app.infra.orm import Base
from app.infra.qdrant_index import QdrantIndex
from app.infra.query_models import WARM_UP_DOCUMENT, WARM_UP_QUERY, ModelNames, QueryModels
from app.services.retrieval import RetrievalService
from tests.index_data import DENSE_SIZE, make_batch
from tests.manifest_data import make_build
from tests.model_data import pin
from tests.unit.conftest import SettingsFactory
from tests.unit.model_fakes import Builds, FakeSource

pytestmark = pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect in the local Qdrant:UserWarning"
)

ALIAS = "unit_chunks"
BUILD_ID = "20261113t100000z"
COLLECTION = f"unit_chunks_v1_{BUILD_ID}"
DENSE_FILES = pin(name="demo/dense", source="demo/dense-src")
SPARSE_FILES = pin(name="demo/bm25", source="demo/bm25-src")
BUILD = make_build(BUILD_ID, collection=COLLECTION, dense=DENSE_FILES, sparse=SPARSE_FILES)
PINS = {"demo/dense": DENSE_FILES, "demo/rerank-a": pin(name="demo/rerank-a")}


@pytest.fixture
def settings(make_settings: SettingsFactory) -> Settings:
    return make_settings(qdrant_collection_alias=ALIAS, reranker_model="demo/rerank-a")


@pytest.fixture
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(location=":memory:")
    index = QdrantIndex(client)
    await index.create_collection(COLLECTION, DENSE_SIZE)
    batch = make_batch(3)
    await index.upsert(COLLECTION, batch.chunks, batch.dense, batch.sparse)
    yield client
    await client.close()


@pytest.fixture
async def engine(tmp_path: pathlib.Path) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'manifest.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


async def record(engine: AsyncEngine, build: IndexBuild = BUILD) -> None:
    await SqlManifestStore(create_session_factory(engine)).record(build)


def models_factory(
    source: FakeSource | None = None, builds: Builds | None = None
) -> Callable[[Settings, Executor], QueryModels]:
    def factory(settings: Settings, executor: Executor) -> QueryModels:
        made = builds or Builds()
        return QueryModels(
            source or FakeSource(),
            ModelNames("demo/dense", "demo/bm25", settings.reranker_model),
            executor,
            build_embedder=made.embedder,
            build_reranker=made.reranker,
            warm_up_candidates=settings.retrieval_rerank_candidates,
        )

    return factory


def opened(
    settings: Settings,
    qdrant: AsyncQdrantClient,
    engine: AsyncEngine,
    *,
    collection: str | None = None,
    source: FakeSource | None = None,
    pins: dict[str, ModelFiles] | None = None,
) -> AbstractAsyncContextManager[Runtime]:
    return open_runtime(
        settings,
        collection,
        qdrant_factory=lambda _: qdrant,
        engine_factory=lambda _: engine,
        models_factory=models_factory(source),
        pins=lambda: PINS if pins is None else pins,
    )


async def test_the_alias_resolves_to_its_build_with_chunks_and_models(
    settings: Settings, qdrant: AsyncQdrantClient, engine: AsyncEngine
) -> None:
    await QdrantIndex(qdrant).point_alias(ALIAS, COLLECTION)
    await record(engine)

    async with opened(settings, qdrant, engine) as runtime:
        assert runtime.index.alias == ALIAS
        assert runtime.index.collection == COLLECTION
        assert runtime.index.build_id == BUILD_ID
        assert runtime.index.chunk_count == BUILD.chunk_count
        assert runtime.index.chunking_version == "v1"
        # A build recorded before its BM25 params: fastembed's defaults.
        assert (runtime.index.bm25_avg_len, runtime.index.bm25_avg_len_auto) == (256.0, False)
        chunks = await runtime.chunks()
        assert sorted(c.chunk_id for c in chunks) == sorted(
            c.chunk_id for c in make_batch(3).chunks
        )
        retriever = await runtime.retriever("demo/rerank-b")
        assert isinstance(retriever, RetrievalService)
        assert retriever.reranker_model == "demo/rerank-b"
        assert runtime.revisions == {name: files.revision for name, files in PINS.items()}


async def test_a_second_reranker_is_warmed_at_full_length_when_its_retriever_is_built(
    settings: Settings, qdrant: AsyncQdrantClient, engine: AsyncEngine
) -> None:
    """The lifespan warms only the configured reranker; the benchmark's other is warmed
    on load, with the same full-length batch, before any retrieval can use it."""
    await QdrantIndex(qdrant).point_alias(ALIAS, COLLECTION)
    await record(engine)
    builds = Builds()
    context = open_runtime(
        settings,
        None,
        qdrant_factory=lambda _: qdrant,
        engine_factory=lambda _: engine,
        models_factory=models_factory(builds=builds),
        pins=lambda: PINS,
    )

    async with context as runtime:
        await runtime.retriever("demo/rerank-b")

    [second] = [r for r in builds.rerankers if r.model == "demo/rerank-b"]
    candidates = settings.retrieval_rerank_candidates
    assert second.calls == [(WARM_UP_QUERY, [WARM_UP_DOCUMENT] * candidates)]


async def test_an_explicit_collection_is_read_without_the_alias(
    settings: Settings, qdrant: AsyncQdrantClient, engine: AsyncEngine
) -> None:
    await record(engine)

    async with opened(settings, qdrant, engine, collection=COLLECTION) as runtime:
        assert runtime.index.alias is None
        assert runtime.index.collection == COLLECTION


async def test_the_index_info_names_the_builds_bm25_avg_len(
    settings: Settings, qdrant: AsyncQdrantClient, engine: AsyncEngine
) -> None:
    """The avg_len ablation is a run on another collection; its header must say so."""
    await record(engine, replace(BUILD, sparse_params=Bm25Params(107.25, avg_len_auto=True)))

    async with opened(settings, qdrant, engine, collection=COLLECTION) as runtime:
        assert (runtime.index.bm25_avg_len, runtime.index.bm25_avg_len_auto) == (107.25, True)


async def failure(context: AbstractAsyncContextManager[Runtime]) -> Failure:
    with pytest.raises(EvaluationError) as caught:
        async with context:
            pass
    return caught.value.failure


async def test_no_alias_is_no_live_index(
    settings: Settings, qdrant: AsyncQdrantClient, engine: AsyncEngine
) -> None:
    assert await failure(opened(settings, qdrant, engine)) is Failure.NO_LIVE_INDEX


async def test_a_collection_without_a_succeeded_build_is_refused(
    settings: Settings, qdrant: AsyncQdrantClient, engine: AsyncEngine
) -> None:
    result = await failure(opened(settings, qdrant, engine, collection=COLLECTION))

    assert result is Failure.NO_SUCCEEDED_BUILD


async def test_models_that_fail_to_load_stop_the_run(
    settings: Settings, qdrant: AsyncQdrantClient, engine: AsyncEngine
) -> None:
    await record(engine)
    broken = FakeSource(errors={"demo/dense": OSError("disk")})

    result = await failure(opened(settings, qdrant, engine, collection=COLLECTION, source=broken))

    assert result is Failure.MODELS_FAILED


async def test_an_index_built_with_other_model_files_is_refused(
    settings: Settings, qdrant: AsyncQdrantClient, engine: AsyncEngine
) -> None:
    await record(engine, make_build(BUILD_ID, collection=COLLECTION, dense=pin(name="x/other")))

    result = await failure(opened(settings, qdrant, engine, collection=COLLECTION))

    assert result is Failure.MODEL_FILES_MISMATCH

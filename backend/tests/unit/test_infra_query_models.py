"""Loading the query models in the background, and the readiness probe over them."""

import asyncio
import json
from collections.abc import Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path

import pytest

from app.core.settings import Settings
from app.domain.health import DependencyUnavailableError, FailureReason
from app.domain.ingestion import (
    ModelIntegrityError,
    ModelNotAllowedError,
    VerifiedModel,
)
from app.domain.retrieval import ModelsNotReadyError, QueryEmbedder, Reranker
from app.infra.inference import create_inference_executor
from app.infra.query_models import ModelNames, ModelsProbe, ModelsState, QueryModels
from tests.unit.conftest import LogReader, SettingsFactory
from tests.unit.model_fakes import Builds, FakeSource, StandInReranker

CANARY = "CANARY-digest-0f0f"
NAMES = ModelNames(dense="demo/dense", sparse="demo/bm25", reranker="demo/rerank-a")


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    pool = create_inference_executor(1)
    yield pool
    pool.shutdown()


def models(
    source: FakeSource, builds: Builds, executor: ThreadPoolExecutor, candidates: int = 20
) -> QueryModels:
    return QueryModels(
        source,
        NAMES,
        executor,
        build_embedder=builds.embedder,
        build_reranker=builds.reranker,
        warm_up_candidates=candidates,
    )


# Every word is at least one WordPiece token, so this many words reach the models'
# 512-token input; the model tests count the real tokens (tests/embeddings).
TOKEN_LIMIT = 512


# ------------------------------------------------------------------ loading


async def test_models_are_loading_until_load_ends(executor: ThreadPoolExecutor) -> None:
    subject = models(FakeSource(), Builds(), executor)

    assert subject.state is ModelsState.LOADING
    with pytest.raises(ModelsNotReadyError):
        subject.query_embedder()
    with pytest.raises(ModelsNotReadyError):
        await subject.reranker()


async def test_load_verifies_builds_off_the_loop_and_warms_each_model(
    executor: ThreadPoolExecutor,
) -> None:
    source, builds = FakeSource(), Builds()
    subject = models(source, builds, executor)

    await subject.load()

    assert subject.state is ModelsState.READY
    assert source.fetched == ["demo/dense", "demo/bm25", "demo/rerank-a"]
    [embedder], [reranker] = builds.embedders, builds.rerankers
    assert embedder.built_on.startswith("inference")
    assert reranker.built_on.startswith("inference")
    assert (embedder.dense.files.name, embedder.sparse.files.name) == ("demo/dense", "demo/bm25")
    assert len(embedder.queries) == 2  # one dense and one sparse warm-up call
    assert len(reranker.calls) == 1
    assert subject.query_embedder() is embedder
    assert await subject.reranker() is reranker


async def test_a_second_reranker_is_loaded_only_when_selected(executor: ThreadPoolExecutor) -> None:
    source, builds = FakeSource(), Builds()
    subject = models(source, builds, executor)
    await subject.load()

    assert [r.model for r in builds.rerankers] == ["demo/rerank-a"]

    second = await subject.reranker("demo/rerank-b")
    again = await subject.reranker("demo/rerank-b")

    assert second is again
    assert isinstance(second, StandInReranker)
    assert second.model == "demo/rerank-b"
    assert [r.model for r in builds.rerankers] == ["demo/rerank-a", "demo/rerank-b"]
    assert source.fetched.count("demo/rerank-b") == 1
    assert len(second.calls) == 1  # warmed before first use
    assert await subject.reranker("demo/rerank-a") is builds.rerankers[0]


async def test_an_unpinned_reranker_cannot_be_selected(executor: ThreadPoolExecutor) -> None:
    subject = models(FakeSource(), Builds(), executor)
    await subject.load()

    with pytest.raises(ModelNotAllowedError):
        await subject.reranker("demo/unknown")


async def test_the_loaded_models_are_named_for_readiness(executor: ThreadPoolExecutor) -> None:
    subject = models(FakeSource(), Builds(), executor)
    await subject.load()

    assert subject.describe() == {
        "dense": "demo/dense",
        "sparse": "demo/bm25",
        "reranker": "demo/rerank-a",
    }


# ------------------------------------------------------------------ failures


async def test_a_verification_failure_leaves_the_models_failed_with_its_type_only(
    executor: ThreadPoolExecutor, json_logs: LogReader
) -> None:
    error = ModelIntegrityError("demo/dense", "model.onnx", expected=CANARY, actual="0" * 64)
    subject = models(FakeSource(errors={"demo/dense": error}), Builds(), executor)

    await subject.load()  # never raises: it runs as a background task

    assert subject.state is ModelsState.FAILED
    assert subject.error_type == "ModelIntegrityError"
    with pytest.raises(ModelsNotReadyError):
        subject.query_embedder()
    lines = json_logs()
    assert CANARY not in json.dumps(lines)
    (failed,) = [line for line in lines if line["message"] == "query models failed to load"]
    assert failed["error_type"] == "ModelIntegrityError"


async def test_a_build_failure_leaves_the_models_failed(executor: ThreadPoolExecutor) -> None:
    subject = models(FakeSource(), Builds(error=RuntimeError(CANARY)), executor)

    await subject.load()

    assert (subject.state, subject.error_type) == (ModelsState.FAILED, "RuntimeError")


async def test_models_from_settings_never_download(
    executor: ThreadPoolExecutor, make_settings: SettingsFactory, tmp_path: Path
) -> None:
    """An empty store fails to verify instead of reaching the hub (ADR 0008)."""
    settings = make_settings(embedding_cache_dir=tmp_path, reranker_model="demo/none")
    subject = QueryModels.from_settings(settings, executor)

    await subject.load()

    assert subject.state is ModelsState.FAILED
    assert subject.error_type == "ModelDownloadError"
    assert subject.names == ModelNames("BAAI/bge-small-en-v1.5", "Qdrant/bm25", "demo/none")


# ------------------------------------------------------------------ full-length warm-up


async def test_each_reranker_is_warmed_with_a_full_batch_of_full_length_pairs(
    executor: ThreadPoolExecutor,
) -> None:
    """ONNX sizes its buffers on first use: warm with the production shape (commit 10c)."""
    builds = Builds()
    subject = models(FakeSource(), builds, executor, candidates=7)

    await subject.load()
    await subject.reranker("demo/rerank-b")  # loaded on demand (the benchmark's ablation)

    for reranker in builds.rerankers:
        [(query, texts)] = reranker.calls
        assert len(texts) == 7
        assert all(len(text.split()) > TOKEN_LIMIT for text in texts)
        assert 0 < len(query.split()) < TOKEN_LIMIT  # a pair: query and document both


async def test_the_query_embedder_is_warmed_with_full_length_text(
    executor: ThreadPoolExecutor,
) -> None:
    builds = Builds()

    await models(FakeSource(), builds, executor).load()

    [embedder] = builds.embedders
    dense = [q for q in embedder.queries if q.startswith("dense:")]
    assert [len(q.removeprefix("dense:").split()) > TOKEN_LIMIT for q in dense] == [True]


async def test_models_stay_loading_until_the_warm_up_ends(
    executor: ThreadPoolExecutor,
) -> None:
    gate = asyncio.Event()
    builds = Builds(gate=gate)
    subject = models(FakeSource(), builds, executor)
    probe = ModelsProbe(subject)
    loading = asyncio.create_task(subject.load())
    await builds.started.wait()  # the warm-up call is running, held by the gate

    during = subject.state
    with pytest.raises(DependencyUnavailableError) as before:
        await probe.ping()
    gate.set()
    await loading

    assert (during, before.value.reason) == (ModelsState.LOADING, FailureReason.LOADING)
    assert subject.state is ModelsState.READY
    assert await probe.ping() == subject.describe()


async def test_ready_logs_the_warm_up_time(
    executor: ThreadPoolExecutor, json_logs: LogReader
) -> None:
    await models(FakeSource(), Builds(), executor).load()

    [ready] = [r for r in json_logs() if r["message"] == "query models ready"]
    assert isinstance(ready["warm_up_seconds"], float)
    assert ready["warm_up_seconds"] >= 0


# ------------------------------------------------------------------ probe


async def test_the_probe_reports_loading_then_ready_with_the_model_names(
    executor: ThreadPoolExecutor,
) -> None:
    subject = models(FakeSource(), Builds(), executor)
    probe = ModelsProbe(subject)

    with pytest.raises(DependencyUnavailableError) as loading:
        await probe.ping()
    await subject.load()

    assert probe.name == "models"
    assert loading.value.reason is FailureReason.LOADING
    assert await probe.ping() == subject.describe()


async def test_the_probe_reports_a_failed_load_without_its_message(
    executor: ThreadPoolExecutor,
) -> None:
    subject = models(FakeSource(errors={"demo/bm25": RuntimeError(CANARY)}), Builds(), executor)
    await subject.load()

    with pytest.raises(DependencyUnavailableError) as failed:
        await ModelsProbe(subject).ping()

    assert failed.value.reason is FailureReason.FAILED
    assert CANARY not in str(failed.value)


async def test_models_from_settings_build_the_adapters_with_the_settings_and_the_pool(
    executor: ThreadPoolExecutor, make_settings: SettingsFactory
) -> None:
    settings = make_settings(
        embedding_dense_model="demo/dense",
        embedding_sparse_model="demo/bm25",
        reranker_model="demo/rerank-a",
    )
    seen: list[tuple[object, ...]] = []
    builds = Builds()

    def embedder_factory(
        given: Settings, dense: VerifiedModel, sparse: VerifiedModel, pool: Executor
    ) -> QueryEmbedder:
        seen.append(("embedder", given, pool))
        return builds.embedder(dense, sparse)

    def reranker_factory(given: Settings, model: VerifiedModel, pool: Executor | None) -> Reranker:
        seen.append(("reranker", given, pool))
        return builds.reranker(model)

    subject = QueryModels.from_settings(
        settings,
        executor,
        source=FakeSource(),
        embedder_factory=embedder_factory,
        reranker_factory=reranker_factory,
    )
    await subject.load()

    assert subject.state is ModelsState.READY
    assert seen == [("embedder", settings, executor), ("reranker", settings, executor)]
    [(_, warm_up)] = builds.rerankers[0].calls
    assert len(warm_up) == settings.retrieval_rerank_candidates


async def test_the_verified_model_files_are_known_once_loaded(
    executor: ThreadPoolExecutor,
) -> None:
    """The index probe compares them with the files the index was built with."""
    subject = models(FakeSource(), Builds(), executor)
    with pytest.raises(ModelsNotReadyError):
        subject.verified_files()

    await subject.load()

    dense, sparse = subject.verified_files()
    assert (dense.name, dense.source, sparse.name) == ("demo/dense", "demo/dense-src", "demo/bm25")

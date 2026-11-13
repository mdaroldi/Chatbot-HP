"""The fastembed cross-encoder adapter with a stand-in model class (no ONNX session)."""

import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import ClassVar

from pydantic import SecretStr

from app.core.settings import Settings
from app.domain.ingestion import VerifiedModel
from app.infra.fastembed_reranker import FastembedReranker
from app.infra.inference import create_inference_executor
from tests.model_data import pin

VERIFIED = VerifiedModel(
    files=pin(name="demo/reranker", source="demo/reranker-onnx"), path=Path("/m/r")
)


class FakeCrossEncoder:
    built: ClassVar[list["FakeCrossEncoder"]] = []

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str,
        threads: int | None,
        specific_model_path: str,
        local_files_only: bool,
    ) -> None:
        self.model_name, self.cache_dir, self.threads = model_name, cache_dir, threads
        self.path, self.local_only = specific_model_path, local_files_only
        self.calls: list[tuple[str, list[str], int]] = []
        self.thread_names: list[str] = []
        self.built.append(self)

    def rerank(self, query: str, documents: Iterable[str], batch_size: int) -> Iterator[float]:
        texts = list(documents)
        self.calls.append((query, texts, batch_size))
        self.thread_names.append(threading.current_thread().name)
        for text in texts:
            yield len(text) - len(query) / 10


def reranker(threads: int = 2) -> FastembedReranker:
    FakeCrossEncoder.built = []
    return FastembedReranker(
        VERIFIED, Path("/models"), threads=threads, executor=None, model_class=FakeCrossEncoder
    )


async def test_scores_come_back_as_floats_in_input_order() -> None:
    scores = await reranker().rerank("abcdefghij", ["xx", "xxxxx", "x"])

    assert scores == [1.0, 4.0, 0.0]
    assert all(type(score) is float for score in scores)


async def test_all_candidates_go_in_one_batch_with_the_raw_query() -> None:
    subject = reranker()

    await subject.rerank("q", ["a", "b", "c"])

    assert FakeCrossEncoder.built[0].calls == [("q", ["a", "b", "c"], 3)]


async def test_no_candidates_means_no_model_call() -> None:
    subject = reranker()

    assert await subject.rerank("q", []) == []
    assert FakeCrossEncoder.built[0].calls == []


def test_the_model_is_built_once_offline_from_its_verified_folder() -> None:
    subject = reranker(threads=3)

    [model] = FakeCrossEncoder.built
    assert (model.model_name, model.path, model.local_only) == ("demo/reranker", "/m/r", True)
    assert (model.cache_dir, model.threads) == ("/models", 3)
    assert subject.model == "demo/reranker"


async def test_reranking_runs_on_the_given_executor() -> None:
    executor = create_inference_executor(1)
    FakeCrossEncoder.built = []
    try:
        subject = FastembedReranker(
            VERIFIED, Path("/models"), threads=1, executor=executor, model_class=FakeCrossEncoder
        )
        await subject.rerank("q", ["a"])
    finally:
        executor.shutdown()

    assert FakeCrossEncoder.built[0].thread_names[0].startswith("inference")


def test_from_settings_uses_the_model_store_and_inference_threads() -> None:
    FakeCrossEncoder.built = []
    settings = Settings(
        postgres_password=SecretStr("unused"),
        embedding_cache_dir=Path("/opt/models"),
        inference_onnx_threads=4,
    )

    FastembedReranker.from_settings(settings, VERIFIED, None, model_class=FakeCrossEncoder)

    [model] = FakeCrossEncoder.built
    assert (model.cache_dir, model.threads) == ("/opt/models", 4)

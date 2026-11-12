"""The fastembed adapters with fake model classes: no download, no ONNX session."""

import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import ClassVar

import numpy as np
from fastembed.common.types import NumpyArray
from fastembed.sparse.sparse_embedding_base import SparseEmbedding
from pydantic import SecretStr

from app.core.settings import Settings
from app.domain.ingestion import SparseVector, VerifiedModel
from app.infra.fastembed_embedder import FastembedEmbedder
from app.infra.fastembed_tokens import FastembedTokenCounter
from tests.model_data import pin


class FakeModel:
    """Records how it was built and called, like a fastembed model class."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str,
        lazy_load: bool,
        specific_model_path: str,
        local_files_only: bool,
    ) -> None:
        self.model_name, self.cache_dir, self.lazy_load = model_name, cache_dir, lazy_load
        self.path, self.local_only = specific_model_path, local_files_only
        self.batch_sizes: list[int] = []
        self.threads: list[int] = []
        self.built.append(self)

    built: ClassVar[list["FakeModel"]]

    def _record(self, batch_size: int) -> None:
        self.batch_sizes.append(batch_size)
        self.threads.append(threading.get_ident())


class FakeDense(FakeModel):
    built: ClassVar[list[FakeModel]] = []

    def embed(self, documents: Iterable[str], batch_size: int) -> Iterator[NumpyArray]:
        self._record(batch_size)
        for text in documents:
            yield np.array([len(text), 0.5], dtype=np.float32)

    def token_count(self, texts: str) -> int:
        return len(texts.split()) + 2  # like bge-small: [CLS] and [SEP] count


class FakeSparse(FakeModel):
    built: ClassVar[list[FakeModel]] = []

    def embed(self, documents: Iterable[str], batch_size: int) -> Iterator[SparseEmbedding]:
        self._record(batch_size)
        for text in documents:
            yield SparseEmbedding(
                values=np.array([1.5, float(len(text))]), indices=np.array([7, 3], dtype=np.int64)
            )


def verified(name: str, folder: str) -> VerifiedModel:
    return VerifiedModel(files=pin(name=name, source=f"{name}-repo"), path=Path(folder))


DENSE = verified("dense-model", "/cache/dense-snapshot")
SPARSE = verified("sparse-model", "/cache/sparse-snapshot")


def embedder(cache_dir: Path = Path("cache"), batch_size: int = 4) -> FastembedEmbedder:
    FakeDense.built, FakeSparse.built = [], []
    return FastembedEmbedder(
        DENSE,
        SPARSE,
        cache_dir,
        batch_size,
        dense_class=FakeDense,
        sparse_class=FakeSparse,
    )


def settings() -> Settings:
    return Settings(
        postgres_password=SecretStr("unused"),
        embedding_cache_dir=Path("/var/cache/models"),
        embedding_batch_size=16,
    )


# ------------------------------------------------------------------ embedder


async def test_dense_vectors_are_float_lists_in_input_order() -> None:
    vectors = await embedder().embed_dense(["ab", "abcd", "a"])

    assert vectors == [[2.0, 0.5], [4.0, 0.5], [1.0, 0.5]]
    assert all(type(value) is float for vector in vectors for value in vector)


async def test_sparse_vectors_become_domain_sparse_vectors_in_input_order() -> None:
    vectors = await embedder().embed_sparse(["ab", "abc"])

    assert vectors == [
        SparseVector(indices=(7, 3), values=(1.5, 2.0)),
        SparseVector(indices=(7, 3), values=(1.5, 3.0)),
    ]
    assert all(type(i) is int for v in vectors for i in v.indices)
    assert all(type(x) is float for v in vectors for x in v.values)


async def test_each_model_is_loaded_once_from_its_verified_folder_offline() -> None:
    subject = embedder(cache_dir=Path("/srv/models"))  # a name only, never created
    await subject.embed_dense(["a"])
    await subject.embed_dense(["b"])
    await subject.embed_sparse(["c"])

    [dense], [sparse] = FakeDense.built, FakeSparse.built
    built = [(m.model_name, m.path, m.local_only, m.lazy_load) for m in (dense, sparse)]
    assert built == [
        ("dense-model", "/cache/dense-snapshot", True, False),
        ("sparse-model", "/cache/sparse-snapshot", True, False),
    ]
    assert {dense.cache_dir, sparse.cache_dir} == {"/srv/models"}
    assert (subject.dense_model, subject.sparse_model) == ("dense-model", "sparse-model")


async def test_the_batch_size_reaches_both_models() -> None:
    subject = embedder(batch_size=3)
    await subject.embed_dense(["a", "b"])
    await subject.embed_sparse(["a"])

    assert (FakeDense.built[0].batch_sizes, FakeSparse.built[0].batch_sizes) == ([3], [3])


async def test_embedding_runs_off_the_event_loop_thread() -> None:
    subject = embedder()
    await subject.embed_dense(["a"])
    await subject.embed_sparse(["a"])

    loop_thread = threading.get_ident()
    assert FakeDense.built[0].threads + FakeSparse.built[0].threads
    assert loop_thread not in FakeDense.built[0].threads + FakeSparse.built[0].threads


async def test_embedder_from_settings_uses_the_embedding_settings() -> None:
    FakeDense.built, FakeSparse.built = [], []
    subject = FastembedEmbedder.from_settings(
        settings(), DENSE, SPARSE, dense_class=FakeDense, sparse_class=FakeSparse
    )
    await subject.embed_dense(["a"])

    [dense], [sparse] = FakeDense.built, FakeSparse.built
    assert (dense.path, dense.cache_dir) == ("/cache/dense-snapshot", "/var/cache/models")
    assert (sparse.path, sparse.cache_dir) == ("/cache/sparse-snapshot", "/var/cache/models")
    assert dense.batch_sizes == [16]


# ------------------------------------------------------------------ token counter


def test_tokens_are_counted_by_the_dense_tokenizer_from_the_verified_folder() -> None:
    FakeDense.built = []
    counter = FastembedTokenCounter(DENSE, Path("cache"), model_class=FakeDense)

    assert (counter.count("one two three"), counter.count("")) == (5, 2)
    [model] = FakeDense.built
    assert (model.model_name, model.path, model.local_only) == (
        "dense-model",
        "/cache/dense-snapshot",
        True,
    )
    assert (model.cache_dir, model.lazy_load) == ("cache", True)


def test_token_counter_from_settings_uses_the_cache_dir() -> None:
    FakeDense.built = []
    FastembedTokenCounter.from_settings(settings(), DENSE, model_class=FakeDense)

    [model] = FakeDense.built
    assert (model.path, model.cache_dir) == ("/cache/dense-snapshot", "/var/cache/models")

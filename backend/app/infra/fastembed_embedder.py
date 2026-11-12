"""Dense and sparse embeddings with local fastembed models on CPU (ADR 0003)."""

import asyncio
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol, Self

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.common.types import NumpyArray
from fastembed.sparse.sparse_embedding_base import SparseEmbedding

from app.core.settings import Settings
from app.domain.ingestion import SparseVector


class DenseModel(Protocol):
    def embed(self, documents: Iterable[str], batch_size: int) -> Iterable[NumpyArray]: ...


class SparseModel(Protocol):
    def embed(self, documents: Iterable[str], batch_size: int) -> Iterable[SparseEmbedding]: ...


class DenseModelClass(Protocol):
    def __call__(self, model_name: str, *, cache_dir: str, lazy_load: bool) -> DenseModel: ...


class SparseModelClass(Protocol):
    def __call__(self, model_name: str, *, cache_dir: str, lazy_load: bool) -> SparseModel: ...


class FastembedEmbedder:
    """Embedder over fastembed's ONNX models.

    The models are loaded (and downloaded into the cache on first use) when the
    embedder is built, so a missing or broken model fails before any work starts;
    build it only when an index is really built. Inference is CPU-bound and runs in
    a worker thread so it never blocks the event loop.
    """

    def __init__(  # noqa: PLR0913 - four settings plus the two injectable model classes
        self,
        dense_model: str,
        sparse_model: str,
        cache_dir: Path,
        batch_size: int,
        *,
        dense_class: DenseModelClass = TextEmbedding,
        sparse_class: SparseModelClass = SparseTextEmbedding,
    ) -> None:
        self._dense_name, self._sparse_name = dense_model, sparse_model
        self._batch_size = batch_size
        self._dense = dense_class(dense_model, cache_dir=str(cache_dir), lazy_load=False)
        self._sparse = sparse_class(sparse_model, cache_dir=str(cache_dir), lazy_load=False)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        dense_class: DenseModelClass = TextEmbedding,
        sparse_class: SparseModelClass = SparseTextEmbedding,
    ) -> Self:
        return cls(
            settings.embedding_dense_model,
            settings.embedding_sparse_model,
            settings.embedding_cache_dir,
            settings.embedding_batch_size,
            dense_class=dense_class,
            sparse_class=sparse_class,
        )

    @property
    def dense_model(self) -> str:
        return self._dense_name

    @property
    def sparse_model(self) -> str:
        return self._sparse_name

    async def embed_dense(self, texts: Sequence[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_dense, list(texts))

    async def embed_sparse(self, texts: Sequence[str]) -> list[SparseVector]:
        return await asyncio.to_thread(self._embed_sparse, list(texts))

    def _embed_dense(self, texts: list[str]) -> list[list[float]]:
        vectors = self._dense.embed(texts, batch_size=self._batch_size)
        return [[float(value) for value in vector] for vector in vectors]

    def _embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(
                indices=tuple(int(index) for index in vector.indices),
                values=tuple(float(value) for value in vector.values),
            )
            for vector in self._sparse.embed(texts, batch_size=self._batch_size)
        ]

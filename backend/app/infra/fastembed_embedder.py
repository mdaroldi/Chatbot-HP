"""Dense and sparse embeddings with local fastembed models on CPU (ADR 0003)."""

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Executor
from pathlib import Path
from typing import Protocol, Self

import onnxruntime
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.common.types import NumpyArray
from fastembed.sparse.sparse_embedding_base import SparseEmbedding

from app.core.settings import Settings
from app.domain.ingestion import FASTEMBED_BM25, Bm25Params, SparseVector, VerifiedModel
from app.infra.inference import run_inference


class DenseModel(Protocol):
    def embed(self, documents: Iterable[str], batch_size: int) -> Iterable[NumpyArray]: ...

    def query_embed(self, query: str) -> Iterable[NumpyArray]: ...


class SparseModel(Protocol):
    def embed(self, documents: Iterable[str], batch_size: int) -> Iterable[SparseEmbedding]: ...

    def query_embed(self, query: str) -> Iterable[SparseEmbedding]: ...


class DenseModelClass(Protocol):
    def __call__(  # noqa: PLR0913 - mirrors fastembed's constructor keywords
        self,
        model_name: str,
        *,
        cache_dir: str,
        lazy_load: bool,
        specific_model_path: str,
        local_files_only: bool,
        threads: int | None,
    ) -> DenseModel: ...


class SparseModelClass(Protocol):
    def __call__(  # noqa: PLR0913 - mirrors fastembed's constructor keywords
        self,
        model_name: str,
        *,
        cache_dir: str,
        lazy_load: bool,
        specific_model_path: str,
        local_files_only: bool,
        threads: int | None,
        k: float,
        b: float,
        avg_len: float,
    ) -> SparseModel: ...


class FastembedEmbedder:
    """Embedder over fastembed's ONNX models, loaded from verified folders (ADR 0006).

    The models are loaded when the embedder is built, from folders whose files were
    just hashed, and never from the network (`local_files_only`). Ingestion builds one
    per index build and leaves the thread count to onnxruntime; the API builds one with
    `for_queries`, on the inference pool with explicit ONNX threads (ADR 0008).
    Inference is CPU-bound and never runs on the event loop. onnxruntime's telemetry is
    switched off before the first session; ORT_DISABLE_TELEMETRY in compose also keeps
    it from starting. The BM25 parameters are always passed, never left to fastembed's
    defaults; the query side keeps those defaults, which cannot change a unit-weight query.
    """

    def __init__(  # noqa: PLR0913 - two models, two settings, BM25, injectable classes, switch
        self,
        dense: VerifiedModel,
        sparse: VerifiedModel,
        cache_dir: Path,
        batch_size: int,
        *,
        threads: int | None = None,
        executor: Executor | None = None,
        bm25: Bm25Params = FASTEMBED_BM25,
        dense_class: DenseModelClass = TextEmbedding,
        sparse_class: SparseModelClass = SparseTextEmbedding,
        disable_telemetry: Callable[[], None] = onnxruntime.disable_telemetry_events,
    ) -> None:
        disable_telemetry()
        self._dense_name, self._sparse_name = dense.files.name, sparse.files.name
        self._batch_size = batch_size
        self._dense = dense_class(
            dense.files.name,
            cache_dir=str(cache_dir),
            lazy_load=False,
            specific_model_path=str(dense.path),
            local_files_only=True,
            threads=threads,
        )
        self._sparse = sparse_class(
            sparse.files.name,
            cache_dir=str(cache_dir),
            lazy_load=False,
            specific_model_path=str(sparse.path),
            local_files_only=True,
            threads=threads,
            k=bm25.k,
            b=bm25.b,
            avg_len=bm25.avg_len,
        )
        self._executor = executor

    @classmethod
    def from_settings(  # noqa: PLR0913 - settings, two models, BM25, two injectable classes
        cls,
        settings: Settings,
        dense: VerifiedModel,
        sparse: VerifiedModel,
        *,
        bm25: Bm25Params,
        dense_class: DenseModelClass = TextEmbedding,
        sparse_class: SparseModelClass = SparseTextEmbedding,
    ) -> Self:
        """Ingestion's embedder: `bm25` is the build's own, fixed or measured (ADR 0009)."""
        return cls(
            dense,
            sparse,
            settings.embedding_cache_dir,
            settings.embedding_batch_size,
            bm25=bm25,
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
        batch = list(texts)
        return await run_inference(self._executor, lambda: self._embed_dense(batch))

    async def embed_sparse(self, texts: Sequence[str]) -> list[SparseVector]:
        batch = list(texts)
        return await run_inference(self._executor, lambda: self._embed_sparse(batch))

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

    @classmethod
    def for_queries(  # noqa: PLR0913 - settings, three collaborators, two injectable classes
        cls,
        settings: Settings,
        dense: VerifiedModel,
        sparse: VerifiedModel,
        executor: Executor,
        *,
        dense_class: DenseModelClass = TextEmbedding,
        sparse_class: SparseModelClass = SparseTextEmbedding,
    ) -> Self:
        """The API's query encoder: the same models as ingestion, on the inference pool."""
        return cls(
            dense,
            sparse,
            settings.embedding_cache_dir,
            settings.embedding_batch_size,
            threads=settings.inference_onnx_threads,
            executor=executor,
            dense_class=dense_class,
            sparse_class=sparse_class,
        )

    async def embed_query_dense(self, text: str) -> list[float]:
        """bge-small's query path; fastembed adds no instruction for it (ADR 0009)."""
        return await run_inference(self._executor, lambda: self._query_dense(text))

    async def embed_query_sparse(self, text: str) -> SparseVector:
        """BM25's query path: unit weights; the collection applies IDF (ADR 0002)."""
        return await run_inference(self._executor, lambda: self._query_sparse(text))

    def _query_dense(self, text: str) -> list[float]:
        (vector,) = self._dense.query_embed(text)
        return [float(value) for value in vector]

    def _query_sparse(self, text: str) -> SparseVector:
        (vector,) = self._sparse.query_embed(text)
        return SparseVector(
            indices=tuple(int(index) for index in vector.indices),
            values=tuple(float(value) for value in vector.values),
        )

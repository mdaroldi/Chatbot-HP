"""The API's query models, loaded in the background at startup (ADR 0008).

Loading verifies each model's pinned files, builds one instance per model on the
inference pool (an ONNX session takes a moment to build) and warms each with one call
at full length and, for a reranker, with a full batch, so the first request does not
pay for it. Until then the models are `loading`; a
failure leaves them `failed` with its exception type, never its message, which holds
digests and paths. The API reads only the baked store (no downloads), and a second
reranker is loaded only when one is asked for by name (the benchmark's ablation).
"""

import asyncio
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

from app.core.settings import Settings
from app.domain.health import DependencyUnavailableError, FailureReason
from app.domain.ingestion import ModelFiles, ModelSource, VerifiedModel
from app.domain.retrieval import ModelsNotReadyError, QueryEmbedder, Reranker
from app.infra.fastembed_embedder import FastembedEmbedder
from app.infra.fastembed_reranker import FastembedReranker
from app.infra.inference import run_inference
from app.infra.model_files import PinnedModelSource, load_pins

logger = logging.getLogger(__name__)

# Full-length warm-up (Phase 2, commit 10c). onnxruntime sizes its buffers for the
# shapes it has seen, so a two-word warm-up left the first real rerank (20 pairs cut
# at 512 tokens) to pay for it: 10.1 s against about 2 s steady. Every word is at least
# one token, so 600 words reach the 512-token input of the dense model and both
# rerankers (counted with the real tokenizers in tests/embeddings).
WARM_UP_WORDS = 600
WARM_UP_QUERY_WORDS = 32  # a long question; the pair still fills 512 tokens
WARM_UP_DOCUMENT = " ".join(["warm"] * WARM_UP_WORDS)
WARM_UP_QUERY = " ".join(["warm"] * WARM_UP_QUERY_WORDS)

EmbedderFactory = Callable[[Settings, VerifiedModel, VerifiedModel, Executor], QueryEmbedder]
RerankerFactory = Callable[[Settings, VerifiedModel, Executor | None], Reranker]


class ModelsState(StrEnum):
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class ModelNames:
    dense: str
    sparse: str
    reranker: str


class _NoDownloads:
    """HubClient that refuses: the API reads the baked store only (ADR 0008)."""

    def download(  # the HubClient signature; every argument is unused by design
        self,
        source: str,  # noqa: ARG002
        revision: str,  # noqa: ARG002
        files: Sequence[str],  # noqa: ARG002
        cache_dir: Path,  # noqa: ARG002
        *,
        force: bool,  # noqa: ARG002
    ) -> Path:
        raise RuntimeError("model downloads are disabled in the API")


class QueryModels:
    def __init__(  # noqa: PLR0913 - source, names and pool, two builders, warm-up batch
        self,
        source: ModelSource,
        names: ModelNames,
        executor: Executor,
        *,
        build_embedder: Callable[[VerifiedModel, VerifiedModel], QueryEmbedder],
        build_reranker: Callable[[VerifiedModel], Reranker],
        warm_up_candidates: int,
    ) -> None:
        self.names = names
        self.state = ModelsState.LOADING
        self.error_type: str | None = None
        self._source, self._executor = source, executor
        self._build_embedder, self._build_reranker = build_embedder, build_reranker
        self._embedder: QueryEmbedder | None = None
        self._verified: tuple[ModelFiles, ModelFiles] | None = None
        self._rerankers: dict[str, Reranker] = {}
        self._reranker_lock = asyncio.Lock()
        self._warm_up_batch = [WARM_UP_DOCUMENT] * warm_up_candidates

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        executor: Executor,
        *,
        source: ModelSource | None = None,
        embedder_factory: EmbedderFactory = FastembedEmbedder.for_queries,
        reranker_factory: RerankerFactory = FastembedReranker.from_settings,
    ) -> Self:
        """The API's models: pinned files from the baked store, no downloads."""
        store = source or PinnedModelSource(
            load_pins(), settings.embedding_cache_dir, _NoDownloads()
        )

        def embedder(dense: VerifiedModel, sparse: VerifiedModel) -> QueryEmbedder:
            return embedder_factory(settings, dense, sparse, executor)

        def reranker(model: VerifiedModel) -> Reranker:
            return reranker_factory(settings, model, executor)

        names = ModelNames(
            settings.embedding_dense_model,
            settings.embedding_sparse_model,
            settings.reranker_model,
        )
        return cls(
            store,
            names,
            executor,
            build_embedder=embedder,
            build_reranker=reranker,
            warm_up_candidates=settings.retrieval_rerank_candidates,
        )

    async def load(self) -> None:
        """Load every model; runs as a background task, so it records and never raises."""
        try:
            dense = await self._source.fetch(self.names.dense)
            sparse = await self._source.fetch(self.names.sparse)
            embedder = await run_inference(
                self._executor, lambda: self._build_embedder(dense, sparse)
            )
            start = time.perf_counter()
            await embedder.embed_query_dense(WARM_UP_DOCUMENT)
            await embedder.embed_query_sparse(WARM_UP_DOCUMENT)
            await self._load_reranker(self.names.reranker)
            warm_up_seconds = round(time.perf_counter() - start, 3)
        except Exception as exc:
            self.state, self.error_type = ModelsState.FAILED, type(exc).__name__
            logger.error(  # noqa: TRY400 - the traceback would carry the message
                "query models failed to load", extra={"error_type": self.error_type}
            )
            return
        self._embedder, self._verified = embedder, (dense.files, sparse.files)
        self.state = ModelsState.READY
        # Model building is in the time too: it is not separable from the first call.
        logger.info(
            "query models ready", extra={**self.describe(), "warm_up_seconds": warm_up_seconds}
        )

    async def _load_reranker(self, name: str) -> Reranker:
        verified = await self._source.fetch(name)
        reranker = await run_inference(self._executor, lambda: self._build_reranker(verified))
        await reranker.rerank(WARM_UP_QUERY, self._warm_up_batch)
        self._rerankers[name] = reranker
        return reranker

    def _check_ready(self) -> None:
        if self.state is not ModelsState.READY:
            raise ModelsNotReadyError(f"query models are {self.state.value}")

    def query_embedder(self) -> QueryEmbedder:
        if self.state is not ModelsState.READY or self._embedder is None:
            raise ModelsNotReadyError(f"query models are {self.state.value}")
        return self._embedder

    async def reranker(self, name: str | None = None) -> Reranker:
        """The configured reranker, or another pinned one, loaded on first use."""
        self._check_ready()
        wanted = name or self.names.reranker
        async with self._reranker_lock:
            if wanted not in self._rerankers:
                await self._load_reranker(wanted)
        return self._rerankers[wanted]

    def verified_files(self) -> tuple[ModelFiles, ModelFiles]:
        """The dense and sparse files hashed at load; the index must use the same."""
        if self.state is not ModelsState.READY or self._verified is None:
            raise ModelsNotReadyError(f"query models are {self.state.value}")
        return self._verified

    def describe(self) -> dict[str, str]:
        return {
            "dense": self.names.dense,
            "sparse": self.names.sparse,
            "reranker": self.names.reranker,
        }


class ModelsProbe:
    """Readiness of the query models: ready only once all are loaded and warm."""

    name = "models"

    def __init__(self, models: QueryModels) -> None:
        self._models = models

    async def ping(self) -> Mapping[str, str] | None:
        state = self._models.state
        if state is ModelsState.LOADING:
            raise DependencyUnavailableError(
                "query models are loading", reason=FailureReason.LOADING
            )
        if state is ModelsState.FAILED:
            raise DependencyUnavailableError("query models failed", reason=FailureReason.FAILED)
        return self._models.describe()

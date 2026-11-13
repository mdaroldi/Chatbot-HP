"""Retrieval orchestration: embed the question, search, rerank, judge relevance.

Every stage is timed, because Phase 6 needs to see where a request spends its time,
and every failure becomes one domain error naming its stage. Logs carry lengths,
counts, ids, scores and timings, never the question or chunk text (CLAUDE.md).
"""

import logging
import time
from collections.abc import Awaitable, Callable, Sequence

from app.core.settings import Settings
from app.domain.ingestion import SparseVector
from app.domain.retrieval import (
    AppliedFilter,
    FilterSource,
    ModelsNotReadyError,
    QueryEmbedder,
    Relevance,
    Reranker,
    RetrievalConfig,
    RetrievalResult,
    RetrievalUnavailableError,
    RetrievedChunk,
    SearchHit,
    SearchRequest,
    SearchStore,
    StageTimings,
)

logger = logging.getLogger(__name__)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


async def _stage[T](stage: str, call: Callable[[], Awaitable[T]]) -> tuple[T, float]:
    """Run one stage and time it; any failure becomes RetrievalUnavailableError.

    A domain error raised inside the stage is kept as it is, and so is
    ModelsNotReadyError, which the API reports differently.
    """
    start = time.perf_counter()
    try:
        result = await call()
    except (RetrievalUnavailableError, ModelsNotReadyError) as exc:
        _log_failure(getattr(exc, "stage", stage), type(exc).__name__)
        raise
    except Exception as exc:
        _log_failure(stage, type(exc).__name__)
        raise RetrievalUnavailableError(stage, type(exc).__name__) from exc
    return result, _ms(start)


def _log_failure(stage: str, error_type: str) -> None:
    logger.warning("retrieval failed", extra={"stage": stage, "error_type": error_type})


def _ranked(
    hits: list[SearchHit], scores: list[float] | None, top_k: int
) -> tuple[RetrievedChunk, ...]:
    order: Sequence[tuple[SearchHit, float | None]]
    if scores is None:
        order = [(hit, None) for hit in hits]
    else:
        # sorted() is stable: equal reranker scores keep the search order.
        reranked = sorted(zip(hits, scores, strict=True), key=lambda pair: -pair[1])
        order = reranked
    return tuple(
        RetrievedChunk(hit=hit, rank=rank, rerank_score=score)
        for rank, (hit, score) in enumerate(order[:top_k], 1)
    )


class RetrievalService:
    """One retrieval pipeline over injected adapters; configurations are per call, so
    the benchmark compares them in one process without reloading models.
    """

    def __init__(self, embedder: QueryEmbedder, store: SearchStore, reranker: Reranker) -> None:
        self._embedder, self._store, self._reranker = embedder, store, reranker

    @property
    def reranker_model(self) -> str:
        return self._reranker.model

    async def retrieve(
        self, query: str, config: RetrievalConfig, *, doc_id: str | None = None
    ) -> RetrievalResult:
        start = time.perf_counter()
        dense, dense_ms = await self._dense(query, config)
        sparse, sparse_ms = await self._sparse(query, config)
        applied = AppliedFilter(doc_id, FilterSource.REQUEST) if doc_id is not None else None
        request = SearchRequest(
            mode=config.mode,
            dense=dense,
            sparse=sparse,
            limit=config.candidates,
            prefetch_dense=config.prefetch_dense,
            prefetch_sparse=config.prefetch_sparse,
            rrf_k=config.rrf_k,
            doc_id=applied.doc_id if applied is not None else None,
        )
        hits, search_ms = await _stage("search", lambda: self._store.search(request))
        scores, rerank_ms = await self._rerank(query, hits, config)
        chunks = _ranked(hits, scores, config.top_k)
        top = chunks[0].rerank_score if chunks else None
        result = RetrievalResult(
            chunks=chunks,
            relevance=Relevance.from_top_score(top, threshold=config.relevance_threshold),
            filter=applied,
            timings=StageTimings(dense_ms, sparse_ms, search_ms, rerank_ms, _ms(start)),
            config=config,
        )
        _log_done(query, result)
        return result

    async def _dense(
        self, query: str, config: RetrievalConfig
    ) -> tuple[list[float] | None, float | None]:
        if not config.mode.uses_dense:
            return None, None
        text = config.dense_query_prefix + query
        return await _stage("embed", lambda: self._embedder.embed_query_dense(text))

    async def _sparse(
        self, query: str, config: RetrievalConfig
    ) -> tuple[SparseVector | None, float | None]:
        if not config.mode.uses_sparse:
            return None, None
        return await _stage("embed", lambda: self._embedder.embed_query_sparse(query))

    async def _rerank(
        self, query: str, hits: list[SearchHit], config: RetrievalConfig
    ) -> tuple[list[float] | None, float | None]:
        if not config.rerank or not hits:
            return None, None
        texts = [hit.text for hit in hits]
        return await _stage("rerank", lambda: self._reranker.rerank(query, texts))


def _log_done(query: str, result: RetrievalResult) -> None:
    timings = result.timings
    logger.info(
        "retrieval done",
        extra={
            "query_chars": len(query),
            "mode": result.config.mode.value,
            "rerank": result.config.rerank,
            "candidates": result.config.candidates,
            "results": len(result.chunks),
            "chunk_ids": [chunk.hit.chunk_id for chunk in result.chunks],
            "top_score": result.chunks[0].score if result.chunks else None,
            "above_threshold": result.relevance.above_threshold,
            "filter_doc_id": result.filter.doc_id if result.filter else None,
            "filter_source": result.filter.source.value if result.filter else None,
            "dense_embed_ms": timings.dense_embed_ms,
            "sparse_embed_ms": timings.sparse_embed_ms,
            "search_ms": timings.search_ms,
            "rerank_ms": timings.rerank_ms,
            "total_ms": timings.total_ms,
        },
    )


def config_from_settings(settings: Settings) -> RetrievalConfig:
    return RetrievalConfig(
        mode=settings.retrieval_mode,
        rerank=settings.retrieval_rerank_enabled,
        top_k=settings.retrieval_top_k,
        rerank_candidates=settings.retrieval_rerank_candidates,
        prefetch_dense=settings.retrieval_prefetch_dense_limit,
        prefetch_sparse=settings.retrieval_prefetch_sparse_limit,
        rrf_k=settings.retrieval_rrf_k,
        product_filter=settings.retrieval_product_filter_enabled,
        relevance_threshold=settings.retrieval_relevance_threshold,
        dense_query_prefix=settings.retrieval_dense_query_prefix,
    )

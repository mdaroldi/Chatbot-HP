"""POST /v1/search: the retrieval pipeline over HTTP, for debugging and the benchmark.

It is mounted only when SEARCH_ENDPOINT_ENABLED is set (rejected in production), and
it returns chunk text, so it is not a public API. The limits come from settings, so
the request model is built per app.
"""

import dataclasses
from collections.abc import Collection
from http import HTTPStatus
from typing import Annotated, cast

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.dependencies import get_app_settings, get_client_id, get_retrieval_service
from app.core.settings import Settings
from app.domain.retrieval import RetrievalResult, RetrievedChunk, SearchMode
from app.services.retrieval import RetrievalService, config_from_settings

# The API never returns more than this, whatever the candidates setting allows.
MAX_TOP_K = 20


class Pages(BaseModel):
    pdf_start: int
    pdf_end: int
    printed_start: int | None
    printed_end: int | None


class Scores(BaseModel):
    fused: float = Field(description="The store's score: RRF when hybrid, else cosine or BM25.")
    rerank: float | None = Field(description="The cross-encoder logit; null without rerank.")


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    section_path: list[str]
    pages: Pages
    content_type: str
    has_warning: bool
    score: float = Field(description="The score that ordered the results.")
    scores: Scores
    text: str

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> "SearchResult":
        hit = chunk.hit
        return cls(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            section_path=list(hit.section_path),
            pages=Pages(**dataclasses.asdict(hit.pages)),
            content_type=hit.content_type.value,
            has_warning=hit.has_warning,
            score=chunk.score,
            scores=Scores(fused=hit.score, rerank=chunk.rerank_score),
            text=hit.text,
        )


class RelevanceInfo(BaseModel):
    signal: float | None = Field(description="Top reranker score; null without rerank.")
    threshold: float | None
    flag: bool | None = Field(description="signal >= threshold; null if either is unset.")


class FilterInfo(BaseModel):
    doc_id: str
    source: str = Field(description="`request` or `product_rule`.")


class TimingsMs(BaseModel):
    dense_embed: float | None
    sparse_embed: float | None
    search: float
    rerank: float | None
    total: float


class ConfigInfo(BaseModel):
    mode: SearchMode
    rerank: bool
    rrf_k: int
    candidates: int


class SearchResponse(BaseModel):
    results: list[SearchResult]
    relevance: RelevanceInfo
    filter: FilterInfo | None
    timings_ms: TimingsMs
    config: ConfigInfo

    @classmethod
    def from_result(cls, result: RetrievalResult) -> "SearchResponse":
        timings, config = result.timings, result.config
        applied = result.filter
        return cls(
            results=[SearchResult.from_chunk(chunk) for chunk in result.chunks],
            relevance=RelevanceInfo(
                signal=result.relevance.signal,
                threshold=result.relevance.threshold,
                flag=result.relevance.above_threshold,
            ),
            filter=FilterInfo(doc_id=applied.doc_id, source=applied.source.value)
            if applied is not None
            else None,
            timings_ms=TimingsMs(
                dense_embed=timings.dense_embed_ms,
                sparse_embed=timings.sparse_embed_ms,
                search=timings.search_ms,
                rerank=timings.rerank_ms,
                total=timings.total_ms,
            ),
            config=ConfigInfo(
                mode=config.mode,
                rerank=config.rerank,
                rrf_k=config.rrf_k,
                candidates=config.candidates,
            ),
        )


class SearchQuery(BaseModel):
    """The request fields; the per-app subclass adds this app's limits."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str
    top_k: int
    doc_id: str | None = None


def search_request_model(settings: Settings, doc_ids: Collection[str]) -> type[SearchQuery]:
    """The request schema under this app's limits; unknown fields are rejected."""
    known = frozenset(doc_ids)

    class SearchRequestBody(SearchQuery):
        query: str = Field(min_length=1, max_length=settings.query_max_chars)
        top_k: int = Field(
            default=settings.retrieval_top_k,
            ge=1,
            le=min(MAX_TOP_K, settings.retrieval_rerank_candidates),
        )
        doc_id: str | None = Field(default=None, description="Search one manual only.")

        @field_validator("doc_id")
        @classmethod
        def _known_document(cls, value: str | None) -> str | None:
            # The message names the rule, never the value.
            if value is not None and value not in known:
                raise ValueError("doc_id must name a corpus document")
            return value

    return SearchRequestBody


def build_search_router(settings: Settings, doc_ids: Collection[str]) -> APIRouter:
    body_model = search_request_model(settings, doc_ids)
    # The client ID is checked like the request ID; Phase 3's rate limiter keys on it.
    router = APIRouter(prefix="/v1", tags=["search"], dependencies=[Depends(get_client_id)])

    @router.post(
        "/search",
        responses={
            HTTPStatus.UNPROCESSABLE_ENTITY: {"description": "Invalid request (input not echoed)."},
            HTTPStatus.SERVICE_UNAVAILABLE: {
                "description": "`retrieval_unavailable` or `models_not_ready`."
            },
        },
    )
    async def search(
        body: body_model,  # type: ignore[valid-type]  # built per app from settings
        app_settings: Annotated[Settings, Depends(get_app_settings)],
        service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    ) -> SearchResponse:
        """Retrieve chunks for one question with the configured pipeline."""
        request = cast(SearchQuery, body)
        config = dataclasses.replace(config_from_settings(app_settings), top_k=request.top_k)
        result = await service.retrieve(request.query, config, doc_id=request.doc_id)
        return SearchResponse.from_result(result)

    return router

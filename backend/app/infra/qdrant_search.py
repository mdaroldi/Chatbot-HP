"""Search over Qdrant's Query API, and the probe that the live index fits the API.

Request building is pure (`build_query`), so the shape of every query is unit-tested:

- dense or sparse: one query on that named vector;
- hybrid: a dense and a sparse prefetch, fused with reciprocal rank fusion and its `k`.

Every query asks for the union of both prefetch limits, which in hybrid mode is every
fused point, and the store cuts to the request's limit itself: by score descending, then
chunk id ascending. Qdrant breaks score ties at its own limit differently between runs
(RRF ties are common), so a server-side cut made identical runs give different candidate
lists (ADR 0009).

A document filter goes into every prefetch and at the top level. A filter applied
only after fusion would leave the fused list short, or empty, when the nearest chunks
belong to the other manual; the integration tests check both placements on the server
(ADR 0009). All reads go through the alias, never a collection name.
"""

import asyncio
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient, models

from app.core.settings import Settings
from app.domain.health import DependencyUnavailableError, FailureReason
from app.domain.ingestion import ContentType, ManifestStore, ModelFiles, PageRange
from app.domain.retrieval import (
    ModelsNotReadyError,
    RetrievalUnavailableError,
    SearchHit,
    SearchMode,
    SearchRequest,
)
from app.infra.qdrant_index import DENSE, SPARSE

# What a hit needs; never the vectors.
PAYLOAD_FIELDS = (
    "doc_id",
    "section_path",
    "pdf_page_start",
    "pdf_page_end",
    "printed_page_start",
    "printed_page_end",
    "content_type",
    "has_warning",
    "text",
)


def create_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    """REST, no compatibility probe (ADR 0002); creating it opens no connection.

    The client timeout is whole seconds; the store adds a precise per-search deadline.
    """
    return AsyncQdrantClient(
        url=str(settings.qdrant_url),
        timeout=max(1, math.ceil(settings.qdrant_search_timeout_seconds)),
        check_compatibility=False,
    )


def dense_dimension(model: str) -> int:
    """From fastembed's model list; no model file is read."""
    return TextEmbedding.get_embedding_size(model)


@dataclass(frozen=True)
class QueryPlan:
    """The keyword arguments of one `query_points` call."""

    query: list[float] | models.SparseVector | models.RrfQuery
    using: str | None
    prefetch: list[models.Prefetch] | None
    query_filter: models.Filter | None
    limit: int
    with_payload: list[str]
    with_vectors: bool = False


def _doc_filter(doc_id: str | None) -> models.Filter | None:
    if doc_id is None:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    )


def build_query(request: SearchRequest) -> QueryPlan:
    found = _doc_filter(request.doc_id)
    dense = list(request.dense) if request.dense is not None else None
    sparse = (
        models.SparseVector(
            indices=list(request.sparse.indices), values=list(request.sparse.values)
        )
        if request.sparse is not None
        else None
    )
    payload = list(PAYLOAD_FIELDS)
    fetch = request.prefetch_dense + request.prefetch_sparse
    if request.mode is SearchMode.DENSE and dense is not None:
        return QueryPlan(dense, DENSE, None, found, fetch, payload)
    if request.mode is SearchMode.SPARSE and sparse is not None:
        return QueryPlan(sparse, SPARSE, None, found, fetch, payload)
    prefetch = [
        models.Prefetch(query=dense, using=DENSE, limit=request.prefetch_dense, filter=found),
        models.Prefetch(query=sparse, using=SPARSE, limit=request.prefetch_sparse, filter=found),
    ]
    fusion = models.RrfQuery(rrf=models.Rrf(k=request.rrf_k))
    return QueryPlan(fusion, None, prefetch, found, fetch, payload)


def cut(hits: list[SearchHit], limit: int) -> list[SearchHit]:
    """The same hits for the same scores, whatever order the server returned ties in."""
    return sorted(hits, key=lambda hit: (-hit.score, hit.chunk_id))[:limit]


def to_hit(point: models.ScoredPoint) -> SearchHit:
    """A KeyError or ValueError here means the index is not what ingestion writes."""
    payload = point.payload or {}
    return SearchHit(
        chunk_id=str(point.id),
        doc_id=str(payload["doc_id"]),
        section_path=tuple(str(part) for part in payload["section_path"]),
        pages=PageRange(
            payload["pdf_page_start"],
            payload["pdf_page_end"],
            payload["printed_page_start"],
            payload["printed_page_end"],
        ),
        content_type=ContentType(payload["content_type"]),
        has_warning=bool(payload["has_warning"]),
        text=str(payload["text"]),
        score=float(point.score),
    )


class QdrantSearchStore:
    """SearchStore over the live alias, REST, with a hard deadline per search.

    Any failure (server down, timeout, missing alias, unexpected payload) becomes
    RetrievalUnavailableError with the exception type only; the service logs it.
    """

    def __init__(self, client: AsyncQdrantClient, alias: str, *, timeout_seconds: float) -> None:
        self._client, self._alias, self._timeout = client, alias, timeout_seconds

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        plan = build_query(request)
        try:
            async with asyncio.timeout(self._timeout):
                response = await self._client.query_points(
                    self._alias,
                    query=plan.query,
                    using=plan.using,
                    prefetch=plan.prefetch,
                    query_filter=plan.query_filter,
                    limit=plan.limit,
                    with_payload=plan.with_payload,
                    with_vectors=plan.with_vectors,
                )
            return cut([to_hit(point) for point in response.points], request.limit)
        except Exception as exc:
            raise RetrievalUnavailableError("search", type(exc).__name__) from exc


# ---------------------------------------------------------------- index probe


class VerifiedModelFiles(Protocol):
    def verified_files(self) -> tuple[ModelFiles, ModelFiles]:
        """The dense and sparse files the API loaded; ModelsNotReadyError until then."""
        ...


def _incompatible(what: str) -> DependencyUnavailableError:
    return DependencyUnavailableError(what, reason=FailureReason.INCOMPATIBLE)


def _check_vectors(params: models.CollectionParams, dense_size: int) -> None:
    vectors = params.vectors if isinstance(params.vectors, dict) else {}
    dense = vectors.get(DENSE)
    if dense is None or dense.size != dense_size or dense.distance != models.Distance.COSINE:
        raise _incompatible("the dense vector does not match the query model")
    sparse = (params.sparse_vectors or {}).get(SPARSE)
    if sparse is None or sparse.modifier != models.Modifier.IDF:
        raise _incompatible("the sparse vector is missing or has no IDF modifier")


class IndexProbe:
    """Ready when the alias names a collection this API can query correctly.

    The collection must have the dense vector of the query model's size (cosine) and
    the IDF-weighted sparse vector, a succeeded build in the manifest, and that build's
    model files must be exactly the files the API verified and loaded (ADR 0006): query
    vectors from other files would be silently wrong. The BM25 `avg_len` affects only
    document vectors (ADR 0009), so any value the build recorded is accepted.
    """

    name = "index"

    def __init__(
        self,
        client: AsyncQdrantClient,
        alias: str,
        manifest: ManifestStore,
        models: VerifiedModelFiles,
        *,
        dense_size: int,
    ) -> None:
        self._client, self._alias, self._manifest = client, alias, manifest
        self._models, self._dense_size = models, dense_size

    async def ping(self) -> Mapping[str, str] | None:
        aliases = (await self._client.get_aliases()).aliases
        collection = next((a.collection_name for a in aliases if a.alias_name == self._alias), None)
        if collection is None:
            raise DependencyUnavailableError("no live index", reason=FailureReason.MISSING)
        _check_vectors(
            (await self._client.get_collection(collection)).config.params, self._dense_size
        )
        build = await self._manifest.succeeded_build(collection)
        if build is None:
            raise DependencyUnavailableError("no succeeded build", reason=FailureReason.MISSING)
        try:
            dense, sparse = self._models.verified_files()
        except ModelsNotReadyError as exc:
            raise DependencyUnavailableError(
                "waiting for the query models", reason=FailureReason.LOADING
            ) from exc
        if (build.dense, build.sparse) != (dense, sparse):
            raise _incompatible("the index was built with other model files")
        return {"collection": collection, "build_id": build.build_id}

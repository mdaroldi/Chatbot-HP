"""Retrieval types and interfaces (pure, no I/O).

A question becomes a dense and a sparse query vector, Qdrant returns candidates
(dense, sparse, or both fused with reciprocal rank fusion), and a cross-encoder may
rerank them. Every knob is a `RetrievalConfig` value, so the benchmark can compare
configurations in one process and Phase 5 can rerun them from env vars alone.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from app.domain.ingestion import ContentType, PageRange, SparseVector

# Qdrant's own default is 2; the paper that introduced RRF used 60.
DEFAULT_RRF_K = 60


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class SearchMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"

    @property
    def uses_dense(self) -> bool:
        return self is not SearchMode.SPARSE

    @property
    def uses_sparse(self) -> bool:
        return self is not SearchMode.DENSE


@dataclass(frozen=True)
class RetrievalConfig:
    """One retrieval configuration.

    `rerank_candidates` chunks are reranked down to `top_k`; without rerank, the
    search returns `top_k` directly. Each prefetch must return at least as many
    chunks as the fusion is asked for, or a filtered or fused search comes back short.
    """

    mode: SearchMode
    rerank: bool
    top_k: int
    rerank_candidates: int
    prefetch_dense: int
    prefetch_sparse: int
    rrf_k: int
    product_filter: bool
    relevance_threshold: float | None
    dense_query_prefix: str

    def __post_init__(self) -> None:
        _require(self.top_k >= 1, "top_k must be at least 1")
        _require(self.rerank_candidates >= self.top_k, "rerank_candidates must be at least top_k")
        _require(
            min(self.prefetch_dense, self.prefetch_sparse) >= self.rerank_candidates,
            "prefetch limits must be at least rerank_candidates",
        )
        _require(self.rrf_k >= 1, "rrf_k must be at least 1")

    @property
    def candidates(self) -> int:
        """How many chunks the search returns."""
        return self.rerank_candidates if self.rerank else self.top_k


@dataclass(frozen=True)
class SearchRequest:
    """What the vector store is asked: the vectors the mode needs and every limit."""

    mode: SearchMode
    dense: Sequence[float] | None
    sparse: SparseVector | None
    limit: int
    prefetch_dense: int
    prefetch_sparse: int
    rrf_k: int = DEFAULT_RRF_K
    doc_id: str | None = None

    def __post_init__(self) -> None:
        _require(
            (self.dense is not None) == self.mode.uses_dense,
            f"a {self.mode.value} search needs exactly the dense vector it uses",
        )
        _require(
            (self.sparse is not None) == self.mode.uses_sparse,
            f"a {self.mode.value} search needs exactly the sparse vector it uses",
        )
        _require(self.limit >= 1, "limit must be at least 1")
        _require(
            min(self.prefetch_dense, self.prefetch_sparse) >= self.limit,
            "prefetch limits must be at least the limit",
        )
        _require(self.rrf_k >= 1, "rrf_k must be at least 1")
        _require(self.doc_id is None or bool(self.doc_id.strip()), "doc_id must not be blank")


@dataclass(frozen=True)
class SearchHit:
    """One chunk as the store returned it; `score` is the store's (cosine, BM25 or RRF)."""

    chunk_id: str
    doc_id: str
    section_path: tuple[str, ...]
    pages: PageRange
    content_type: ContentType
    has_warning: bool
    text: str
    score: float


@dataclass(frozen=True)
class RetrievedChunk:
    hit: SearchHit
    rank: int
    rerank_score: float | None

    def __post_init__(self) -> None:
        _require(self.rank >= 1, "rank starts at 1")

    @property
    def score(self) -> float:
        """The score that ordered this chunk: the reranker's when it ran."""
        return self.rerank_score if self.rerank_score is not None else self.hit.score


def _single_word(value: str) -> bool:
    return bool(value) and value.split() == [value]


@dataclass(frozen=True)
class ProductName:
    """How a question names one manual's product: `ENVY`, or `envy 6000`."""

    doc_id: str
    name: str
    model_number: str | None

    def __post_init__(self) -> None:
        _require(_single_word(self.name), "a product name must be a single word")
        _require(
            self.model_number is None or _single_word(self.model_number),
            "a model number must be a single word",
        )


def _mentions(question: str, product: ProductName) -> bool:
    name = re.escape(product.name)
    for match in re.finditer(rf"(?<![^\W_]){name}(?![^\W_])", question, re.IGNORECASE):
        if not match.group().islower():
            return True
    if product.model_number is None:
        return False
    model = re.escape(product.model_number)
    return re.search(rf"(?<![^\W_]){name}[\s-]*{model}(?!\d)", question, re.IGNORECASE) is not None


def named_product(question: str, products: Sequence[ProductName]) -> str | None:
    """The manual a question names explicitly, when it names exactly one.

    A bare name counts only when it is not all lowercase ("ENVY", "Omen"), because
    "envy" and "omen" are also English words; a name in any case followed by its
    model number ("envy 6000", "omen 17") always counts. Category words such as
    "printer" or "laptop" never count. Both products or none: no filter.
    """
    named = {product.doc_id for product in products if _mentions(question, product)}
    return named.pop() if len(named) == 1 else None


class FilterSource(StrEnum):
    REQUEST = "request"
    PRODUCT_RULE = "product_rule"


@dataclass(frozen=True)
class AppliedFilter:
    doc_id: str
    source: FilterSource

    def __post_init__(self) -> None:
        _require(bool(self.doc_id.strip()), "doc_id must not be blank")


@dataclass(frozen=True)
class Relevance:
    """The top reranker score against the calibrated threshold.

    Fused RRF scores are rank-based, not a relevance signal, so without the reranker
    the signal is unavailable. Without a threshold the signal is still reported, with
    no flag. Phase 3 decides what the flag does.
    """

    signal: float | None
    threshold: float | None
    above_threshold: bool | None

    @classmethod
    def from_top_score(cls, score: float | None, *, threshold: float | None) -> "Relevance":
        above = None if score is None or threshold is None else score >= threshold
        return cls(signal=score, threshold=threshold, above_threshold=above)

    @property
    def available(self) -> bool:
        return self.signal is not None


@dataclass(frozen=True)
class StageTimings:
    """Milliseconds per stage; None for a stage the configuration skipped."""

    dense_embed_ms: float | None
    sparse_embed_ms: float | None
    search_ms: float
    rerank_ms: float | None
    total_ms: float


@dataclass(frozen=True)
class RetrievalResult:
    chunks: tuple[RetrievedChunk, ...]
    relevance: Relevance
    filter: AppliedFilter | None
    timings: StageTimings
    config: RetrievalConfig


class RetrievalUnavailableError(Exception):
    """A retrieval stage could not answer: the store is down, timed out or has no
    live index, or a model call failed.

    It names the stage and the original error type only, never its message, which can
    hold the query, hosts or server details.
    """

    def __init__(self, stage: str, error_type: str) -> None:
        super().__init__(f"retrieval stage {stage} failed ({error_type})")
        self.stage, self.error_type = stage, error_type


@dataclass(frozen=True)
class IndexBuildRef:
    """The collection behind the live alias and the manifest build that wrote it."""

    collection: str
    build_id: str


class ModelsNotReadyError(Exception):
    """The query models are still loading or failed to load."""


# ---------------------------------------------------------------- interfaces


class QueryEmbedder(Protocol):
    """Query-side encoders. They must match the document encoders the index was
    built with, but use the query paths (BM25 queries carry unit weights).
    """

    async def embed_query_dense(self, text: str) -> list[float]: ...

    async def embed_query_sparse(self, text: str) -> SparseVector: ...


class SearchStore(Protocol):
    async def search(self, request: SearchRequest) -> list[SearchHit]:
        """Best first; raise RetrievalUnavailableError on any store failure."""
        ...


class Reranker(Protocol):
    @property
    def model(self) -> str: ...

    async def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        """One score per text, in input order; higher is more relevant."""
        ...


class IndexIdentity(Protocol):
    """Names the index build a request searches, so each stored answer can be traced
    to the exact chunks and manifest it came from. Raises RetrievalUnavailableError.
    """

    async def current(self) -> IndexBuildRef: ...

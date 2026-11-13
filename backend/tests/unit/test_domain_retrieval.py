"""Retrieval domain types: configuration rules, search requests and results."""

import dataclasses
from typing import Any

import pytest

from app.domain.ingestion import ContentType, PageRange, SparseVector
from app.domain.retrieval import (
    AppliedFilter,
    FilterSource,
    Relevance,
    RetrievalConfig,
    RetrievedChunk,
    SearchHit,
    SearchMode,
    SearchRequest,
)

BASE = RetrievalConfig(
    mode=SearchMode.HYBRID,
    rerank=True,
    top_k=5,
    rerank_candidates=20,
    prefetch_dense=20,
    prefetch_sparse=20,
    rrf_k=60,
    product_filter=False,
    relevance_threshold=None,
    dense_query_prefix="",
)


# Any: per-case overrides of fields with different types; replace() validates them.
def config(**overrides: Any) -> RetrievalConfig:
    # replace() runs __post_init__ again, so every variant is validated.
    return dataclasses.replace(BASE, **overrides)


def hit(score: float = 0.5) -> SearchHit:
    return SearchHit(
        chunk_id="c1",
        doc_id="omen-guide",
        section_path=("Removal", "Bottom cover"),
        pages=PageRange(10, 11, 8, 9),
        content_type=ContentType.PROCEDURE,
        has_warning=False,
        text="text",
        score=score,
    )


# ------------------------------------------------------------------ configuration


def test_candidates_are_the_rerank_pool_with_rerank_and_top_k_without() -> None:
    assert config().candidates == 20
    assert config(rerank=False).candidates == 5


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"top_k": 0}, "top_k"),
        ({"rerank_candidates": 4}, "rerank_candidates"),
        ({"prefetch_dense": 19}, "prefetch"),
        ({"prefetch_sparse": 19}, "prefetch"),
        ({"rrf_k": 0}, "rrf_k"),
    ],
)
def test_configuration_rules(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        config(**overrides)


def test_limits_at_the_bounds_are_accepted() -> None:
    edge = config(top_k=20, rerank_candidates=20, prefetch_dense=20, prefetch_sparse=20, rrf_k=1)

    assert edge.candidates == 20


# ------------------------------------------------------------------ search requests


DENSE = (0.1, 0.2)
SPARSE = SparseVector((1, 2), (1.0, 1.0))


@pytest.mark.parametrize(
    ("mode", "dense", "sparse"),
    [
        (SearchMode.DENSE, DENSE, None),
        (SearchMode.SPARSE, None, SPARSE),
        (SearchMode.HYBRID, DENSE, SPARSE),
    ],
)
def test_a_request_carries_the_vectors_its_mode_needs(
    mode: SearchMode, dense: tuple[float, ...] | None, sparse: SparseVector | None
) -> None:
    request = SearchRequest(mode, dense, sparse, limit=5, prefetch_dense=5, prefetch_sparse=5)

    assert request.doc_id is None
    assert request.rrf_k == 60


@pytest.mark.parametrize(
    ("mode", "dense", "sparse"),
    [
        (SearchMode.DENSE, None, SPARSE),
        (SearchMode.SPARSE, DENSE, None),
        (SearchMode.HYBRID, DENSE, None),
        (SearchMode.HYBRID, None, SPARSE),
    ],
)
def test_a_request_without_a_needed_vector_is_rejected(
    mode: SearchMode, dense: tuple[float, ...] | None, sparse: SparseVector | None
) -> None:
    with pytest.raises(ValueError, match="vector"):
        SearchRequest(mode, dense, sparse, limit=5, prefetch_dense=5, prefetch_sparse=5)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"limit": 0}, "limit"),
        ({"prefetch_dense": 4}, "prefetch"),
        ({"prefetch_sparse": 4}, "prefetch"),
        ({"rrf_k": 0}, "rrf_k"),
        ({"doc_id": " "}, "doc_id"),
    ],
)
def test_request_limits_are_checked(overrides: dict[str, Any], message: str) -> None:
    base = SearchRequest(SearchMode.HYBRID, DENSE, SPARSE, 5, 5, 5)
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(base, **overrides)


# ------------------------------------------------------------------ results


def test_a_reranked_chunk_is_scored_by_the_reranker() -> None:
    assert RetrievedChunk(hit(0.5), rank=1, rerank_score=3.2).score == 3.2
    assert RetrievedChunk(hit(0.5), rank=1, rerank_score=None).score == 0.5


def test_a_rank_starts_at_one() -> None:
    with pytest.raises(ValueError, match="rank"):
        RetrievedChunk(hit(), rank=0, rerank_score=None)


def test_relevance_is_unavailable_without_a_reranker_score() -> None:
    relevance = Relevance.from_top_score(None, threshold=1.0)

    assert not relevance.available
    assert relevance.above_threshold is None


@pytest.mark.parametrize(("score", "expected"), [(0.99, False), (1.0, True), (4.0, True)])
def test_relevance_compares_the_top_score_with_the_threshold(score: float, expected: bool) -> None:
    relevance = Relevance.from_top_score(score, threshold=1.0)

    assert relevance.available
    assert relevance.signal == score
    assert relevance.above_threshold is expected


def test_relevance_without_a_threshold_reports_the_signal_but_no_flag() -> None:
    relevance = Relevance.from_top_score(2.0, threshold=None)

    assert relevance.signal == 2.0
    assert relevance.above_threshold is None


def test_an_applied_filter_names_its_document_and_source() -> None:
    applied = AppliedFilter("omen-guide", FilterSource.PRODUCT_RULE)

    assert applied.source.value == "product_rule"
    with pytest.raises(ValueError, match="doc_id"):
        AppliedFilter("", FilterSource.REQUEST)

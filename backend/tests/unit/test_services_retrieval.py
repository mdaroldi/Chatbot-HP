"""RetrievalService over fakes: modes, rerank, filters, relevance, timings, errors, logs."""

import dataclasses
import json
from typing import Any

import pytest

from app.domain.retrieval import (
    FilterSource,
    ModelsNotReadyError,
    RetrievalConfig,
    RetrievalUnavailableError,
    SearchMode,
)
from app.services.retrieval import RetrievalService, config_from_settings
from tests.unit.conftest import LogReader, SettingsFactory
from tests.unit.retrieval_fakes import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    FakeQueryEmbedder,
    FakeReranker,
    FakeSearchStore,
    make_hit,
)

CANARY = "CANARY-7f3e-QUERY"

BASE = RetrievalConfig(
    mode=SearchMode.HYBRID,
    rerank=True,
    top_k=2,
    rerank_candidates=4,
    prefetch_dense=6,
    prefetch_sparse=8,
    rrf_k=60,
    product_filter=False,
    relevance_threshold=None,
    dense_query_prefix="",
)


# Any: per-case overrides of fields with different types; replace() validates them.
def config(**overrides: Any) -> RetrievalConfig:
    return dataclasses.replace(BASE, **overrides)


def service(
    hits: int = 5,
    *,
    embedder: FakeQueryEmbedder | None = None,
    store: FakeSearchStore | None = None,
    reranker: FakeReranker | None = None,
) -> tuple[RetrievalService, FakeQueryEmbedder, FakeSearchStore, FakeReranker]:
    embedder = embedder or FakeQueryEmbedder()
    store = store or FakeSearchStore([make_hit(f"c{n}", 1.0 - n / 10) for n in range(hits)])
    reranker = reranker or FakeReranker()
    return RetrievalService(embedder, store, reranker), embedder, store, reranker


# ------------------------------------------------------------------ modes


async def test_hybrid_embeds_both_ways_and_asks_for_the_rerank_candidates() -> None:
    retrieval, embedder, store, _ = service()

    await retrieval.retrieve("How do I remove the bottom cover?", config())

    assert embedder.dense_texts == ["How do I remove the bottom cover?"]
    assert embedder.sparse_texts == ["How do I remove the bottom cover?"]
    (request,) = store.requests
    assert request.mode is SearchMode.HYBRID
    assert list(request.dense or []) == DENSE_VECTOR
    assert request.sparse == SPARSE_VECTOR
    assert (request.limit, request.prefetch_dense, request.prefetch_sparse) == (4, 6, 8)
    assert request.rrf_k == 60
    assert request.doc_id is None


async def test_dense_mode_embeds_only_the_dense_query() -> None:
    retrieval, embedder, store, _ = service()

    await retrieval.retrieve("q", config(mode=SearchMode.DENSE))

    assert (embedder.dense_texts, embedder.sparse_texts) == (["q"], [])
    assert store.requests[0].sparse is None


async def test_sparse_mode_embeds_only_the_sparse_query() -> None:
    retrieval, embedder, store, _ = service()

    await retrieval.retrieve("q", config(mode=SearchMode.SPARSE))

    assert (embedder.dense_texts, embedder.sparse_texts) == ([], ["q"])
    assert store.requests[0].dense is None


async def test_the_dense_prefix_reaches_the_dense_query_only() -> None:
    retrieval, embedder, _, reranker = service()

    await retrieval.retrieve("q", config(dense_query_prefix="Represent: "))

    assert embedder.dense_texts == ["Represent: q"]
    assert embedder.sparse_texts == ["q"]
    assert reranker.calls[0][0] == "q"


async def test_rrf_k_reaches_the_store() -> None:
    retrieval, _, store, _ = service()

    await retrieval.retrieve("q", config(rrf_k=2))

    assert store.requests[0].rrf_k == 2


# ------------------------------------------------------------------ rerank


async def test_rerank_orders_the_candidates_by_reranker_score_and_cuts_at_top_k() -> None:
    hits = [make_hit("a", 0.9, text="xx"), make_hit("b", 0.8, text="xxxx")]
    hits += [make_hit("c", 0.7, text="x"), make_hit("d", 0.6, text="xxx")]
    retrieval, _, _, reranker = service(store=FakeSearchStore(hits))

    result = await retrieval.retrieve("q", config())

    assert reranker.calls == [("q", ["xx", "xxxx", "x", "xxx"])]
    assert [c.hit.chunk_id for c in result.chunks] == ["b", "d"]
    assert [c.rank for c in result.chunks] == [1, 2]
    assert [c.rerank_score for c in result.chunks] == [4.0, 3.0]
    assert [c.score for c in result.chunks] == [4.0, 3.0]


async def test_rerank_ties_keep_the_search_order() -> None:
    hits = [make_hit("a", text="same"), make_hit("b", text="same"), make_hit("c", text="same")]
    retrieval, _, _, _ = service(store=FakeSearchStore(hits))

    result = await retrieval.retrieve("q", config())

    assert [c.hit.chunk_id for c in result.chunks] == ["a", "b"]


async def test_without_rerank_the_search_returns_top_k_in_store_order() -> None:
    retrieval, _, store, reranker = service()

    result = await retrieval.retrieve("q", config(rerank=False))

    assert store.requests[0].limit == 2
    assert reranker.calls == []
    assert [c.hit.chunk_id for c in result.chunks] == ["c0", "c1"]
    assert [c.rerank_score for c in result.chunks] == [None, None]
    assert result.chunks[0].score == 1.0


async def test_fewer_candidates_than_top_k_are_all_returned() -> None:
    retrieval, _, _, _ = service(hits=1)

    result = await retrieval.retrieve("q", config())

    assert [c.hit.chunk_id for c in result.chunks] == ["c0"]


async def test_an_empty_result_skips_the_reranker() -> None:
    retrieval, _, _, reranker = service(hits=0)

    result = await retrieval.retrieve("q", config())

    assert result.chunks == ()
    assert reranker.calls == []
    assert not result.relevance.available
    assert result.timings.rerank_ms is None


# ------------------------------------------------------------------ filter


async def test_an_explicit_document_filter_reaches_the_store() -> None:
    hits = [make_hit("e1", doc_id="envy-6000-user-guide"), make_hit("o1")]
    retrieval, _, store, _ = service(store=FakeSearchStore(hits))

    result = await retrieval.retrieve("q", config(), doc_id="envy-6000-user-guide")

    assert store.requests[0].doc_id == "envy-6000-user-guide"
    assert result.filter is not None
    assert (result.filter.doc_id, result.filter.source) == (
        "envy-6000-user-guide",
        FilterSource.REQUEST,
    )
    assert [c.hit.chunk_id for c in result.chunks] == ["e1"]


async def test_no_filter_without_a_document() -> None:
    retrieval, _, _, _ = service()

    assert (await retrieval.retrieve("q", config())).filter is None


# ------------------------------------------------------------------ relevance


@pytest.mark.parametrize(("threshold", "above"), [(5.0, True), (5.01, False)])
async def test_relevance_compares_the_top_reranker_score_with_the_threshold(
    threshold: float, above: bool
) -> None:
    hits = [make_hit("a", text="x" * 3), make_hit("b", text="x" * 5)]
    retrieval, _, _, _ = service(store=FakeSearchStore(hits))

    result = await retrieval.retrieve("q", config(relevance_threshold=threshold))

    assert result.relevance.signal == 5.0
    assert result.relevance.threshold == threshold
    assert result.relevance.above_threshold is above


async def test_relevance_is_unavailable_without_rerank() -> None:
    retrieval, _, _, _ = service()

    result = await retrieval.retrieve("q", config(rerank=False, relevance_threshold=1.0))

    assert not result.relevance.available
    assert result.relevance.above_threshold is None


# ------------------------------------------------------------------ timings and result


async def test_timings_cover_every_stage_that_ran() -> None:
    retrieval, _, _, _ = service()

    timings = (await retrieval.retrieve("q", config())).timings

    stages = [timings.dense_embed_ms, timings.sparse_embed_ms, timings.rerank_ms]
    assert all(t is not None and t >= 0 for t in stages)
    assert timings.search_ms >= 0
    assert timings.total_ms >= sum(t for t in stages if t is not None) + timings.search_ms


async def test_timings_of_skipped_stages_are_none() -> None:
    retrieval, _, _, _ = service()

    timings = (await retrieval.retrieve("q", config(mode=SearchMode.SPARSE, rerank=False))).timings

    assert timings.dense_embed_ms is None
    assert timings.rerank_ms is None
    assert timings.sparse_embed_ms is not None


async def test_the_result_carries_its_configuration_and_reranker() -> None:
    retrieval, _, _, _ = service()
    chosen = config(rrf_k=7)

    result = await retrieval.retrieve("q", chosen)

    assert result.config == chosen
    assert retrieval.reranker_model == "fake/cross-encoder"


# ------------------------------------------------------------------ errors


@pytest.mark.parametrize(
    ("part", "stage"),
    [("embedder", "embed"), ("store", "search"), ("reranker", "rerank")],
)
async def test_a_failure_becomes_the_domain_error_naming_its_stage(part: str, stage: str) -> None:
    boom = RuntimeError(f"secret detail {CANARY}")
    embedder = FakeQueryEmbedder(error=boom if part == "embedder" else None)
    store = FakeSearchStore([make_hit("a")], error=boom if part == "store" else None)
    reranker = FakeReranker(error=boom if part == "reranker" else None)
    retrieval, _, _, _ = service(embedder=embedder, store=store, reranker=reranker)

    with pytest.raises(RetrievalUnavailableError) as caught:
        await retrieval.retrieve(CANARY, config())

    assert caught.value.stage == stage
    assert caught.value.error_type == "RuntimeError"
    assert CANARY not in str(caught.value)
    assert caught.value.__cause__ is boom


async def test_a_domain_error_from_the_store_is_kept_as_it_is() -> None:
    original = RetrievalUnavailableError("search", "TimeoutError")
    retrieval, _, _, _ = service(store=FakeSearchStore(error=original))

    with pytest.raises(RetrievalUnavailableError) as caught:
        await retrieval.retrieve("q", config())

    assert caught.value is original


async def test_models_that_are_not_ready_are_reported_as_such() -> None:
    retrieval, _, _, _ = service(embedder=FakeQueryEmbedder(error=ModelsNotReadyError("loading")))

    with pytest.raises(ModelsNotReadyError):
        await retrieval.retrieve("q", config())


# ------------------------------------------------------------------ logs


async def test_logs_hold_counts_scores_and_timings_but_never_the_query_or_text(
    json_logs: LogReader,
) -> None:
    hits = [make_hit("a", text=f"chunk {CANARY}"), make_hit("b", text="other")]
    retrieval, _, _, _ = service(store=FakeSearchStore(hits))

    await retrieval.retrieve(f"how {CANARY}?", config(), doc_id="omen-17-service-guide")

    lines = json_logs()
    assert CANARY not in json.dumps(lines)
    (done,) = [line for line in lines if line["message"] == "retrieval done"]
    assert done["query_chars"] == len(f"how {CANARY}?")
    assert done["mode"] == "hybrid"
    assert done["rerank"] is True
    assert done["results"] == 2
    assert done["filter_source"] == "request"
    assert done["filter_doc_id"] == "omen-17-service-guide"
    assert done["top_score"] == float(len(f"chunk {CANARY}"))
    assert done["total_ms"] >= 0


async def test_a_failure_logs_the_stage_and_error_type_only(json_logs: LogReader) -> None:
    store = FakeSearchStore(error=RuntimeError(CANARY))
    retrieval, _, _, _ = service(store=store)

    with pytest.raises(RetrievalUnavailableError):
        await retrieval.retrieve(CANARY, config())

    lines = json_logs()
    assert CANARY not in json.dumps(lines)
    (failed,) = [line for line in lines if line["message"] == "retrieval failed"]
    assert (failed["stage"], failed["error_type"]) == ("search", "RuntimeError")


# ------------------------------------------------------------------ settings


def test_the_configuration_comes_from_settings(make_settings: SettingsFactory) -> None:
    settings = make_settings(
        retrieval_mode=SearchMode.DENSE,
        retrieval_rerank_enabled=False,
        retrieval_top_k=3,
        retrieval_rerank_candidates=12,
        retrieval_prefetch_dense_limit=30,
        retrieval_prefetch_sparse_limit=40,
        retrieval_rrf_k=2,
        retrieval_product_filter_enabled=True,
        retrieval_relevance_threshold=0.5,
        retrieval_dense_query_prefix="P: ",
    )

    assert config_from_settings(settings) == RetrievalConfig(
        mode=SearchMode.DENSE,
        rerank=False,
        top_k=3,
        rerank_candidates=12,
        prefetch_dense=30,
        prefetch_sparse=40,
        rrf_k=2,
        product_filter=True,
        relevance_threshold=0.5,
        dense_query_prefix="P: ",
    )

"""The Qdrant search adapter: pure request builders, and searches on qdrant-client's
local mode (in memory, no server). Filter placement on the real server is an
integration test.
"""

import asyncio
import dataclasses
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http import models as rest

from app.domain.ingestion import ContentType, PageRange, SparseVector
from app.domain.retrieval import RetrievalUnavailableError, SearchHit, SearchMode, SearchRequest
from app.infra.qdrant_index import QdrantIndex
from app.infra.qdrant_search import PAYLOAD_FIELDS, QdrantSearchStore, build_query, to_hit
from tests.index_data import DENSE_SIZE, make_batch

pytestmark = pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect in the local Qdrant:UserWarning"
)

ALIAS = "unit_chunks"
DENSE = (1.0, 2.0, 0.5, 0.25)
SPARSE = SparseVector((2, 102), (1.0, 1.0))


def request(mode: SearchMode = SearchMode.HYBRID, **overrides: Any) -> SearchRequest:
    # Any: per-case overrides of fields with different types; replace() validates them.
    base = SearchRequest(
        mode=mode,
        dense=DENSE if mode.uses_dense else None,
        sparse=SPARSE if mode.uses_sparse else None,
        limit=3,
        prefetch_dense=6,
        prefetch_sparse=8,
        rrf_k=60,
    )
    return dataclasses.replace(base, **overrides)


DOC_FILTER = models.Filter(
    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value="doc-b"))]
)


# ------------------------------------------------------------------ request builders (pure)


def test_a_dense_search_queries_the_dense_vector_directly() -> None:
    plan = build_query(request(SearchMode.DENSE))

    assert plan.query == list(DENSE)
    assert plan.using == "dense"
    assert plan.prefetch is None
    assert plan.query_filter is None
    assert plan.limit == 14
    assert plan.with_payload == list(PAYLOAD_FIELDS)
    assert plan.with_vectors is False


def test_a_sparse_search_queries_the_sparse_vector_directly() -> None:
    plan = build_query(request(SearchMode.SPARSE))

    assert plan.query == models.SparseVector(indices=[2, 102], values=[1.0, 1.0])
    assert plan.using == "sparse"
    assert plan.prefetch is None
    assert plan.limit == 14


def test_a_hybrid_search_fuses_two_prefetches_with_rrf_and_its_k() -> None:
    plan = build_query(request(rrf_k=7))

    assert plan.query == models.RrfQuery(rrf=models.Rrf(k=7))
    assert plan.using is None
    assert plan.prefetch == [
        models.Prefetch(query=list(DENSE), using="dense", limit=6, filter=None),
        models.Prefetch(
            query=models.SparseVector(indices=[2, 102], values=[1.0, 1.0]),
            using="sparse",
            limit=8,
            filter=None,
        ),
    ]
    # The union of both prefetches: every fused point, so no tie is cut by the server.
    assert plan.limit == 14


def test_a_document_filter_goes_into_every_prefetch_and_the_top_level() -> None:
    plan = build_query(request(doc_id="doc-b"))

    assert plan.query_filter == DOC_FILTER
    assert plan.prefetch is not None
    assert [prefetch.filter for prefetch in plan.prefetch] == [DOC_FILTER, DOC_FILTER]


@pytest.mark.parametrize("mode", [SearchMode.DENSE, SearchMode.SPARSE])
def test_a_single_vector_search_filters_at_the_top_level(mode: SearchMode) -> None:
    assert build_query(request(mode, doc_id="doc-b")).query_filter == DOC_FILTER


def test_the_payload_fields_are_what_a_hit_needs_and_never_the_vectors() -> None:
    assert set(PAYLOAD_FIELDS) == {
        "doc_id",
        "section_path",
        "pdf_page_start",
        "pdf_page_end",
        "printed_page_start",
        "printed_page_end",
        "content_type",
        "has_warning",
        "text",
    }


def test_a_scored_point_becomes_a_hit() -> None:
    point = models.ScoredPoint(
        id="0b5f6c1e-1d1b-5c51-9c7e-6d2a1e8f3b11",
        version=1,
        score=0.75,
        payload={
            "doc_id": "doc-b",
            "section_path": ["Removal", "Bottom cover"],
            "pdf_page_start": 30,
            "pdf_page_end": 31,
            "printed_page_start": None,
            "printed_page_end": None,
            "content_type": "procedure",
            "has_warning": True,
            "text": "chunk text",
        },
    )

    assert to_hit(point) == SearchHit(
        chunk_id="0b5f6c1e-1d1b-5c51-9c7e-6d2a1e8f3b11",
        doc_id="doc-b",
        section_path=("Removal", "Bottom cover"),
        pages=PageRange(30, 31, None, None),
        content_type=ContentType.PROCEDURE,
        has_warning=True,
        text="chunk text",
        score=0.75,
    )


# ------------------------------------------------------------------ searches (local mode)


@pytest.fixture
async def client() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(location=":memory:")
    index = QdrantIndex(client)
    await index.create_collection("unit_c1", DENSE_SIZE)
    a, b = make_batch(4, doc_id="doc-a"), make_batch(4, doc_id="doc-b", first=10)
    await index.upsert("unit_c1", a.chunks + b.chunks, a.dense + b.dense, a.sparse + b.sparse)
    await index.point_alias(ALIAS, "unit_c1")
    yield client
    await client.close()


def store(client: object, timeout: float = 2.0) -> QdrantSearchStore:
    # cast: the stand-ins implement only query_points, the one method the store calls.
    return QdrantSearchStore(cast(AsyncQdrantClient, client), ALIAS, timeout_seconds=timeout)


async def test_a_dense_search_ranks_by_cosine_through_the_alias(client: AsyncQdrantClient) -> None:
    # make_batch's dense vectors are (1, n, 0.5, 0.25): (1, 2, .5, .25) is chunk n=2 itself.
    hits = await store(client).search(
        request(SearchMode.DENSE, limit=2, dense=(1.0, 2.0, 0.5, 0.25))
    )

    assert [hit.section_path[-1] for hit in hits] == ["Step 2", "Step 3"]
    assert hits[0].score > hits[1].score
    assert all(isinstance(hit, SearchHit) for hit in hits)


async def test_a_sparse_search_matches_on_shared_terms(client: AsyncQdrantClient) -> None:
    hits = await store(client).search(
        request(SearchMode.SPARSE, sparse=SparseVector((11, 111), (1.0, 1.0)))
    )

    assert hits[0].section_path[-1] == "Step 11"
    assert hits[0].doc_id == "doc-b"


async def test_a_filtered_hybrid_search_returns_only_that_document(
    client: AsyncQdrantClient,
) -> None:
    hits = await store(client).search(request(doc_id="doc-b", limit=3))

    assert len(hits) == 3
    assert {hit.doc_id for hit in hits} == {"doc-b"}


async def test_a_missing_alias_is_the_domain_error(client: AsyncQdrantClient) -> None:
    missing = QdrantSearchStore(client, "no_such_alias", timeout_seconds=2.0)

    with pytest.raises(RetrievalUnavailableError) as caught:
        await missing.search(request())

    assert caught.value.stage == "search"


def tied_point(chunk_id: str, score: float) -> models.ScoredPoint:
    payload: dict[str, object] = {
        "doc_id": "doc-a",
        "section_path": ["Section"],
        "pdf_page_start": 1,
        "pdf_page_end": 1,
        "printed_page_start": None,
        "printed_page_end": None,
        "content_type": "procedure",
        "has_warning": False,
        "text": "chunk text",
    }
    return models.ScoredPoint(id=chunk_id, version=1, score=score, payload=payload)


ID_A = "00000000-0000-5000-8000-00000000000a"
ID_B = "00000000-0000-5000-8000-00000000000b"
ID_C = "00000000-0000-5000-8000-00000000000c"
ID_D = "00000000-0000-5000-8000-00000000000d"


class FixedClient:
    """Answers every query with the given points, in the given order."""

    def __init__(self, points: list[models.ScoredPoint]) -> None:
        self.points = points

    async def query_points(self, *_args: object, **_kwargs: object) -> rest.QueryResponse:
        return rest.QueryResponse(points=self.points)


@pytest.mark.parametrize("mode", list(SearchMode))
async def test_tied_scores_at_the_cut_give_the_same_hits_in_any_server_order(
    mode: SearchMode,
) -> None:
    # Three chunks tie for the last two places; the server returns them in two orders.
    first = [tied_point(ID_D, 0.9), tied_point(ID_C, 0.5), tied_point(ID_A, 0.5)]
    second = [tied_point(ID_D, 0.9), tied_point(ID_A, 0.5), tied_point(ID_B, 0.5)]
    first.append(tied_point(ID_B, 0.5))
    second.append(tied_point(ID_C, 0.5))

    hits = [
        await store(FixedClient(points)).search(request(mode, limit=3))
        for points in (first, second)
    ]

    assert [hit.chunk_id for hit in hits[0]] == [ID_D, ID_A, ID_B]
    assert hits[0] == hits[1]


async def test_hits_are_ordered_by_score_before_the_chunk_id() -> None:
    points = [tied_point(ID_A, 0.1), tied_point(ID_C, 0.7), tied_point(ID_B, 0.4)]

    hits = await store(FixedClient(points)).search(request(limit=3))

    assert [(hit.chunk_id, hit.score) for hit in hits] == [(ID_C, 0.7), (ID_B, 0.4), (ID_A, 0.1)]


async def test_a_search_returns_at_most_the_requested_limit() -> None:
    points = [tied_point(chunk_id, 0.5) for chunk_id in (ID_A, ID_B, ID_C, ID_D)]

    hits = await store(FixedClient(points)).search(request(limit=2))

    assert [hit.chunk_id for hit in hits] == [ID_A, ID_B]


class HangingClient:
    async def query_points(self, *_args: object, **_kwargs: object) -> object:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def test_a_search_over_its_deadline_is_the_domain_error() -> None:
    with pytest.raises(RetrievalUnavailableError) as caught:
        await store(HangingClient(), timeout=0.01).search(request())

    assert (caught.value.stage, caught.value.error_type) == ("search", "TimeoutError")


class MalformedClient:
    async def query_points(self, *_args: object, **_kwargs: object) -> rest.QueryResponse:
        point = models.ScoredPoint(id=1, version=1, score=1.0, payload={"doc_id": "x"})
        return rest.QueryResponse(points=[point])


async def test_a_point_without_the_expected_payload_is_the_domain_error() -> None:
    with pytest.raises(RetrievalUnavailableError) as caught:
        await store(MalformedClient()).search(request())

    assert caught.value.error_type == "KeyError"

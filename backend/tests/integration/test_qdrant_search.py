"""Searches against the real Qdrant server (docker-compose.test.yml), synthetic vectors.

The filter-placement collection is built so that every nearest neighbour of the query
belongs to doc A, while the request asks for doc B: a filter applied only after fusion
would return too few chunks, or none.
"""

import uuid
from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient, models

from app.domain.health import DependencyUnavailableError, FailureReason
from app.domain.ingestion import (
    Chunk,
    ContentType,
    IndexBuild,
    ModelFiles,
    PageRange,
    SparseVector,
)
from app.domain.retrieval import RetrievalUnavailableError, SearchMode, SearchRequest
from app.infra.qdrant_index import QdrantIndex
from app.infra.qdrant_search import IndexProbe, QdrantSearchStore, build_query
from tests.integration.conftest import QDRANT_URL
from tests.manifest_data import make_build

pytestmark = pytest.mark.integration

ALIAS = "it_search_chunks"
BUILD = make_build("20261113t130000z")
COLLECTION = BUILD.collection
SIZE = 4
NAMESPACE = uuid.UUID("7d2c1f4a-5b3e-4c6d-8e9f-0a1b2c3d4e5f")
QUERY_DENSE = (1.0, 0.0, 0.0, 0.0)
QUERY_SPARSE = SparseVector((1, 2), (1.0, 1.0))
PER_DOC = 30  # more than any prefetch limit below


def chunk(doc_id: str, n: int) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid5(NAMESPACE, f"{doc_id}|{n}")),
        doc_id=doc_id,
        product="Demo",
        section_path=("Section", f"{doc_id} {n}"),
        pages=PageRange(n + 1, n + 1, None, None),
        content_type=ContentType.REFERENCE,
        has_warning=False,
        chunking_version="v1",
        token_count=10,
        text=f"{doc_id} chunk {n}",
    )


def vectors(doc_id: str, n: int) -> tuple[list[float], SparseVector]:
    """Doc A sits next to the query on both vectors; doc B is far on both."""
    if doc_id == "doc-a":
        return [1.0, 0.01 * n, 0.0, 0.0], SparseVector((1, 2, 10 + n), (3.0, 3.0, 1.0))
    return [0.0, 0.0, 1.0, 0.01 * n], SparseVector((2, 500 + n), (0.1, 1.0))


@pytest.fixture
async def client() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)
    await _clean(client)
    index = QdrantIndex(client)
    await index.create_collection(COLLECTION, SIZE)
    chunks = [chunk(doc, n) for doc in ("doc-a", "doc-b") for n in range(PER_DOC)]
    pairs = [vectors(c.doc_id, int(c.text.rsplit(" ", 1)[1])) for c in chunks]
    await index.upsert(COLLECTION, chunks, [d for d, _ in pairs], [s for _, s in pairs])
    await index.point_alias(ALIAS, COLLECTION)
    yield client
    await _clean(client)
    await client.close()


async def _clean(client: AsyncQdrantClient) -> None:
    aliases = (await client.get_aliases()).aliases
    if any(a.alias_name == ALIAS for a in aliases):
        await client.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=ALIAS))
            ]
        )
    if await client.collection_exists(COLLECTION):
        await client.delete_collection(COLLECTION)


def hybrid(doc_id: str | None = None, limit: int = 10, prefetch: int = 10) -> SearchRequest:
    return SearchRequest(
        SearchMode.HYBRID,
        QUERY_DENSE,
        QUERY_SPARSE,
        limit=limit,
        prefetch_dense=prefetch,
        prefetch_sparse=prefetch,
        rrf_k=60,
        doc_id=doc_id,
    )


def store(client: AsyncQdrantClient, alias: str = ALIAS) -> QdrantSearchStore:
    return QdrantSearchStore(client, alias, timeout_seconds=5.0)


@pytest.mark.parametrize("mode", list(SearchMode))
async def test_without_a_filter_every_mode_ranks_doc_a_first(
    client: AsyncQdrantClient, mode: SearchMode
) -> None:
    request = SearchRequest(
        mode,
        QUERY_DENSE if mode.uses_dense else None,
        QUERY_SPARSE if mode.uses_sparse else None,
        limit=5,
        prefetch_dense=10,
        prefetch_sparse=10,
    )

    hits = await store(client).search(request)

    assert len(hits) == 5
    assert {hit.doc_id for hit in hits} == {"doc-a"}


async def test_a_filtered_hybrid_search_returns_only_that_document_and_fills_the_limit(
    client: AsyncQdrantClient,
) -> None:
    hits = await store(client).search(hybrid(doc_id="doc-b", limit=10, prefetch=10))

    assert len(hits) == 10
    assert {hit.doc_id for hit in hits} == {"doc-b"}


async def test_a_top_level_filter_alone_also_reaches_the_prefetches_on_this_server(
    client: AsyncQdrantClient,
) -> None:
    """Records Qdrant 1.20's behavior (ADR 0009): the store does not rely on it, and
    puts the filter in every prefetch too.
    """
    plan = build_query(hybrid(doc_id="doc-b", limit=10, prefetch=10))
    assert plan.prefetch is not None
    unfiltered = [prefetch.model_copy(update={"filter": None}) for prefetch in plan.prefetch]

    response = await client.query_points(
        ALIAS,
        prefetch=unfiltered,
        query=plan.query,
        query_filter=plan.query_filter,
        limit=plan.limit,
        with_payload=["doc_id"],
    )

    doc_ids = [str((point.payload or {})["doc_id"]) for point in response.points]
    assert doc_ids == ["doc-b"] * 10


async def test_a_missing_alias_is_the_domain_error(client: AsyncQdrantClient) -> None:
    with pytest.raises(RetrievalUnavailableError) as caught:
        await store(client, "it_no_such_alias").search(hybrid())

    assert caught.value.stage == "search"


async def test_an_unreachable_server_is_the_domain_error(closed_port: int) -> None:
    unreachable = AsyncQdrantClient(
        url=f"http://127.0.0.1:{closed_port}", timeout=1, check_compatibility=False
    )
    try:
        with pytest.raises(RetrievalUnavailableError) as caught:
            await store(unreachable).search(hybrid())
    finally:
        await unreachable.close()

    assert caught.value.stage == "search"


class Manifest:
    async def record(self, build: IndexBuild) -> None:  # noqa: ARG002 - never called
        raise AssertionError("the probe never writes")

    async def succeeded_build(self, collection: str) -> IndexBuild | None:
        return BUILD if collection == COLLECTION else None


class Models:
    def verified_files(self) -> tuple[ModelFiles, ModelFiles]:
        return BUILD.dense, BUILD.sparse


async def test_the_index_probe_accepts_the_live_collection(client: AsyncQdrantClient) -> None:
    probe = IndexProbe(client, ALIAS, Manifest(), Models(), dense_size=SIZE)

    assert await probe.ping() == {"collection": COLLECTION, "build_id": BUILD.build_id}


async def test_the_index_probe_catches_a_wrong_dimension(client: AsyncQdrantClient) -> None:
    probe = IndexProbe(client, ALIAS, Manifest(), Models(), dense_size=384)

    with pytest.raises(DependencyUnavailableError) as caught:
        await probe.ping()

    assert caught.value.reason is FailureReason.INCOMPATIBLE


async def test_the_index_probe_catches_a_missing_vector_name(client: AsyncQdrantClient) -> None:
    await _clean(client)
    await client.create_collection(
        COLLECTION,
        vectors_config={"dense": models.VectorParams(size=SIZE, distance=models.Distance.COSINE)},
    )
    await QdrantIndex(client).point_alias(ALIAS, COLLECTION)
    probe = IndexProbe(client, ALIAS, Manifest(), Models(), dense_size=SIZE)

    with pytest.raises(DependencyUnavailableError) as caught:
        await probe.ping()

    assert caught.value.reason is FailureReason.INCOMPATIBLE

"""QdrantIndex and publishing against the real Qdrant server (docker-compose.test.yml)."""

from collections.abc import AsyncIterator
from importlib.metadata import version

import pytest
from qdrant_client import AsyncQdrantClient, models

from app.domain.ingestion import EmbeddedChunks, IndexPublishError
from app.infra.qdrant_index import QdrantIndex
from app.ingestion.publish import COLLECTION_PREFIX, collection_name, publish
from tests.index_data import DENSE_SIZE, make_batch
from tests.integration.conftest import QDRANT_URL

pytestmark = pytest.mark.integration

ALIAS = "it_hpdocs_chunks"
SERVER_VERSION = "1.20.0"  # docker-compose.yml and docker-compose.test.yml pin this image


def embedded(count: int = 3) -> EmbeddedChunks:
    batch = make_batch(count)
    return EmbeddedChunks(chunks=batch.chunks, dense=batch.dense, sparse=batch.sparse)


async def _clean(client: AsyncQdrantClient) -> None:
    aliases = (await client.get_aliases()).aliases
    if any(a.alias_name == ALIAS for a in aliases):
        await client.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=ALIAS))
            ]
        )
    for collection in (await client.get_collections()).collections:
        if collection.name.startswith((COLLECTION_PREFIX, "it_")):
            await client.delete_collection(collection.name)


@pytest.fixture
async def client() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=QDRANT_URL)  # warns (an error here) on a version mismatch
    await _clean(client)
    yield client
    await _clean(client)
    await client.close()


@pytest.fixture
def index(client: AsyncQdrantClient) -> QdrantIndex:
    return QdrantIndex(client)


async def test_server_and_client_are_the_same_version(client: AsyncQdrantClient) -> None:
    assert (await client.info()).version == SERVER_VERSION
    assert version("qdrant-client") == SERVER_VERSION


async def test_the_collection_schema_on_the_server(
    client: AsyncQdrantClient, index: QdrantIndex
) -> None:
    await index.create_collection("it_schema", DENSE_SIZE)

    info = await client.get_collection("it_schema")
    params = info.config.params
    assert isinstance(params.vectors, dict)
    assert (params.vectors["dense"].size, params.vectors["dense"].distance) == (
        DENSE_SIZE,
        models.Distance.COSINE,
    )
    assert params.sparse_vectors is not None
    assert params.sparse_vectors["sparse"].modifier == models.Modifier.IDF
    assert {name: s.data_type for name, s in info.payload_schema.items()} == {
        "doc_id": models.PayloadSchemaType.KEYWORD,
        "content_type": models.PayloadSchemaType.KEYWORD,
    }


async def test_rebuilding_yields_the_same_ids_without_duplicates(
    client: AsyncQdrantClient, index: QdrantIndex
) -> None:
    batch = embedded(5)
    first = await publish(index, ALIAS, "b1", batch, keep_previous=True)
    await index.upsert(first.collection, batch.chunks, batch.dense, batch.sparse)
    second = await publish(index, ALIAS, "b2", embedded(5), keep_previous=True)

    async def ids(name: str) -> list[str]:
        points, _ = await client.scroll(name, limit=100)
        return sorted(str(p.id) for p in points)

    expected = sorted(c.chunk_id for c in batch.chunks)
    assert await ids(first.collection) == expected
    assert await ids(second.collection) == expected
    assert await index.count(ALIAS) == 5


async def test_an_empty_sparse_vector_is_stored_and_sparse_search_still_works(
    client: AsyncQdrantClient, index: QdrantIndex
) -> None:
    batch = embedded(3)  # the last chunk's sparse vector is empty (stopwords only)
    await publish(index, ALIAS, "b1", batch, keep_previous=True)

    [empty] = await client.retrieve(ALIAS, [batch.chunks[-1].chunk_id], with_vectors=True)
    hits = await client.query_points(
        ALIAS, query=models.SparseVector(indices=[1, 101], values=[1.0, 1.0]), using="sparse"
    )

    assert isinstance(empty.vector, dict)
    assert empty.vector["sparse"] == models.SparseVector(indices=[], values=[])
    assert [str(p.id) for p in hits.points] == [batch.chunks[1].chunk_id]


async def test_alias_swap_keeps_the_previous_build_and_prunes_older_ones(
    index: QdrantIndex,
) -> None:
    for build_id in ("b1", "b2", "b3"):
        result = await publish(index, ALIAS, build_id, embedded(3), keep_previous=True)

    assert (result.previous, result.pruned) == (collection_name("b2"), (collection_name("b1"),))
    assert await index.alias_target(ALIAS) == collection_name("b3")
    assert await index.list_collections(COLLECTION_PREFIX) == [
        collection_name("b2"),
        collection_name("b3"),
    ]


async def test_a_failed_check_leaves_the_alias_and_deletes_the_new_collection(
    index: QdrantIndex,
) -> None:
    await publish(index, ALIAS, "b1", embedded(3), keep_previous=True)
    batch = embedded(3)
    duplicated = EmbeddedChunks(
        chunks=[*batch.chunks, batch.chunks[0]],
        dense=[*batch.dense, batch.dense[0]],
        sparse=[*batch.sparse, batch.sparse[0]],
    )

    with pytest.raises(IndexPublishError):
        await publish(index, ALIAS, "b2", duplicated, keep_previous=True)

    assert await index.alias_target(ALIAS) == collection_name("b1")
    assert await index.list_collections(COLLECTION_PREFIX) == [collection_name("b1")]

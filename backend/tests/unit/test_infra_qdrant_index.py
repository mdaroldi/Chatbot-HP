"""QdrantIndex on qdrant-client local mode (in memory, no server).

Local mode ignores payload indexes and warns when one is created; the integration
tests check the payload schema on the real server.
"""

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient, models

from app.infra.qdrant_index import QdrantIndex
from tests.index_data import DENSE_SIZE, make_batch

pytestmark = pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect in the local Qdrant:UserWarning"
)


@pytest.fixture
async def client() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(location=":memory:")
    yield client
    await client.close()


@pytest.fixture
def index(client: AsyncQdrantClient) -> QdrantIndex:
    return QdrantIndex(client, upsert_batch_size=2)


async def test_a_collection_has_named_dense_and_idf_weighted_sparse_vectors(
    client: AsyncQdrantClient, index: QdrantIndex
) -> None:
    await index.create_collection("c1", DENSE_SIZE)

    params = (await client.get_collection("c1")).config.params
    assert isinstance(params.vectors, dict)
    dense = params.vectors["dense"]
    assert (dense.size, dense.distance) == (DENSE_SIZE, models.Distance.COSINE)
    assert params.sparse_vectors is not None
    assert params.sparse_vectors["sparse"].modifier == models.Modifier.IDF


async def test_upserted_points_carry_the_chunk_id_payload_and_both_vectors(
    client: AsyncQdrantClient, index: QdrantIndex
) -> None:
    batch = make_batch(5)  # more than one upsert batch; the last sparse vector is empty
    await index.create_collection("c1", DENSE_SIZE)

    await index.upsert("c1", batch.chunks, batch.dense, batch.sparse)

    points = await client.retrieve(
        "c1", [c.chunk_id for c in batch.chunks], with_payload=True, with_vectors=True
    )
    by_id = {str(p.id): p for p in points}
    assert await index.count("c1") == 5
    for chunk, dense, sparse in zip(batch.chunks, batch.dense, batch.sparse, strict=True):
        point = by_id[chunk.chunk_id]
        assert point.payload == chunk.payload()
        assert isinstance(point.vector, dict)
        stored = point.vector["sparse"]
        assert isinstance(stored, models.SparseVector)
        assert (tuple(stored.indices), tuple(stored.values)) == (sparse.indices, sparse.values)
        assert point.vector["dense"] == pytest.approx(
            [v / sum(x * x for x in dense) ** 0.5 for v in dense]  # cosine stores unit vectors
        )


async def test_upserting_the_same_chunks_again_adds_no_points(index: QdrantIndex) -> None:
    batch = make_batch(3)
    await index.create_collection("c1", DENSE_SIZE)

    await index.upsert("c1", batch.chunks, batch.dense, batch.sparse)
    await index.upsert("c1", batch.chunks, batch.dense, batch.sparse)

    assert await index.count("c1") == 3


async def test_upsert_rejects_vectors_that_do_not_match_the_chunks(index: QdrantIndex) -> None:
    batch = make_batch(3)
    await index.create_collection("c1", DENSE_SIZE)

    with pytest.raises(ValueError, match="zip"):
        await index.upsert("c1", batch.chunks, batch.dense[:2], batch.sparse)
    with pytest.raises(ValueError, match="zip"):
        await index.upsert("c1", batch.chunks, batch.dense, batch.sparse[:2])

    assert await index.count("c1") == 0


async def test_an_alias_is_created_then_moved_atomically(index: QdrantIndex) -> None:
    await index.create_collection("c1", DENSE_SIZE)
    await index.create_collection("c2", DENSE_SIZE)
    assert await index.alias_target("live") is None

    await index.point_alias("live", "c1")
    first = await index.alias_target("live")
    await index.point_alias("live", "c2")

    assert (first, await index.alias_target("live")) == ("c1", "c2")


async def test_collections_are_listed_by_prefix_and_deleted(index: QdrantIndex) -> None:
    for name in ("hp_b", "other", "hp_a"):
        await index.create_collection(name, DENSE_SIZE)

    listed = await index.list_collections("hp_")
    await index.delete_collection("hp_a")

    assert listed == ["hp_a", "hp_b"]
    assert await index.list_collections("") == ["hp_b", "other"]

"""Publishing a build: versioned collection, count check, alias swap, pruning.

Runs QdrantIndex on qdrant-client local mode; failures are injected by subclasses
that fail one operation.
"""

from collections.abc import AsyncIterator, Sequence

import pytest
from qdrant_client import AsyncQdrantClient

from app.domain.ingestion import Chunk, EmbeddedChunks, IndexPublishError, SparseVector
from app.infra.qdrant_index import QdrantIndex
from app.ingestion.publish import COLLECTION_PREFIX, collection_name, publish
from tests.index_data import make_batch

pytestmark = pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect in the local Qdrant:UserWarning"
)

ALIAS = "hpdocs_chunks"


def embedded(count: int = 3, first: int = 0) -> EmbeddedChunks:
    batch = make_batch(count, first=first)
    return EmbeddedChunks(chunks=batch.chunks, dense=batch.dense, sparse=batch.sparse)


@pytest.fixture
async def client() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(location=":memory:")
    yield client
    await client.close()


@pytest.fixture
def index(client: AsyncQdrantClient) -> QdrantIndex:
    return QdrantIndex(client)


class FailingUpsert(QdrantIndex):
    async def upsert(
        self,
        name: str,
        chunks: Sequence[Chunk],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[SparseVector],
    ) -> None:
        await super().upsert(name, chunks[:1], dense[:1], sparse[:1])
        raise ConnectionError("lost the server mid-upload")


class FailingSwap(QdrantIndex):
    async def point_alias(self, alias: str, name: str) -> None:
        del alias, name
        raise ConnectionError("lost the server before the swap")


class SwapFailsAfterMoving(QdrantIndex):
    """The server applied the swap, but the reply was lost (e.g. a timeout)."""

    async def point_alias(self, alias: str, name: str) -> None:
        await super().point_alias(alias, name)
        raise TimeoutError("no reply")


# ------------------------------------------------------------------ names


def test_collection_names_are_versioned_by_build() -> None:
    assert collection_name("20261112t160000z") == f"{COLLECTION_PREFIX}20261112t160000z"
    assert COLLECTION_PREFIX == "hpdocs_chunks_v1_"


@pytest.mark.parametrize("build_id", ["", "Build", "a b", "a/b", "a.b"])
def test_a_build_id_outside_lowercase_letters_digits_and_dashes_is_rejected(
    build_id: str,
) -> None:
    assert collection_name("b-1_x") == f"{COLLECTION_PREFIX}b-1_x"
    with pytest.raises(ValueError, match="build id"):
        collection_name(build_id)


# ------------------------------------------------------------------ success


async def test_the_first_build_goes_live_under_the_alias(index: QdrantIndex) -> None:
    result = await publish(index, ALIAS, "b1", embedded(3), keep_previous=True)

    assert result.collection == collection_name("b1")
    assert (result.previous, result.pruned) == (None, ())
    assert await index.alias_target(ALIAS) == collection_name("b1")
    assert await index.count(ALIAS) == 3


async def test_a_new_build_moves_the_alias_and_keeps_only_the_previous_one(
    index: QdrantIndex,
) -> None:
    await publish(index, ALIAS, "b1", embedded(3), keep_previous=True)
    second = await publish(index, ALIAS, "b2", embedded(4), keep_previous=True)
    third = await publish(index, ALIAS, "b3", embedded(2), keep_previous=True)

    assert (second.previous, second.pruned) == (collection_name("b1"), ())
    assert (third.previous, third.pruned) == (collection_name("b2"), (collection_name("b1"),))
    assert await index.alias_target(ALIAS) == collection_name("b3")
    assert await index.list_collections(COLLECTION_PREFIX) == [
        collection_name("b2"),
        collection_name("b3"),
    ]


async def test_without_keep_previous_every_older_build_is_pruned(index: QdrantIndex) -> None:
    await publish(index, ALIAS, "b1", embedded(3), keep_previous=True)

    result = await publish(index, ALIAS, "b2", embedded(3), keep_previous=False)

    assert (result.previous, result.pruned) == (collection_name("b1"), (collection_name("b1"),))
    assert await index.list_collections(COLLECTION_PREFIX) == [collection_name("b2")]


async def test_pruning_never_touches_collections_outside_the_prefix(index: QdrantIndex) -> None:
    await index.create_collection("hpdocs_eval_scratch", 4)

    await publish(index, ALIAS, "b1", embedded(3), keep_previous=False)

    assert "hpdocs_eval_scratch" in await index.list_collections("")


# ------------------------------------------------------------------ failures


async def live_build(index: QdrantIndex) -> None:
    await publish(index, ALIAS, "b1", embedded(3), keep_previous=True)


async def test_a_count_mismatch_deletes_the_new_collection_and_keeps_the_alias(
    index: QdrantIndex,
) -> None:
    await live_build(index)
    batch = embedded(3)
    duplicated = EmbeddedChunks(  # the same id twice: Qdrant keeps one point
        chunks=[*batch.chunks, batch.chunks[0]],
        dense=[*batch.dense, batch.dense[0]],
        sparse=[*batch.sparse, batch.sparse[0]],
    )

    with pytest.raises(IndexPublishError, match="3 points for 4 chunks"):
        await publish(index, ALIAS, "b2", duplicated, keep_previous=True)

    assert await index.alias_target(ALIAS) == collection_name("b1")
    assert await index.list_collections(COLLECTION_PREFIX) == [collection_name("b1")]


async def test_a_failed_upload_deletes_the_partial_collection_and_keeps_the_alias(
    client: AsyncQdrantClient,
) -> None:
    await live_build(QdrantIndex(client))
    failing = FailingUpsert(client)

    with pytest.raises(ConnectionError):
        await publish(failing, ALIAS, "b2", embedded(3), keep_previous=True)

    assert await failing.alias_target(ALIAS) == collection_name("b1")
    assert await failing.list_collections(COLLECTION_PREFIX) == [collection_name("b1")]


async def test_a_failed_swap_deletes_the_new_collection_and_keeps_the_alias(
    client: AsyncQdrantClient,
) -> None:
    await live_build(QdrantIndex(client))
    failing = FailingSwap(client)

    with pytest.raises(ConnectionError):
        await publish(failing, ALIAS, "b2", embedded(3), keep_previous=True)

    assert await failing.alias_target(ALIAS) == collection_name("b1")
    assert await failing.list_collections(COLLECTION_PREFIX) == [collection_name("b1")]


async def test_a_swap_that_happened_despite_an_error_keeps_the_live_collection(
    client: AsyncQdrantClient,
) -> None:
    await live_build(QdrantIndex(client))
    failing = SwapFailsAfterMoving(client)

    with pytest.raises(TimeoutError):
        await publish(failing, ALIAS, "b2", embedded(3), keep_previous=True)

    assert await failing.alias_target(ALIAS) == collection_name("b2")
    assert await failing.count(ALIAS) == 3

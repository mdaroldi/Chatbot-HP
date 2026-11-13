"""Reading every indexed chunk back for the preflight (qdrant-client local mode)."""

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient

from app.domain.evaluation import IndexedChunk
from app.infra.qdrant_chunks import scroll_chunks
from app.infra.qdrant_index import QdrantIndex
from tests.index_data import DENSE_SIZE, make_batch

pytestmark = pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect in the local Qdrant:UserWarning"
)

COLLECTION = "unit_chunks_v1_20261113t000000z"


@pytest.fixture
async def client() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(location=":memory:")
    await QdrantIndex(client).create_collection(COLLECTION, DENSE_SIZE)
    yield client
    await client.close()


async def test_every_chunk_is_read_back_across_pages(client: AsyncQdrantClient) -> None:
    batch = make_batch(7)
    await QdrantIndex(client).upsert(COLLECTION, batch.chunks, batch.dense, batch.sparse)

    chunks = await scroll_chunks(client, COLLECTION, page_size=3)

    assert sorted(chunks, key=lambda c: c.chunk_id) == sorted(
        (IndexedChunk(c.chunk_id, c.doc_id, c.section_path, c.text) for c in batch.chunks),
        key=lambda c: c.chunk_id,
    )


async def test_an_empty_collection_has_no_chunks(client: AsyncQdrantClient) -> None:
    assert await scroll_chunks(client, COLLECTION) == []

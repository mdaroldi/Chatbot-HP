"""Excerpt payloads read by point id through an alias on the real Qdrant server."""

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient, models

from app.evaluation.answers.excerpts import QdrantChunkSource, rebuild_excerpts
from app.infra.qdrant_index import QdrantIndex
from tests.index_data import DENSE_SIZE, make_batch
from tests.integration.conftest import QDRANT_URL
from tests.manifest_data import make_build

pytestmark = pytest.mark.integration

ALIAS = "it_eval_excerpts"
COLLECTION = make_build("20261114t120000z").collection
BATCH = make_batch(4)


async def _clean(client: AsyncQdrantClient) -> None:
    if any(a.alias_name == ALIAS for a in (await client.get_aliases()).aliases):
        await client.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=ALIAS))
            ]
        )
    if await client.collection_exists(COLLECTION):
        await client.delete_collection(COLLECTION)


@pytest.fixture
async def client() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)
    await _clean(client)
    index = QdrantIndex(client)
    await index.create_collection(COLLECTION, DENSE_SIZE)
    await index.upsert(COLLECTION, BATCH.chunks, BATCH.dense, BATCH.sparse)
    await index.point_alias(ALIAS, COLLECTION)
    yield client
    await _clean(client)
    await client.close()


async def test_points_are_read_by_id_through_the_alias_in_stored_order(
    client: AsyncQdrantClient,
) -> None:
    ranked = [BATCH.chunks[2], BATCH.chunks[0]]

    excerpts = await rebuild_excerpts(
        [c.chunk_id for c in ranked], QdrantChunkSource(client, ALIAS), {"demo-guide": "Demo"}
    )

    assert [(e.tag, e.chunk.chunk_id) for e in excerpts] == [
        ("S1", ranked[0].chunk_id),
        ("S2", ranked[1].chunk_id),
    ]
    assert excerpts[0].text.startswith('<excerpt tag="S1" document="Demo" section="Setup &gt; ')

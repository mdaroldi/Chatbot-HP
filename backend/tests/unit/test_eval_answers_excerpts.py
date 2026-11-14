"""Excerpts rebuilt from stored chunk ids (qdrant-client local mode for the payloads)."""

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient

from app.domain.evaluation import IndexedChunk
from app.domain.ingestion import Chunk
from app.domain.prompt import render_excerpt
from app.domain.retrieval import SearchHit
from app.evaluation.answers.excerpts import (
    ChunkSource,
    ExcerptError,
    QdrantChunkSource,
    rebuild_excerpts,
)
from app.evaluation.answers.scoring import Excerpt
from app.infra.qdrant_index import QdrantIndex
from tests.index_data import DENSE_SIZE, make_batch

pytestmark = pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect in the local Qdrant:UserWarning"
)

COLLECTION = "unit_chunks_v1_20261114t000000z"
TITLES = {"demo-guide": "Demo Guide"}
MISSING = "00000000-0000-4000-8000-00000000dead"


@pytest.fixture
async def client() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(location=":memory:")
    await QdrantIndex(client).create_collection(COLLECTION, DENSE_SIZE)
    yield client
    await client.close()


@pytest.fixture
async def chunks(client: AsyncQdrantClient) -> list[Chunk]:
    batch = make_batch(5)
    await QdrantIndex(client).upsert(COLLECTION, batch.chunks, batch.dense, batch.sparse)
    return batch.chunks


def as_hit(chunk: Chunk) -> SearchHit:
    return SearchHit(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        section_path=chunk.section_path,
        pages=chunk.pages,
        content_type=chunk.content_type,
        has_warning=chunk.has_warning,
        text=chunk.text,
        score=0.0,
    )


def test_the_qdrant_source_implements_the_port(client: AsyncQdrantClient) -> None:
    source = QdrantChunkSource(client, COLLECTION)
    port: ChunkSource = source  # mypy checks the structural match
    assert port is source


async def test_the_source_reads_the_full_payload_of_the_requested_points_only(
    client: AsyncQdrantClient, chunks: list[Chunk]
) -> None:
    wanted = [chunks[3], chunks[0]]

    hits = await QdrantChunkSource(client, COLLECTION).hits([c.chunk_id for c in wanted])

    assert hits == {c.chunk_id: as_hit(c) for c in wanted}


async def test_an_id_the_collection_lacks_is_absent_from_the_hits(
    client: AsyncQdrantClient, chunks: list[Chunk]
) -> None:
    hits = await QdrantChunkSource(client, COLLECTION).hits([chunks[1].chunk_id, MISSING])

    assert set(hits) == {chunks[1].chunk_id}


async def test_excerpts_are_tagged_in_stored_rank_order_and_rendered_as_the_prompt_does(
    client: AsyncQdrantClient, chunks: list[Chunk]
) -> None:
    ranked = [chunks[4], chunks[0], chunks[3]]  # rank order, not insertion order

    excerpts = await rebuild_excerpts(
        [c.chunk_id for c in ranked], QdrantChunkSource(client, COLLECTION), TITLES
    )

    assert excerpts == tuple(
        Excerpt(
            tag=f"S{n}",
            chunk=IndexedChunk(c.chunk_id, c.doc_id, c.section_path, c.text),
            text=render_excerpt(f"S{n}", as_hit(c), TITLES),
        )
        for n, c in enumerate(ranked, 1)
    )
    assert 'warning="yes"' in excerpts[2].text  # chunks[3] has a warning note


async def test_a_stored_id_missing_from_the_collection_names_the_id_and_the_tag(
    client: AsyncQdrantClient, chunks: list[Chunk]
) -> None:
    with pytest.raises(ExcerptError, match=rf"S2 \({MISSING}\)") as caught:
        await rebuild_excerpts(
            [chunks[0].chunk_id, MISSING], QdrantChunkSource(client, COLLECTION), TITLES
        )

    assert "Demo" not in str(caught.value)

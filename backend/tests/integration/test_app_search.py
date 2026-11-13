"""POST /v1/search on the assembled app: real models, real Qdrant, the lifespan's wiring."""

import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence

import httpx
import pytest
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient, models

from app.domain.ingestion import Chunk, ContentType, PageRange
from app.infra.qdrant_index import QdrantIndex
from app.main import create_app
from tests.integration.conftest import QDRANT_URL, SettingsFactory

pytestmark = [pytest.mark.integration, pytest.mark.model]

ALIAS = "it_search_api_chunks"
COLLECTION = "it_search_api_collection"
NAMESPACE = uuid.UUID("3b0a6f1e-2d4c-4e8f-9a7b-5c6d7e8f9a0b")
ENVY = "envy-6000-user-guide"
OMEN = "omen-17-service-guide"
# Invented texts, not HP's: the test needs only two clearly different chunks.
TEXTS = {
    OMEN: "Remove the bottom cover screws, then lift the cover from the rear edge.",
    ENVY: "Load plain paper in the input tray and slide the width guides inward.",
}


def chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid5(NAMESPACE, doc_id)),
        doc_id=doc_id,
        product="Demo",
        section_path=("Demo chapter", f"Demo section {doc_id}"),
        pages=PageRange(4, 5, 2, 3),
        content_type=ContentType.PROCEDURE,
        has_warning=doc_id == OMEN,
        chunking_version="v1",
        token_count=20,
        text=text,
    )


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
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)
    await _clean(client)
    yield client
    await _clean(client)
    await client.close()


async def _index_with_the_app_models(application: FastAPI, client: AsyncQdrantClient) -> None:
    """Two chunks embedded by the app's own loaded models, behind the alias."""
    embedder = application.state.infrastructure.models.query_embedder()
    chunks: Sequence[Chunk] = [chunk(doc, text) for doc, text in TEXTS.items()]
    dense = [await embedder.embed_query_dense(c.text) for c in chunks]
    sparse = [await embedder.embed_query_sparse(c.text) for c in chunks]
    index = QdrantIndex(client)
    await index.create_collection(COLLECTION, len(dense[0]))
    await index.upsert(COLLECTION, chunks, dense, sparse)
    await index.point_alias(ALIAS, COLLECTION)


async def search(
    application: FastAPI, body: dict[str, object], client: AsyncQdrantClient | None = None
) -> httpx.Response:
    async with application.router.lifespan_context(application):
        await asyncio.wait_for(application.state.infrastructure.models_loaded, timeout=60)
        if client is not None:
            await _index_with_the_app_models(application, client)
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            return await http.post("/v1/search", json=body)


async def test_a_search_ranks_the_matching_chunk_first_with_real_scores(
    make_settings: SettingsFactory, qdrant: AsyncQdrantClient
) -> None:
    settings = make_settings(search_endpoint_enabled=True, qdrant_collection_alias=ALIAS)

    response = await search(
        create_app(settings), {"query": "how do I remove the bottom cover"}, qdrant
    )

    assert response.status_code == 200
    body = response.json()
    assert [r["doc_id"] for r in body["results"]] == [OMEN, ENVY]
    top = body["results"][0]
    assert top["section_path"] == ["Demo chapter", f"Demo section {OMEN}"]
    assert top["pages"] == {"pdf_start": 4, "pdf_end": 5, "printed_start": 2, "printed_end": 3}
    assert top["has_warning"] is True
    assert top["score"] == top["scores"]["rerank"] > body["results"][1]["scores"]["rerank"]
    assert body["relevance"]["signal"] == top["score"]
    assert body["config"] == {"mode": "hybrid", "rerank": True, "rrf_k": 60, "candidates": 20}
    assert body["timings_ms"]["rerank"] > 0


async def test_the_product_rule_uses_the_corpus_product_names(
    make_settings: SettingsFactory, qdrant: AsyncQdrantClient
) -> None:
    settings = make_settings(
        search_endpoint_enabled=True,
        qdrant_collection_alias=ALIAS,
        retrieval_product_filter_enabled=True,
    )

    response = await search(create_app(settings), {"query": "remove the cover of the Omen"}, qdrant)

    body = response.json()
    assert body["filter"] == {"doc_id": OMEN, "source": "product_rule"}
    assert [r["doc_id"] for r in body["results"]] == [OMEN]


async def test_a_missing_index_is_503_in_the_error_shape(make_settings: SettingsFactory) -> None:
    settings = make_settings(
        search_endpoint_enabled=True, qdrant_collection_alias="it_no_such_alias"
    )

    response = await search(create_app(settings), {"query": "remove the bottom cover"})

    assert response.status_code == 503
    error = response.json()["error"]
    assert (error["code"], error["message"], error["details"]) == (
        "retrieval_unavailable",
        "Retrieval is unavailable.",
        [],
    )
    assert "Traceback" not in response.text

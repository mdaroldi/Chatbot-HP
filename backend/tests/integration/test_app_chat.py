"""The chat API on the assembled app: real Postgres (migrated), real Qdrant, real models,
the stub LLM. Invented chunk texts only; the document ids are the corpus's, since the
excerpt titles come from the corpus manifest.
"""

import asyncio
import json
import pathlib
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import select

from app.domain.ingestion import Chunk, ContentType, PageRange
from app.infra.db import create_engine, create_session_factory
from app.infra.manifest_repository import SqlManifestStore
from app.infra.orm import MessageRow
from app.infra.qdrant_index import QdrantIndex
from app.main import create_app
from tests.integration.conftest import QDRANT_URL, SettingsFactory, run_alembic
from tests.manifest_data import make_build

pytestmark = [pytest.mark.integration, pytest.mark.model]

ALIAS = "it_chat_api_chunks"
BUILD = make_build("20261113t150000z")
NAMESPACE = uuid.UUID("7c1d2e3f-4a5b-4c6d-8e9f-0a1b2c3d4e5f")
ENVY = "envy-6000-user-guide"
OMEN = "omen-17-service-guide"
TEXTS = {
    OMEN: "Demo > Demo chapter > Removing the cover\n\nRemove the bottom cover screws, then "
    "lift the cover from the rear edge.",
    ENVY: "Demo > Demo chapter > Loading paper\n\nLoad plain paper in the input tray and slide "
    "the width guides inward.",
}
CLIENT_A = {"X-Client-ID": "it-client-a"}
CLIENT_B = {"X-Client-ID": "it-client-b"}
QUESTION = "How do I remove the bottom cover?"


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
        token_count=30,
        text=text,
    )


async def _clean(client: AsyncQdrantClient) -> None:
    if any(a.alias_name == ALIAS for a in (await client.get_aliases()).aliases):
        await client.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=ALIAS))
            ]
        )
    if await client.collection_exists(BUILD.collection):
        await client.delete_collection(BUILD.collection)


@pytest.fixture
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)
    await _clean(client)
    yield client
    await _clean(client)
    await client.close()


@pytest.fixture
async def chat_db(scratch_db: str, make_settings: SettingsFactory, tmp_path: pathlib.Path) -> str:
    """Migrated, with the build behind the alias recorded as succeeded (build id)."""
    run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path)
    engine = create_engine(make_settings(postgres_db=scratch_db))
    await SqlManifestStore(create_session_factory(engine)).record(BUILD)
    await engine.dispose()
    return scratch_db


async def _index(application: FastAPI, client: AsyncQdrantClient) -> None:
    embedder = application.state.infrastructure.models.query_embedder()
    chunks: Sequence[Chunk] = [chunk(doc, text) for doc, text in TEXTS.items()]
    dense = [await embedder.embed_query_dense(c.text) for c in chunks]
    sparse = [await embedder.embed_query_sparse(c.text) for c in chunks]
    index = QdrantIndex(client)
    await index.create_collection(BUILD.collection, len(dense[0]))
    await index.upsert(BUILD.collection, chunks, dense, sparse)
    await index.point_alias(ALIAS, BUILD.collection)


@asynccontextmanager
async def running(
    application: FastAPI, client: AsyncQdrantClient
) -> AsyncIterator[httpx.AsyncClient]:
    async with application.router.lifespan_context(application):
        await asyncio.wait_for(application.state.infrastructure.models_loaded, timeout=60)
        await _index(application, client)
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


@dataclass
class Stream:
    events: list[tuple[str, dict[str, Any]]]  # Any: each event's JSON payload

    def names(self) -> list[str]:
        return list(dict.fromkeys(name for name, _ in self.events))

    def data(self, name: str) -> dict[str, Any]:  # Any: one event's JSON payload
        (payload,) = [data for event, data in self.events if event == name]
        return payload

    def text(self) -> str:
        return "".join(data["text"] for name, data in self.events if name == "token")


def parse(raw: str) -> Stream:
    events = []
    for frame in raw.split("\n\n")[:-1]:
        if frame.startswith(":"):
            continue
        name, data = frame.split("\n")
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return Stream(events)


async def ask(
    http: httpx.AsyncClient, conversation_id: str, content: str, headers: dict[str, str]
) -> tuple[httpx.Response, Stream | None]:
    response = await http.post(
        f"/v1/conversations/{conversation_id}/messages", json={"content": content}, headers=headers
    )
    streamed = response.headers.get("content-type", "").startswith("text/event-stream")
    return response, parse(response.text) if streamed else None


async def stored_answer(
    database: str, make_settings: SettingsFactory, message_id: str
) -> MessageRow:
    engine = create_engine(make_settings(postgres_db=database))
    try:
        async with create_session_factory(engine)() as session:
            return (
                await session.execute(select(MessageRow).where(MessageRow.id == message_id))
            ).scalar_one()
    finally:
        await engine.dispose()


async def test_a_chat_session_over_the_real_stack(
    make_settings: SettingsFactory, qdrant: AsyncQdrantClient, chat_db: str
) -> None:
    settings = make_settings(postgres_db=chat_db, qdrant_collection_alias=ALIAS)

    async with running(create_app(settings), qdrant) as http:
        created = await http.post("/v1/conversations", headers=CLIENT_A)
        assert created.status_code == 201
        conversation_id = created.json()["id"]

        response, stream = await ask(http, conversation_id, QUESTION, CLIENT_A)
        assert response.status_code == 200
        assert stream is not None
        assert stream.names() == ["token", "sources", "done"]
        done = stream.data("done")
        # The stub cites S1, and S2 when a second excerpt exists (both chunks come back).
        assert (done["outcome"], done["refusal"], done["citations"]) == (
            "completed",
            False,
            ["S1", "S2"],
        )
        sources = stream.data("sources")["sources"]
        assert [(s["tag"], s["document_id"]) for s in sources] == [("S1", OMEN), ("S2", ENVY)]
        assert sources[0]["pages"] == {
            "pdf_start": 4,
            "pdf_end": 5,
            "printed_start": 2,
            "printed_end": 3,
        }
        assert (sources[0]["has_warning"], sources[1]["has_warning"]) == (True, False)

        _, follow_up = await ask(http, conversation_id, "And then what?", CLIENT_A)
        assert follow_up is not None
        assert follow_up.data("done")["outcome"] in {"completed", "refused"}
        follow_up_row = await stored_answer(
            chat_db, make_settings, follow_up.data("done")["message_id"]
        )
        assert follow_up_row.rewritten_query == "And then what?"  # the stub echoes it
        assert follow_up_row.rewrite_ms is not None

        listed = (await http.get("/v1/conversations", headers=CLIENT_A)).json()
        assert [c["id"] for c in listed["items"]] == [conversation_id]
        assert listed["items"][0]["title"] == QUESTION
        thread = (await http.get(f"/v1/conversations/{conversation_id}", headers=CLIENT_A)).json()
        assert [m["role"] for m in thread["messages"]] == ["user", "assistant"] * 2
        assert thread["messages"][1]["sources"][0]["document_id"] == OMEN

        for method, path in (
            ("GET", f"/v1/conversations/{conversation_id}"),
            ("DELETE", f"/v1/conversations/{conversation_id}"),
        ):
            other = await http.request(method, path, headers=CLIENT_B)
            assert other.status_code == 404, method
            assert other.json()["error"]["code"] == "conversation_not_found"
        other_post, _ = await ask(http, conversation_id, QUESTION, CLIENT_B)
        assert other_post.status_code == 404

        row = await stored_answer(chat_db, make_settings, done["message_id"])
        assert (row.model, row.index_build_id) == ("stub", BUILD.build_id)
        assert row.chunk_ids[0] == str(uuid.uuid5(NAMESPACE, OMEN))
        assert row.input_tokens
        assert row.output_tokens
        for field in (
            "dense_embed_ms",
            "search_ms",
            "rerank_ms",
            "retrieval_ms",
            "ttft_ms",
            "generation_ms",
            "total_ms",
            "relevance_signal",
        ):
            assert getattr(row, field) is not None, field
        assert row.relevance_threshold == settings.retrieval_relevance_threshold
        assert row.rewritten_query is None  # the first turn is not rewritten

        assert (
            await http.delete(f"/v1/conversations/{conversation_id}", headers=CLIENT_A)
        ).status_code == 204
        gone = await http.get(f"/v1/conversations/{conversation_id}", headers=CLIENT_A)
        assert gone.status_code == 404


async def test_a_question_below_the_threshold_is_refused_without_the_model(
    make_settings: SettingsFactory, qdrant: AsyncQdrantClient, chat_db: str
) -> None:
    """A threshold no reranker score reaches: layer 1 refuses before any LLM call."""
    settings = make_settings(
        postgres_db=chat_db, qdrant_collection_alias=ALIAS, retrieval_relevance_threshold=50.0
    )

    async with running(create_app(settings), qdrant) as http:
        conversation_id = (await http.post("/v1/conversations", headers=CLIENT_A)).json()["id"]
        _, stream = await ask(http, conversation_id, QUESTION, CLIENT_A)

    assert stream is not None
    done = stream.data("done")
    assert (done["outcome"], done["refusal_reason"]) == ("refused", "below_threshold")
    assert stream.data("sources") == {"sources": []}
    row = await stored_answer(chat_db, make_settings, done["message_id"])
    assert (row.model, row.input_tokens, row.output_tokens) == (None, None, None)
    assert row.chunk_ids

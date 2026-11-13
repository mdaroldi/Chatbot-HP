"""The assembled app, booted through its lifespan, against real dependencies."""

import asyncio
import pathlib
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI
from pydantic import HttpUrl
from qdrant_client import AsyncQdrantClient, models

from app.infra.db import create_engine, create_session_factory
from app.infra.manifest_repository import SqlManifestStore
from app.infra.model_files import load_pins
from app.infra.qdrant_index import QdrantIndex
from app.main import create_app
from tests.integration.conftest import QDRANT_URL, SettingsFactory, run_alembic
from tests.manifest_data import make_build

pytestmark = pytest.mark.integration

ALIAS = "it_ready_chunks"
PINS = load_pins()
# A build made with the real pinned files, as ingestion records it (ADR 0006).
BUILD = make_build(
    "20261113t140000z",
    dense=PINS["BAAI/bge-small-en-v1.5"],
    sparse=PINS["Qdrant/bm25"],
)
MODELS_OK = {
    "status": "ok",
    "info": {
        "dense": "BAAI/bge-small-en-v1.5",
        "sparse": "Qdrant/bm25",
        "reranker": "Xenova/ms-marco-MiniLM-L-6-v2",
    },
}


async def get(application: FastAPI, path: str, *, after_models: bool = False) -> httpx.Response:
    async with application.router.lifespan_context(application):
        if after_models:  # the real models load in the background (ADR 0008)
            await asyncio.wait_for(application.state.infrastructure.models_loaded, timeout=60)
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            return await http.get(path)


async def _drop_alias(client: AsyncQdrantClient) -> None:
    if any(a.alias_name == ALIAS for a in (await client.get_aliases()).aliases):
        await client.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=ALIAS))
            ]
        )
    if await client.collection_exists(BUILD.collection):
        await client.delete_collection(BUILD.collection)


@pytest.fixture
async def migrated_db(scratch_db: str, tmp_path: pathlib.Path) -> str:
    run_alembic(scratch_db, "upgrade", "head", cwd=tmp_path)
    return scratch_db


@pytest.fixture
async def live_index(migrated_db: str, make_settings: SettingsFactory) -> AsyncIterator[str]:
    """A 384-dimension collection behind the alias, recorded as a succeeded build."""
    engine = create_engine(make_settings(postgres_db=migrated_db))
    await SqlManifestStore(create_session_factory(engine)).record(BUILD)
    await engine.dispose()
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)
    await _drop_alias(client)
    index = QdrantIndex(client)
    await index.create_collection(BUILD.collection, 384)
    await index.point_alias(ALIAS, BUILD.collection)
    yield migrated_db
    await _drop_alias(client)
    await client.close()


@pytest.mark.model
async def test_ready_is_200_and_names_the_models_and_the_index(
    make_settings: SettingsFactory, live_index: str
) -> None:
    settings = make_settings(postgres_db=live_index, qdrant_collection_alias=ALIAS)

    response = await get(create_app(settings), "/ready", after_models=True)

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {
            "postgres": {"status": "ok"},
            "qdrant": {"status": "ok"},
            "models": MODELS_OK,
            "index": {
                "status": "ok",
                "info": {"collection": BUILD.collection, "build_id": BUILD.build_id},
            },
        },
        "llm": "stub",
    }


@pytest.mark.model
async def test_ready_says_the_index_is_missing_without_a_stack_trace(
    make_settings: SettingsFactory, migrated_db: str
) -> None:
    settings = make_settings(postgres_db=migrated_db, qdrant_collection_alias="it_no_such_alias")

    response = await get(create_app(settings), "/ready", after_models=True)

    assert response.status_code == 503
    assert response.json()["checks"]["index"] == {"status": "down", "detail": "missing"}
    assert response.json()["checks"]["models"] == MODELS_OK
    assert "Traceback" not in response.text


@pytest.fixture
def app_with_dependencies_down(make_settings: SettingsFactory, closed_port: int) -> FastAPI:
    return create_app(
        make_settings(
            postgres_port=closed_port,
            qdrant_url=HttpUrl(f"http://127.0.0.1:{closed_port}"),
        )
    )


async def test_app_boots_and_is_alive_with_dependencies_down(
    app_with_dependencies_down: FastAPI,
) -> None:
    response = await get(app_with_dependencies_down, "/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.model
async def test_ready_is_503_with_dependencies_down(app_with_dependencies_down: FastAPI) -> None:
    response = await get(app_with_dependencies_down, "/ready", after_models=True)

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {
            "postgres": {"status": "down", "detail": "unreachable"},
            "qdrant": {"status": "down", "detail": "unreachable"},
            "models": MODELS_OK,
            "index": {"status": "down", "detail": "unreachable"},
        },
        "llm": "stub",
    }

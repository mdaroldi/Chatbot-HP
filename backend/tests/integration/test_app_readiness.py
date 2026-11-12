"""The assembled app, booted through its lifespan, against real dependencies."""

import httpx
import pytest
from fastapi import FastAPI
from pydantic import HttpUrl

from app.main import create_app
from tests.integration.conftest import SettingsFactory

pytestmark = pytest.mark.integration


async def get(application: FastAPI, path: str) -> httpx.Response:
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            return await http.get(path)


async def test_ready_is_200_against_real_postgres_and_qdrant(
    make_settings: SettingsFactory,
) -> None:
    response = await get(create_app(make_settings()), "/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"postgres": {"status": "ok"}, "qdrant": {"status": "ok"}},
    }


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


async def test_ready_is_503_with_dependencies_down(app_with_dependencies_down: FastAPI) -> None:
    response = await get(app_with_dependencies_down, "/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {
            "postgres": {"status": "down", "detail": "unreachable"},
            "qdrant": {"status": "down", "detail": "unreachable"},
        },
    }

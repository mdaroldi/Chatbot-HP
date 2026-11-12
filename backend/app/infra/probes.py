"""Real DependencyProbe adapters for Postgres and Qdrant."""

from http import HTTPStatus

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.domain.health import DependencyUnavailableError


class PostgresProbe:
    """Ready when a pooled connection can run `SELECT 1`."""

    name = "postgres"

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def ping(self) -> None:
        async with self._engine.connect() as connection:
            await connection.execute(text("SELECT 1"))


class QdrantProbe:
    """Ready only when Qdrant's /readyz answers 200 (not just any HTTP response)."""

    name = "qdrant"

    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        self._client = client
        self._url = f"{base_url.rstrip('/')}/readyz"

    async def ping(self) -> None:
        try:
            response = await self._client.get(self._url)
        except httpx.TimeoutException as exc:
            # Report as a timeout, the same as hitting the readiness deadline.
            raise TimeoutError from exc
        if response.status_code != HTTPStatus.OK:
            raise DependencyUnavailableError(f"qdrant /readyz returned {response.status_code}")

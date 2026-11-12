"""Startup and shutdown of infrastructure clients, with cleanup on partial failure."""

from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

import httpx
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.settings import Settings
from app.domain.health import DependencyProbe
from app.infra.db import create_engine, create_session_factory
from app.infra.http import create_qdrant_http_client
from app.infra.probes import PostgresProbe, QdrantProbe


@dataclass(frozen=True)
class Infrastructure:
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    http_client: httpx.AsyncClient
    probes: tuple[DependencyProbe, ...]


@asynccontextmanager
async def open_infrastructure(
    settings: Settings,
    *,
    engine_factory: Callable[[Settings], AsyncEngine] = create_engine,
    http_client_factory: Callable[[Settings], httpx.AsyncClient] = create_qdrant_http_client,
) -> AsyncIterator[Infrastructure]:
    """Create clients without connecting, and close each one even if a later step fails.

    Nothing here talks to Postgres or Qdrant, so the app boots while they are down
    and reports that through /ready instead of crash-looping.
    """
    async with AsyncExitStack() as stack:
        engine = engine_factory(settings)
        stack.push_async_callback(engine.dispose)
        http_client = await stack.enter_async_context(http_client_factory(settings))
        yield Infrastructure(
            engine=engine,
            session_factory=create_session_factory(engine),
            http_client=http_client,
            probes=(PostgresProbe(engine), QdrantProbe(http_client, str(settings.qdrant_url))),
        )

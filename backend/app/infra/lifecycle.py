"""Startup and shutdown of infrastructure clients, with cleanup on partial failure."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

import httpx
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.settings import Settings
from app.domain.health import DependencyProbe
from app.infra.db import create_engine, create_session_factory
from app.infra.http import create_qdrant_http_client
from app.infra.inference import (
    CpuAllowance,
    cpu_allowance,
    create_inference_executor,
    warn_if_oversubscribed,
)
from app.infra.manifest_repository import SqlManifestStore
from app.infra.probes import PostgresProbe, QdrantProbe
from app.infra.qdrant_search import IndexProbe, create_qdrant_client, dense_dimension
from app.infra.query_models import ModelsProbe, QueryModels


@dataclass(frozen=True)
class Infrastructure:
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    http_client: httpx.AsyncClient
    probes: tuple[DependencyProbe, ...]
    executor: ThreadPoolExecutor
    models: QueryModels
    qdrant: AsyncQdrantClient
    models_loaded: "asyncio.Task[None]"  # the background load; done once models are ready


@asynccontextmanager
async def open_infrastructure(  # noqa: PLR0913 - settings plus keyword-only factories
    settings: Settings,
    *,
    engine_factory: Callable[[Settings], AsyncEngine] = create_engine,
    http_client_factory: Callable[[Settings], httpx.AsyncClient] = create_qdrant_http_client,
    models_factory: Callable[[Settings, Executor], QueryModels] = QueryModels.from_settings,
    cpu_allowance_factory: Callable[[], CpuAllowance] = cpu_allowance,
    qdrant_client_factory: Callable[[Settings], AsyncQdrantClient] = create_qdrant_client,
) -> AsyncIterator[Infrastructure]:
    """Create clients without connecting, and close each one even if a later step fails.

    Nothing here talks to Postgres or Qdrant, so the app boots while they are down
    and reports that through /ready instead of crash-looping. The query models load in
    a background task on the inference pool, so startup does not wait for them either.
    """
    async with AsyncExitStack() as stack:
        engine = engine_factory(settings)
        stack.push_async_callback(engine.dispose)
        session_factory = create_session_factory(engine)
        http_client = await stack.enter_async_context(http_client_factory(settings))
        executor = create_inference_executor(settings.inference_workers)
        # A model build in progress cannot be interrupted; do not wait for it on exit.
        stack.callback(executor.shutdown, wait=False, cancel_futures=True)
        warn_if_oversubscribed(
            settings.inference_workers, settings.inference_onnx_threads, cpu_allowance_factory()
        )
        models = models_factory(settings, executor)
        qdrant = qdrant_client_factory(settings)
        stack.push_async_callback(qdrant.close)
        index_probe = IndexProbe(
            qdrant,
            settings.qdrant_collection_alias,
            SqlManifestStore(session_factory),
            models,
            dense_size=dense_dimension(settings.embedding_dense_model),
        )
        # The API starts at once (/health green); /ready reports models: loading.
        loading = asyncio.create_task(models.load(), name="load-query-models")
        stack.push_async_callback(_cancel, loading)
        yield Infrastructure(
            engine=engine,
            session_factory=session_factory,
            http_client=http_client,
            probes=(
                PostgresProbe(engine),
                QdrantProbe(http_client, str(settings.qdrant_url)),
                ModelsProbe(models),
                index_probe,
            ),
            qdrant=qdrant,
            executor=executor,
            models=models,
            models_loaded=loading,
        )


async def _cancel(task: "asyncio.Task[None]") -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

"""The real adapters of the `ask` and `judge` commands, inside the backend image on the
compose networks. `ask`: the API at its service address, Qdrant for the alias and the
chunk payloads, Postgres for the manifest and the stored rows (read in READ ONLY
transactions). `judge`: the Message Batches API and Qdrant for the gold passages.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.settings import Settings
from app.domain.evaluation import IndexedChunk
from app.domain.ingestion import ManifestStore
from app.evaluation.answers.ask import ApiBackend, Backend, IndexRef, IndexResolver
from app.evaluation.answers.batch_api import AnthropicBatch, open_batch
from app.evaluation.answers.client import ChatClient, Timeouts
from app.evaluation.answers.excerpts import ChunkSource, QdrantChunkSource
from app.evaluation.answers.judge_run import BatchPort
from app.evaluation.answers.store import AnswerRows, SqlAnswerRows
from app.infra.db import create_engine, create_session_factory
from app.infra.manifest_repository import SqlManifestStore
from app.infra.qdrant_chunks import scroll_chunks
from app.infra.qdrant_index import QdrantIndex
from app.infra.qdrant_search import create_qdrant_client


@dataclass(frozen=True)
class AskRuntime:
    backend: Backend
    index: IndexResolver
    rows: AnswerRows
    chunks: Callable[[str], ChunkSource]  # by collection name


class ManifestIndexResolver:
    """The alias's live collection and the build the manifest says succeeded there:
    the same pair the backend's /ready reports, looked up independently.
    """

    def __init__(
        self, targets: Callable[[str], Awaitable[str | None]], manifest: ManifestStore
    ) -> None:
        self._targets, self._manifest = targets, manifest

    async def resolve(self, alias: str) -> IndexRef | None:
        collection = await self._targets(alias)
        if collection is None:
            return None
        build = await self._manifest.succeeded_build(collection)
        return None if build is None else IndexRef(collection, build.build_id)


@asynccontextmanager
async def open_ask_runtime(
    settings: Settings,
    api_url: str,
    *,
    qdrant_factory: Callable[[Settings], AsyncQdrantClient] = create_qdrant_client,
    engine_factory: Callable[[Settings], AsyncEngine] = create_engine,
) -> AsyncIterator[AskRuntime]:
    """Every handle is closed on exit, also when the run stops with an error."""
    qdrant = qdrant_factory(settings)
    engine = engine_factory(settings)
    try:
        sessions = create_session_factory(engine)
        async with httpx.AsyncClient(base_url=api_url) as http:
            chat = ChatClient(http, timeouts=Timeouts(), clock=time.monotonic, sleep=asyncio.sleep)

            def chunks(collection: str) -> ChunkSource:
                return QdrantChunkSource(qdrant, collection)

            yield AskRuntime(
                backend=ApiBackend(http, chat),
                index=ManifestIndexResolver(
                    QdrantIndex(qdrant).alias_target, SqlManifestStore(sessions)
                ),
                rows=SqlAnswerRows(sessions),
                chunks=chunks,
            )
    finally:
        await qdrant.close()
        await engine.dispose()


class MissingKeyError(Exception):
    """The judge calls the Anthropic API; without a key nothing is opened."""


@dataclass(frozen=True)
class JudgeRuntime:
    port: BatchPort
    chunks: Callable[[str], Awaitable[Sequence[IndexedChunk]]]  # every chunk, by alias


@asynccontextmanager
async def open_judge_runtime(
    settings: Settings,
    *,
    qdrant_factory: Callable[[Settings], AsyncQdrantClient] = create_qdrant_client,
    batch_factory: Callable[[SecretStr], AnthropicBatch] = open_batch,
) -> AsyncIterator[JudgeRuntime]:
    """Unlike `ask`, which never holds the key, the judge needs it: compose passes
    ANTHROPIC_API_KEY from .env to the one-off container. Both handles are closed on exit.
    """
    key = settings.anthropic_api_key
    if key is None:
        raise MissingKeyError("the judge needs ANTHROPIC_API_KEY")
    qdrant = qdrant_factory(settings)
    batch = batch_factory(key)
    try:

        async def chunks(alias: str) -> Sequence[IndexedChunk]:
            return await scroll_chunks(qdrant, alias)

        yield JudgeRuntime(port=batch, chunks=chunks)
    finally:
        await batch.close()
        await qdrant.close()

"""Stand-ins for the query models: a model source and model objects (no ONNX)."""

import asyncio
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.domain.ingestion import ModelNotAllowedError, SparseVector, VerifiedModel
from tests.model_data import pin

PINNED = {"demo/dense", "demo/bm25", "demo/rerank-a", "demo/rerank-b"}


@dataclass
class FakeSource:
    """ModelSource over a set of pinned names; records every fetch."""

    errors: dict[str, Exception] = field(default_factory=dict)
    fetched: list[str] = field(default_factory=list)

    async def fetch(self, name: str) -> VerifiedModel:
        self.fetched.append(name)
        if name in self.errors:
            raise self.errors[name]
        if name not in PINNED:
            raise ModelNotAllowedError(f"{name} has no pin", model=name)
        return VerifiedModel(pin(name=name, source=f"{name}-src"), Path(f"/m/{name}"))


@dataclass
class StandInEmbedder:
    dense: VerifiedModel
    sparse: VerifiedModel
    built_on: str = field(default_factory=lambda: threading.current_thread().name)
    queries: list[str] = field(default_factory=list)

    async def embed_query_dense(self, text: str) -> list[float]:
        self.queries.append(f"dense:{text}")
        return [1.0]

    async def embed_query_sparse(self, text: str) -> SparseVector:
        self.queries.append(f"sparse:{text}")
        return SparseVector((1,), (1.0,))


@dataclass
class StandInReranker:
    verified: VerifiedModel
    built_on: str = field(default_factory=lambda: threading.current_thread().name)
    calls: list[tuple[str, list[str]]] = field(default_factory=list)

    @property
    def model(self) -> str:
        return self.verified.files.name

    async def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        return [0.0 for _ in texts]


@dataclass
class Builds:
    embedders: list[StandInEmbedder] = field(default_factory=list)
    rerankers: list[StandInReranker] = field(default_factory=list)
    error: Exception | None = None

    def embedder(self, dense: VerifiedModel, sparse: VerifiedModel) -> StandInEmbedder:
        if self.error is not None:
            raise self.error
        built = StandInEmbedder(dense, sparse)
        self.embedders.append(built)
        return built

    def reranker(self, verified: VerifiedModel) -> StandInReranker:
        built = StandInReranker(verified)
        self.rerankers.append(built)
        return built


@dataclass
class BlockingSource(FakeSource):
    """A source whose fetch never returns, like a load still in progress."""

    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def fetch(self, name: str) -> VerifiedModel:
        self.fetched.append(name)
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

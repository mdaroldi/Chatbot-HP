"""The real adapters `ask` runs with: the alias-to-build resolver and the runtime that
opens and closes the API client, Qdrant and Postgres handles.
"""

from contextlib import nullcontext
from typing import cast

import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.settings import Settings
from app.domain.ingestion import IndexBuild, ManifestStore
from app.evaluation.answers.ask import ApiBackend, IndexRef
from app.evaluation.answers.excerpts import QdrantChunkSource
from app.evaluation.answers.runtime import ManifestIndexResolver, open_ask_runtime
from app.evaluation.answers.store import SqlAnswerRows


class FakeManifest:
    def __init__(self, builds: dict[str, str]) -> None:
        self.builds = builds

    async def record(self, build: IndexBuild) -> None:
        raise AssertionError(f"the resolver never writes {build}")

    async def succeeded_build(self, collection: str) -> IndexBuild | None:
        build_id = self.builds.get(collection)
        if build_id is None:
            return None
        return cast(IndexBuild, type("Build", (), {"build_id": build_id})())


def resolver(target: str | None, builds: dict[str, str]) -> ManifestIndexResolver:
    async def targets(alias: str) -> str | None:
        assert alias == "hpdocs_chunks"
        return target

    return ManifestIndexResolver(targets, cast(ManifestStore, FakeManifest(builds)))


async def test_the_alias_resolves_to_its_collection_and_succeeded_build() -> None:
    found = await resolver("c-1", {"c-1": "b-1"}).resolve("hpdocs_chunks")
    assert found == IndexRef("c-1", "b-1")


async def test_an_alias_with_no_collection_resolves_to_none() -> None:
    assert await resolver(None, {"c-1": "b-1"}).resolve("hpdocs_chunks") is None


async def test_a_collection_without_a_succeeded_build_resolves_to_none() -> None:
    assert await resolver("c-2", {"c-1": "b-1"}).resolve("hpdocs_chunks") is None


class Closable:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def dispose(self) -> None:
        self.closed = True


def closed(handle: Closable) -> bool:
    """Read through a call: mypy would keep the narrowing from inside the block."""
    return handle.closed


@pytest.mark.parametrize("fails", [False, True])
async def test_the_runtime_wires_the_adapters_and_closes_every_handle(fails: bool) -> None:
    qdrant, engine = Closable(), Closable()
    runtime_cm = open_ask_runtime(
        Settings(),
        "http://backend:8000",
        qdrant_factory=lambda _: cast(AsyncQdrantClient, qdrant),
        engine_factory=lambda _: cast(AsyncEngine, engine),
    )
    with pytest.raises(RuntimeError) if fails else nullcontext():
        async with runtime_cm as runtime:
            assert isinstance(runtime.backend, ApiBackend)
            assert isinstance(runtime.index, ManifestIndexResolver)
            assert isinstance(runtime.rows, SqlAnswerRows)
            assert isinstance(runtime.chunks("c-1"), QdrantChunkSource)
            assert not closed(qdrant)
            if fails:
                raise RuntimeError
    assert closed(qdrant)
    assert closed(engine)

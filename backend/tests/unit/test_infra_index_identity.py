"""Which index build answers a request: one alias lookup per call, build ids cached."""

import asyncio
from collections.abc import AsyncIterator
from typing import cast

import pytest
from qdrant_client import AsyncQdrantClient

from app.domain.ingestion import IndexBuild
from app.domain.retrieval import IndexBuildRef, IndexIdentity, RetrievalUnavailableError
from app.infra.qdrant_index import QdrantIndex
from app.infra.qdrant_search import QdrantIndexIdentity
from tests.index_data import DENSE_SIZE
from tests.manifest_data import make_build

pytestmark = pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect in the local Qdrant:UserWarning"
)

ALIAS = "unit_chunks"
OLD, NEW = make_build("20261113t120000z"), make_build("20261114t120000z")
PRIVATE_HOST = "qdrant-7.internal:6333"


class Manifest:
    """Succeeded builds by collection; counts lookups and can fail."""

    def __init__(self, *builds: IndexBuild, error: Exception | None = None) -> None:
        self.builds = {build.collection: build for build in builds}
        self.error = error
        self.lookups: list[str] = []

    async def record(self, build: IndexBuild) -> None:  # noqa: ARG002 - never called
        raise AssertionError("the identity never writes")

    async def succeeded_build(self, collection: str) -> IndexBuild | None:
        self.lookups.append(collection)
        if self.error is not None:
            raise self.error
        return self.builds.get(collection)


class BrokenClient:
    """Stands in for a Qdrant that fails or hangs on the alias lookup."""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error

    async def get_aliases(self) -> object:
        if self.error is not None:
            raise self.error
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.fixture
async def client() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(location=":memory:")
    yield client
    await client.close()


async def point(client: AsyncQdrantClient, build: IndexBuild) -> None:
    index = QdrantIndex(client)
    if not await client.collection_exists(build.collection):
        await index.create_collection(build.collection, DENSE_SIZE)
    await index.point_alias(ALIAS, build.collection)


def identity(client: AsyncQdrantClient | BrokenClient, manifest: Manifest) -> QdrantIndexIdentity:
    return QdrantIndexIdentity(
        cast(AsyncQdrantClient, client), ALIAS, manifest, timeout_seconds=0.05
    )


async def failure(subject: IndexIdentity) -> RetrievalUnavailableError:
    with pytest.raises(RetrievalUnavailableError) as caught:
        await subject.current()
    return caught.value


async def test_the_live_alias_names_the_collection_and_its_build(
    client: AsyncQdrantClient,
) -> None:
    await point(client, OLD)

    subject: IndexIdentity = identity(client, Manifest(OLD))

    assert await subject.current() == IndexBuildRef(OLD.collection, OLD.build_id)


async def test_a_build_id_is_looked_up_once_per_collection(client: AsyncQdrantClient) -> None:
    await point(client, OLD)
    manifest = Manifest(OLD)
    subject = identity(client, manifest)

    await subject.current()
    await subject.current()

    assert manifest.lookups == [OLD.collection]


async def test_the_alias_is_resolved_on_every_call_so_a_swap_is_seen(
    client: AsyncQdrantClient,
) -> None:
    await point(client, OLD)
    manifest = Manifest(OLD, NEW)
    subject = identity(client, manifest)
    await subject.current()

    await point(client, NEW)

    assert await subject.current() == IndexBuildRef(NEW.collection, NEW.build_id)
    assert manifest.lookups == [OLD.collection, NEW.collection]


async def test_no_alias_means_no_live_index(client: AsyncQdrantClient) -> None:
    error = await failure(identity(client, Manifest(OLD)))

    assert (error.stage, error.error_type) == ("index", "no_live_index")


async def test_a_collection_without_a_succeeded_build_is_not_cached(
    client: AsyncQdrantClient,
) -> None:
    await point(client, OLD)
    manifest = Manifest()
    subject = identity(client, manifest)

    error = await failure(subject)
    assert (error.stage, error.error_type) == ("index", "no_succeeded_build")

    manifest.builds[OLD.collection] = OLD
    assert await subject.current() == IndexBuildRef(OLD.collection, OLD.build_id)


async def test_a_qdrant_failure_names_only_its_type() -> None:
    error = await failure(
        identity(BrokenClient(error=ConnectionError(PRIVATE_HOST)), Manifest(OLD))
    )

    assert (error.stage, error.error_type) == ("index", "ConnectionError")
    assert PRIVATE_HOST not in str(error)


async def test_a_hanging_alias_lookup_hits_the_deadline() -> None:
    error = await failure(identity(BrokenClient(), Manifest(OLD)))

    assert (error.stage, error.error_type) == ("index", "TimeoutError")


async def test_a_manifest_failure_names_only_its_type(client: AsyncQdrantClient) -> None:
    await point(client, OLD)

    error = await failure(identity(client, Manifest(error=OSError(PRIVATE_HOST))))

    assert (error.stage, error.error_type) == ("index", "OSError")
    assert PRIVATE_HOST not in str(error)

"""The index compatibility probe on qdrant-client's local mode (ADRs 0006 and 0008)."""

import dataclasses
from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient, models

from app.domain.health import DependencyUnavailableError, FailureReason
from app.domain.ingestion import IndexBuild, ModelFiles
from app.domain.retrieval import ModelsNotReadyError
from app.infra.qdrant_index import QdrantIndex
from app.infra.qdrant_search import IndexProbe
from tests.index_data import DENSE_SIZE
from tests.manifest_data import make_build

pytestmark = pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect in the local Qdrant:UserWarning"
)

ALIAS = "unit_chunks"
BUILD = make_build("20261113t120000z")
COLLECTION = BUILD.collection


class Manifest:
    def __init__(self, *builds: IndexBuild) -> None:
        self.builds = {build.collection: build for build in builds}

    async def record(self, build: IndexBuild) -> None:  # noqa: ARG002 - never called
        raise AssertionError("the probe never writes")

    async def succeeded_build(self, collection: str) -> IndexBuild | None:
        return self.builds.get(collection)


class Models:
    """What the API verified at load; raises while the models are not loaded."""

    def __init__(self, dense: ModelFiles | None = BUILD.dense, sparse: ModelFiles = BUILD.sparse):
        self.dense, self.sparse = dense, sparse

    def verified_files(self) -> tuple[ModelFiles, ModelFiles]:
        if self.dense is None:
            raise ModelsNotReadyError("loading")
        return self.dense, self.sparse


@pytest.fixture
async def client() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(location=":memory:")
    yield client
    await client.close()


async def live(client: AsyncQdrantClient, dense_size: int = DENSE_SIZE) -> None:
    index = QdrantIndex(client)
    await index.create_collection(COLLECTION, dense_size)
    await index.point_alias(ALIAS, COLLECTION)


def probe(
    client: AsyncQdrantClient, manifest: Manifest | None = None, models_: Models | None = None
) -> IndexProbe:
    return IndexProbe(
        client,
        ALIAS,
        manifest if manifest is not None else Manifest(BUILD),
        models_ or Models(),
        dense_size=DENSE_SIZE,
    )


async def reason(subject: IndexProbe) -> FailureReason | None:
    with pytest.raises(DependencyUnavailableError) as caught:
        await subject.ping()
    return caught.value.reason


async def test_a_compatible_index_names_its_collection_and_build(client: AsyncQdrantClient) -> None:
    await live(client)

    subject = probe(client)

    assert subject.name == "index"
    assert await subject.ping() == {"collection": COLLECTION, "build_id": BUILD.build_id}


async def test_no_alias_means_the_index_is_missing(client: AsyncQdrantClient) -> None:
    assert await reason(probe(client)) is FailureReason.MISSING


async def test_a_collection_without_a_succeeded_build_row_is_missing(
    client: AsyncQdrantClient,
) -> None:
    await live(client)

    assert await reason(probe(client, manifest=Manifest())) is FailureReason.MISSING


async def test_another_dense_size_is_incompatible(client: AsyncQdrantClient) -> None:
    await live(client, dense_size=DENSE_SIZE + 1)

    assert await reason(probe(client)) is FailureReason.INCOMPATIBLE


@pytest.mark.parametrize(
    "sparse_config",
    [None, {"sparse": models.SparseVectorParams()}, {"bm25": models.SparseVectorParams()}],
)
async def test_a_sparse_vector_without_idf_or_by_another_name_is_incompatible(
    client: AsyncQdrantClient, sparse_config: dict[str, models.SparseVectorParams] | None
) -> None:
    await client.create_collection(
        COLLECTION,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_SIZE, distance=models.Distance.COSINE)
        },
        sparse_vectors_config=sparse_config,
    )
    await QdrantIndex(client).point_alias(ALIAS, COLLECTION)

    assert await reason(probe(client)) is FailureReason.INCOMPATIBLE


async def test_a_dense_vector_by_another_name_or_distance_is_incompatible(
    client: AsyncQdrantClient,
) -> None:
    await client.create_collection(
        COLLECTION,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_SIZE, distance=models.Distance.DOT)
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    await QdrantIndex(client).point_alias(ALIAS, COLLECTION)

    assert await reason(probe(client)) is FailureReason.INCOMPATIBLE


@pytest.mark.parametrize("which", ["dense", "sparse"])
async def test_model_files_other_than_the_ones_the_api_verified_are_incompatible(
    client: AsyncQdrantClient, which: str
) -> None:
    """Query vectors from other files than the index's are silently wrong (ADR 0006)."""
    await live(client)
    original = BUILD.dense if which == "dense" else BUILD.sparse
    other = dataclasses.replace(original, files={**original.files, "extra.bin": "f" * 64})
    verified = Models(dense=other) if which == "dense" else Models(sparse=other)

    assert await reason(probe(client, models_=verified)) is FailureReason.INCOMPATIBLE


async def test_the_index_waits_for_the_models_before_it_can_be_checked(
    client: AsyncQdrantClient,
) -> None:
    await live(client)

    assert await reason(probe(client, models_=Models(dense=None))) is FailureReason.LOADING

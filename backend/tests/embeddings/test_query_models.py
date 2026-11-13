"""The query side with the real pinned models (ADRs 0008 and 0009): no network, no index."""

import math
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pytest

from app.core.settings import Settings
from app.infra.fastembed_embedder import FastembedEmbedder
from app.infra.fastembed_reranker import FastembedReranker
from app.infra.inference import create_inference_executor
from app.infra.model_files import load_pins
from app.infra.query_models import ModelsState, QueryModels
from tests.downloads import fetch_model, fetch_or_fail
from tests.model_data import fetch_verified

pytestmark = pytest.mark.model

RERANKERS = ["Xenova/ms-marco-MiniLM-L-6-v2", "jinaai/jina-reranker-v1-turbo-en"]
QUESTION = "How do I replace the ink cartridge?"
RELEVANT = "Replace the cartridges: open the cartridge access door and remove the old cartridge."
UNRELATED = "The laptop fan speed is controlled by the BIOS thermal policy."


@pytest.fixture(scope="module")
def executor() -> Iterator[ThreadPoolExecutor]:
    pool = create_inference_executor(2)
    yield pool
    pool.shutdown()


@pytest.fixture(scope="module")
def query_embedder(model_settings: Settings, executor: ThreadPoolExecutor) -> FastembedEmbedder:
    return FastembedEmbedder.for_queries(
        model_settings,
        fetch_verified(model_settings, model_settings.embedding_dense_model),
        fetch_verified(model_settings, model_settings.embedding_sparse_model),
        executor,
    )


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


async def test_a_dense_query_vector_has_the_index_dimension(
    query_embedder: FastembedEmbedder,
) -> None:
    assert len(await query_embedder.embed_query_dense(QUESTION)) == 384


async def test_dense_query_and_document_encodings_of_one_text_are_the_same(
    query_embedder: FastembedEmbedder,
) -> None:
    """fastembed adds no BGE query instruction for bge-small (ADR 0009, Q2)."""
    [document] = await query_embedder.embed_dense([QUESTION])

    assert cosine(await query_embedder.embed_query_dense(QUESTION), document) > 0.9999


async def test_a_bm25_query_has_unit_weights_on_the_documents_terms(
    query_embedder: FastembedEmbedder,
) -> None:
    query = await query_embedder.embed_query_sparse("replace cartridges")
    [document] = await query_embedder.embed_sparse(["replacing the cartridge"])

    assert set(query.values) == {1.0}
    assert set(query.indices) == set(document.indices)  # same stemmer and stop words
    assert all(value != 1.0 for value in document.values)  # documents carry BM25 weights


@pytest.mark.parametrize("name", RERANKERS)
async def test_each_reranker_scores_the_relevant_text_higher(
    model_settings: Settings, executor: ThreadPoolExecutor, name: str
) -> None:
    verified = await fetch_or_fail(lambda: fetch_model(model_settings, name))
    reranker = FastembedReranker.from_settings(model_settings, verified, executor)

    relevant, unrelated = await reranker.rerank(QUESTION, [RELEVANT, UNRELATED])

    assert relevant > unrelated


async def test_the_api_loads_its_models_from_the_verified_store_without_the_hub(
    model_settings: Settings, executor: ThreadPoolExecutor
) -> None:
    for name in load_pins():  # fills the store for the offline loader
        await fetch_or_fail(partial(fetch_model, model_settings, name))
    models = QueryModels.from_settings(model_settings, executor)

    await models.load()

    assert models.state is ModelsState.READY, models.error_type
    assert (await models.reranker()).model == "Xenova/ms-marco-MiniLM-L-6-v2"
    second = await models.reranker("jinaai/jina-reranker-v1-turbo-en")
    assert second.model == "jinaai/jina-reranker-v1-turbo-en"

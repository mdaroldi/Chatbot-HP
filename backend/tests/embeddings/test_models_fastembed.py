"""The adapters with the real fastembed models: shapes, counts and basic sanity."""

import math

import pytest

from app.infra.fastembed_embedder import FastembedEmbedder
from app.infra.fastembed_tokens import FastembedTokenCounter

pytestmark = pytest.mark.model

DENSE_SIZE = 384  # BAAI/bge-small-en-v1.5
MODEL_INPUT_LIMIT = 512


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))  # the vectors are unit length


def test_token_counts_include_cls_and_sep(token_counter: FastembedTokenCounter) -> None:
    assert token_counter.count("") == 2
    assert token_counter.count("hello world") == 4
    assert token_counter.count("Press the [icon] button.") == 9


def test_token_counts_saturate_at_the_model_input_limit(
    token_counter: FastembedTokenCounter,
) -> None:
    exact = [token_counter.count("word " * n) for n in (509, 510)]
    saturated = [token_counter.count("word " * n) for n in (511, 1000)]

    assert exact == [511, MODEL_INPUT_LIMIT]
    assert saturated == [MODEL_INPUT_LIMIT, MODEL_INPUT_LIMIT]


async def test_dense_vectors_are_unit_length_and_rank_related_text_higher(
    embedder: FastembedEmbedder,
) -> None:
    query, related, unrelated = await embedder.embed_dense(
        ["How do I load paper?", "Load paper into the input tray.", "The battery is charging."]
    )

    assert {len(query), len(related), len(unrelated)} == {DENSE_SIZE}
    assert all(math.isclose(math.hypot(*v), 1.0, abs_tol=1e-3) for v in (query, related))
    assert cosine(query, related) > cosine(query, unrelated)


async def test_bm25_weights_content_words_and_ignores_stopwords(
    embedder: FastembedEmbedder,
) -> None:
    content, stopwords = await embedder.embed_sparse(
        ["Load paper into the input tray.", "the and of"]
    )

    assert len(content.indices) == 4  # load, paper, input, tray
    assert all(value > 0 for value in content.values)
    assert stopwords.indices == ()
    assert (embedder.dense_model, embedder.sparse_model) == (
        "BAAI/bge-small-en-v1.5",
        "Qdrant/bm25",
    )

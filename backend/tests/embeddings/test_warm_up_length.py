"""The API's warm-up inputs fill each model's 512-token input (Phase 2, commit 10c).

A short warm-up left the first full-length request to size ONNX's buffers: 10.1 s
against about 2 s steady. These tests count the warm-up with each pinned tokenizer,
truncation included, so a model with a different vocabulary cannot fall short.
"""

import pytest
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.core.settings import Settings
from app.infra.model_files import load_pins
from app.infra.query_models import WARM_UP_DOCUMENT, WARM_UP_QUERY
from tests.model_data import fetch_verified

pytestmark = pytest.mark.model

TOKEN_LIMIT = 512  # bge-small and the reranker; fastembed truncates there (ADR 0008)
RERANKERS = ["Xenova/ms-marco-MiniLM-L-6-v2"]


def test_every_pinned_reranker_is_covered() -> None:
    pinned = {model.name for model in load_pins().values()}

    assert pinned - {"BAAI/bge-small-en-v1.5", "Qdrant/bm25"} == set(RERANKERS)


@pytest.mark.parametrize("name", RERANKERS)
def test_the_warm_up_pair_reaches_the_rerankers_token_limit(
    model_settings: Settings, name: str
) -> None:
    verified = fetch_verified(model_settings, name)
    encoder = TextCrossEncoder(
        name,
        cache_dir=str(model_settings.embedding_cache_dir),
        specific_model_path=str(verified.path),
        local_files_only=True,
    )

    # fastembed's ONNX models hold a tokenizers.Tokenizer; the base type does not declare it.
    tokenizer = encoder.model.tokenizer  # type: ignore[attr-defined]
    pair = tokenizer.encode(WARM_UP_QUERY, WARM_UP_DOCUMENT)

    assert len(pair.ids) == TOKEN_LIMIT


def test_the_dense_warm_up_reaches_the_token_limit(model_settings: Settings) -> None:
    name = model_settings.embedding_dense_model
    verified = fetch_verified(model_settings, name)
    model = TextEmbedding(
        name,
        cache_dir=str(model_settings.embedding_cache_dir),
        specific_model_path=str(verified.path),
        local_files_only=True,
    )

    tokenizer = model.model.tokenizer  # type: ignore[attr-defined]  # as above
    assert len(tokenizer.encode(WARM_UP_DOCUMENT).ids) == TOKEN_LIMIT

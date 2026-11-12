"""Model fixtures: the real pinned models, verified in EMBEDDING_CACHE_DIR (ADR 0006)."""

import pytest
from pydantic import SecretStr

from app.core.settings import Settings
from app.infra.fastembed_embedder import FastembedEmbedder
from app.infra.fastembed_tokens import FastembedTokenCounter
from tests.model_data import fetch_verified


@pytest.fixture(scope="session")
def model_settings() -> Settings:
    # The password is never used: model tests touch no database.
    return Settings(postgres_password=SecretStr("unused-by-model-tests"))


@pytest.fixture(scope="session")
def token_counter(model_settings: Settings) -> FastembedTokenCounter:
    dense = fetch_verified(model_settings, model_settings.embedding_dense_model)
    return FastembedTokenCounter.from_settings(model_settings, dense)


@pytest.fixture(scope="session")
def embedder(model_settings: Settings) -> FastembedEmbedder:
    return FastembedEmbedder.from_settings(
        model_settings,
        fetch_verified(model_settings, model_settings.embedding_dense_model),
        fetch_verified(model_settings, model_settings.embedding_sparse_model),
    )

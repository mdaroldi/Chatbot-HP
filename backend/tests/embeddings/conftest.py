"""Model fixtures: the real fastembed models, downloaded into EMBEDDING_CACHE_DIR once."""

import pytest
from pydantic import SecretStr

from app.core.settings import Settings
from app.infra.fastembed_embedder import FastembedEmbedder
from app.infra.fastembed_tokens import FastembedTokenCounter


@pytest.fixture(scope="session")
def model_settings() -> Settings:
    # The password is never used: model tests touch no database.
    return Settings(postgres_password=SecretStr("unused-by-model-tests"))


@pytest.fixture(scope="session")
def token_counter(model_settings: Settings) -> FastembedTokenCounter:
    return FastembedTokenCounter.from_settings(model_settings)


@pytest.fixture(scope="session")
def embedder(model_settings: Settings) -> FastembedEmbedder:
    return FastembedEmbedder.from_settings(model_settings)

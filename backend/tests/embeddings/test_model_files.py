"""The real pinned models (ADR 0006): each pin downloads as pinned and loads with no network.

These download into a fresh cache on every run on purpose (the restored CI cache is not
used), because only a real download shows that the pins still match the hub. A hub
failure ends a test as MODEL UNAVAILABLE, never as an assertion error.
"""

from pathlib import Path

import pytest
from fastembed.common import model_management

from app.core.settings import Settings
from app.domain.ingestion import ModelFiles
from app.infra.fastembed_embedder import FastembedEmbedder
from app.infra.fastembed_tokens import FastembedTokenCounter
from app.infra.model_files import HuggingFaceHub, PinnedModelSource, load_pins, snapshot_dir
from tests.downloads import fetch_or_fail

pytestmark = pytest.mark.model

PINS = load_pins()


@pytest.fixture(scope="module")
def fresh_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("model-cache")


@pytest.mark.parametrize("pin", PINS.values(), ids=list(PINS))
async def test_each_pinned_revision_downloads_with_the_pinned_digests(
    pin: ModelFiles, fresh_cache: Path
) -> None:
    source = PinnedModelSource(PINS, fresh_cache, HuggingFaceHub())
    verified = await fetch_or_fail(lambda: source.fetch(pin.name))

    assert verified.files == pin
    # The hub cache layout is the one the next run looks in, so it needs no network.
    assert verified.path == snapshot_dir(fresh_cache, pin.source, pin.revision)


def no_network(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("fastembed tried to reach the hub")


async def test_verified_models_load_and_run_without_the_hub(
    fresh_cache: Path, model_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = PinnedModelSource(PINS, fresh_cache, HuggingFaceHub())
    dense = await fetch_or_fail(lambda: source.fetch(model_settings.embedding_dense_model))
    sparse = await fetch_or_fail(lambda: source.fetch(model_settings.embedding_sparse_model))
    for name in ("snapshot_download", "model_info", "list_repo_tree"):
        monkeypatch.setattr(model_management, name, no_network)

    embedder = FastembedEmbedder(dense, sparse, fresh_cache, batch_size=2)
    counter = FastembedTokenCounter(dense, fresh_cache)

    [vector] = await embedder.embed_dense(["hello world"])
    [sparse_vector] = await embedder.embed_sparse(["printer cartridge"])
    assert len(vector) == 384
    assert sparse_vector.indices
    assert counter.count("hello world") == 4

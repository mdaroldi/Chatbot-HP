"""The ingest CLI: modes, exit codes, the one-line model failure, and the wiring."""

import asyncio
import logging
import pathlib
import runpy
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any, ClassVar

import pytest
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient

from app.core.settings import Settings, get_settings
from app.domain.ingestion import (
    CorpusDownloadError,
    ModelDownloadError,
    ModelIntegrityError,
    ModelNotAllowedError,
    ModelPinsError,
    VerifiedModel,
)
from app.infra.corpus_http import HttpCorpusSource
from app.infra.fastembed_embedder import FastembedEmbedder
from app.infra.manifest_repository import SqlManifestStore
from app.infra.model_files import PinnedModelSource
from app.infra.qdrant_index import QdrantIndex
from app.ingestion import cli
from app.ingestion.chunking import VERSION, StructureChunker
from app.ingestion.cli import Runtime, main, open_runtime
from app.ingestion.pipeline import StructureProcessor
from tests.unit.ingest_fakes import DENSE, DOCUMENTS, PINS, SPARSE
from tests.unit.test_infra_fastembed import FakeDense, FakeSparse
from tests.unit.test_ingestion_pipeline import CONFIG, World

pytestmark = pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect in the local Qdrant:UserWarning"
)

SECRET_DIGEST = "d" * 64


@pytest.fixture
def world(tmp_path: pathlib.Path) -> World:
    # Sync on purpose: main() runs its own event loop. Local mode holds no sockets.
    world = World(AsyncQdrantClient(location=":memory:"), tmp_path)
    world.version = VERSION  # the CLI configures the real chunker's version
    return world


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "unit-test-only")
    monkeypatch.setenv("EMBEDDING_DENSE_MODEL", DENSE.name)
    monkeypatch.setenv("EMBEDDING_SPARSE_MODEL", SPARSE.name)
    monkeypatch.setattr(cli, "configure_logging", lambda *_, **__: None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def runtime_of(world: World) -> Callable[[Settings], Any]:
    """A runtime factory over the world's fakes (Any: an async context manager)."""

    @asynccontextmanager
    async def factory(settings: Settings) -> AsyncIterator[Runtime]:
        assert settings.qdrant_collection_alias == CONFIG.alias
        yield Runtime(documents=DOCUMENTS, pins=PINS, ports=world.ports())

    return factory


def errors(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


# ------------------------------------------------------------------ modes


@pytest.mark.parametrize(
    "argv", [[], ["--if-needed", "--force"], ["--force", "--inspect-chunks", "x.md"], ["--bogus"]]
)
def test_exactly_one_mode_is_required(argv: list[str], world: World) -> None:
    with pytest.raises(SystemExit) as exited:
        main(argv, runtime=runtime_of(world))

    assert exited.value.code == 2


def test_if_needed_builds_once_then_skips(world: World) -> None:
    assert main(["--if-needed"], runtime=runtime_of(world)) == 0
    world.later()
    assert main(["--if-needed"], runtime=runtime_of(world)) == 0

    assert len(world.manifest.builds) == 1


def test_force_always_builds(world: World) -> None:
    assert main(["--force"], runtime=runtime_of(world)) == 0
    world.later()
    assert main(["--force"], runtime=runtime_of(world)) == 0

    assert len(world.manifest.builds) == 2


def test_the_index_settings_reach_the_pipeline(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INDEX_KEEP_PREVIOUS", "false")
    for _ in range(2):
        assert main(["--force"], runtime=runtime_of(world)) == 0
        world.later()

    collections = asyncio.run(world.index.list_collections("hpdocs_chunks_"))
    assert collections == [world.manifest.builds[-1].collection]  # no previous build kept


# ------------------------------------------------------------------ failures


@pytest.mark.parametrize(
    ("error", "file"),
    [
        (
            ModelIntegrityError(DENSE.name, "model.onnx", expected="c" * 64, actual=SECRET_DIGEST),
            "model.onnx",
        ),
        (
            ModelDownloadError(f"{DENSE.name}: cannot download {SECRET_DIGEST}", model=DENSE.name),
            None,
        ),
    ],
)
def test_a_model_failure_logs_one_line_and_exits_non_zero(
    world: World, caplog: pytest.LogCaptureFixture, error: Exception, file: str | None
) -> None:
    world.models.error = error

    assert main(["--if-needed"], runtime=runtime_of(world)) == 1

    [record] = errors(caplog)
    assert record.getMessage() == "embedding model check failed"
    assert (record.__dict__["model"], record.__dict__["file"]) == (DENSE.name, file)
    assert record.__dict__["error_type"] == type(error).__name__
    assert record.exc_info is None
    assert SECRET_DIGEST not in " ".join(str(v) for v in record.__dict__.values())
    assert world.manifest.builds == []


@pytest.mark.parametrize(
    ("error", "model"),
    [
        (ModelNotAllowedError("demo/dense has no pin", model="demo/dense"), "demo/dense"),
        (ModelPinsError("duplicate name: demo/dense", model="demo/dense"), "demo/dense"),
        (ModelPinsError("model pins need at least one model"), None),
    ],
)
def test_a_model_without_a_valid_pin_is_a_model_failure_naming_the_model(
    world: World, caplog: pytest.LogCaptureFixture, error: Exception, model: str | None
) -> None:
    world.models.error = error

    assert main(["--force"], runtime=runtime_of(world)) == 1

    [record] = errors(caplog)
    assert (record.getMessage(), record.__dict__["error_type"]) == (
        "embedding model check failed",
        type(error).__name__,
    )
    assert (record.__dict__["model"], record.__dict__["file"]) == (model, None)


def test_a_failed_build_logs_its_error_type_and_exits_non_zero(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    world.corpus.failing = {"demo-guide": CorpusDownloadError("demo-guide: HTTP 503 from host")}

    assert main(["--force"], runtime=runtime_of(world)) == 1

    [record] = errors(caplog)
    assert record.getMessage() == "ingestion failed"
    assert record.__dict__["error_type"] == "CorpusDownloadError"
    assert "HTTP 503" not in record.getMessage()
    assert world.manifest.builds[-1].error_type == "CorpusDownloadError"


# ------------------------------------------------------------------ inspect-chunks


def test_inspect_chunks_writes_the_report_without_index_or_manifest(
    world: World, tmp_path: pathlib.Path
) -> None:
    out = tmp_path / "review" / "chunks.md"

    assert main(["--inspect-chunks", str(out)], runtime=runtime_of(world)) == 0

    text = out.read_text()
    assert text.startswith("# Chunk review")
    assert "| demo-manual | 4 | 2 |" in text
    assert world.calls.of("models") == [DENSE.name]  # the tokenizer only
    assert {kind for kind, _ in world.calls.log} == {"models", "build", "corpus", "process"}
    assert world.calls.of("build") == ["processor"]
    assert asyncio.run(world.index.alias_target(CONFIG.alias)) is None


# ------------------------------------------------------------------ wiring


class FakeQdrantClient:
    built: ClassVar[list["FakeQdrantClient"]] = []

    def __init__(self, *, url: str, check_compatibility: bool) -> None:
        self.url, self.check_compatibility, self.closed = url, check_compatibility, False
        FakeQdrantClient.built.append(self)

    async def close(self) -> None:
        self.closed = True


async def test_the_runtime_wires_the_real_adapters_from_settings(tmp_path: pathlib.Path) -> None:
    settings = Settings(
        postgres_password=SecretStr("unit-test-only"),
        corpus_dir=tmp_path / "corpus",
        embedding_cache_dir=tmp_path / "models",
        chunk_max_tokens=300,
        chunk_min_tokens=40,
    )
    FakeQdrantClient.built, FakeDense.built, FakeSparse.built = [], [], []
    dense = VerifiedModel(files=DENSE, path=tmp_path / "dense")
    sparse = VerifiedModel(files=SPARSE, path=tmp_path / "sparse")

    async with open_runtime(
        settings,
        qdrant_class=FakeQdrantClient,  # type: ignore[arg-type]  # only url and close()
        dense_class=FakeDense,
        sparse_class=FakeSparse,
        token_class=FakeDense,
    ) as runtime:
        ports = runtime.ports
        assert [d.doc_id for d in runtime.documents] == [
            "envy-6000-user-guide",
            "omen-17-service-guide",
        ]
        assert set(runtime.pins) == {"BAAI/bge-small-en-v1.5", "Qdrant/bm25"}
        assert isinstance(ports.corpus, HttpCorpusSource)
        assert ports.corpus.corpus_dir == tmp_path / "corpus"
        assert isinstance(ports.models, PinnedModelSource)
        assert isinstance(ports.index, QdrantIndex)
        assert isinstance(ports.manifest, SqlManifestStore)
        processor = ports.processor(dense)
        assert isinstance(processor, StructureProcessor)
        assert isinstance(processor.chunker, StructureChunker)
        assert (processor.chunker.max_tokens, processor.chunker.min_tokens) == (300, 40)
        assert isinstance(ports.embedder(dense, sparse), FastembedEmbedder)
        assert ports.clock().utcoffset() is not None
        [client] = FakeQdrantClient.built
        assert (client.url, client.check_compatibility) == ("http://qdrant:6333/", False)
        assert not client.closed

    assert FakeQdrantClient.built[0].closed  # closed with the runtime
    assert isinstance(ports.corpus, HttpCorpusSource)
    with pytest.raises(RuntimeError):  # the HTTP client is closed with the runtime
        await ports.corpus._client.get("https://example.invalid")


def test_main_uses_the_real_runtime_by_default() -> None:
    assert cli.main.__kwdefaults__ == {"runtime": open_runtime}


def test_running_the_package_calls_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "main", lambda: 7)

    with pytest.raises(SystemExit) as exited:
        runpy.run_module("app.ingestion", run_name="__main__")

    assert exited.value.code == 7

"""In-memory fakes of the ingestion ports, recording every call the pipeline makes."""

import pathlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from app.domain.ingestion import (
    CorpusDocument,
    IndexBuild,
    ModelFiles,
    SparseVector,
    VerifiedModel,
)
from app.ingestion.cleaning import CleaningStats
from app.ingestion.pipeline import ProcessedDocument
from tests.index_data import make_batch
from tests.manifest_data import REVISION

DENSE = ModelFiles("demo/dense", "demo/dense-onnx", REVISION, {"model.onnx": "c" * 64})
SPARSE = ModelFiles("demo/sparse", "demo/sparse", REVISION, {"x.txt": "e" * 64})
PINS = {DENSE.name: DENSE, SPARSE.name: SPARSE}
DOCUMENTS = (
    CorpusDocument(
        "demo-guide", "Demo Guide", "Demo Printer 100", "https://d.test/g.pdf", "a" * 64
    ),
    CorpusDocument(
        "demo-manual", "Demo Manual", "Demo Laptop 17", "https://d.test/m.pdf", "b" * 64
    ),
)
CLOCK = datetime(2026, 11, 12, 14, 30, 5, tzinfo=UTC)


@dataclass
class Calls:
    """Everything the pipeline asked for, in order: `("corpus", "demo-guide")`, ..."""

    log: list[tuple[str, str]] = field(default_factory=list)

    def of(self, kind: str) -> list[str]:
        return [name for k, name in self.log if k == kind]


class FakeCorpus:
    def __init__(
        self, calls: Calls, folder: pathlib.Path, failing: Mapping[str, Exception] | None = None
    ) -> None:
        self.calls, self.folder, self.failing = calls, folder, dict(failing or {})

    async def fetch(self, document: CorpusDocument) -> pathlib.Path:
        self.calls.log.append(("corpus", document.doc_id))
        if document.doc_id in self.failing:
            raise self.failing[document.doc_id]
        path = self.folder / f"{document.doc_id}.pdf"
        path.write_bytes(b"%PDF-1.7 " + document.doc_id.encode())
        return path


class FakeModels:
    def __init__(self, calls: Calls, folder: pathlib.Path, error: Exception | None = None) -> None:
        self.calls, self.folder, self.error = calls, folder, error

    async def fetch(self, name: str) -> VerifiedModel:
        self.calls.log.append(("models", name))
        if self.error is not None:
            raise self.error
        return VerifiedModel(files=PINS[name], path=self.folder / name)


STATS = CleaningStats(lines={}, pages={}, empty_sections=0)


class FakeProcessor:
    """Turns each document into `counts[doc_id]` synthetic chunks."""

    def __init__(self, calls: Calls, counts: Mapping[str, int], version: str = "v1") -> None:
        self.calls, self.counts, self.version = calls, counts, version

    def process(self, path: pathlib.Path, document: CorpusDocument) -> ProcessedDocument:
        self.calls.log.append(("process", document.doc_id))
        assert path.read_bytes().startswith(b"%PDF-")
        order = list(self.counts)
        first = sum(self.counts[d] for d in order[: order.index(document.doc_id)])
        batch = make_batch(self.counts[document.doc_id], doc_id=document.doc_id, first=first)
        chunks = tuple(replace(c, chunking_version=self.version) for c in batch.chunks)
        return ProcessedDocument(document=document, pages=4, chunks=chunks, stats=STATS)


class FakeEmbedder:
    def __init__(self, calls: Calls, *, short_by: int = 0) -> None:
        self.calls, self.short_by = calls, short_by
        self.texts: list[str] = []

    @property
    def dense_model(self) -> str:
        return DENSE.name

    @property
    def sparse_model(self) -> str:
        return SPARSE.name

    async def embed_dense(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.log.append(("embed", "dense"))
        self.texts = list(texts)
        return [[1.0, float(n), 0.5, 0.25] for n in range(len(texts) - self.short_by)]

    async def embed_sparse(self, texts: Sequence[str]) -> list[SparseVector]:
        self.calls.log.append(("embed", "sparse"))
        return [SparseVector(indices=(n,), values=(1.0,)) for n in range(len(texts))]


class FakeLengths:
    """Lengths 1, 4, 9, ... by position, so a mean differs from a median, first or last."""

    def __init__(self, calls: Calls, error: Exception | None = None) -> None:
        self.calls, self.error = calls, error
        self.texts: list[str] = []

    def lengths(self, texts: Sequence[str]) -> list[int]:
        self.calls.log.append(("measure", "bm25"))
        if self.error is not None:
            raise self.error
        self.texts = list(texts)
        return [(n + 1) ** 2 for n in range(len(texts))]


class InMemoryManifest:
    def __init__(self, calls: Calls) -> None:
        self.calls = calls
        self.builds: list[IndexBuild] = []

    async def record(self, build: IndexBuild) -> None:
        self.calls.log.append(("record", build.status.value))
        self.builds.append(build)

    async def succeeded_build(self, collection: str) -> IndexBuild | None:
        self.calls.log.append(("manifest", collection))
        return next(
            (b for b in self.builds if b.collection == collection and b.error_type is None), None
        )

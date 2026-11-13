"""The ingest run: check, verify models, build, publish, record (Phase 1 plan, item 1).

Order matters and is tested:

1. Unless forced, the live build is compared with the current inputs (`freshness`).
   A match ends the run before any download, hash or model load.
2. Both embedding models are verified. A model failure ends the run before a build
   ID exists, so it leaves no collection and no manifest row (ADR 0006).
3. The build starts: every document is fetched, parsed and chunked; the BM25 avg_len
   is fixed or, for `auto`, measured on the chunks; the chunks are embedded and
   published. A failure from here on is recorded as a failed build,
   with its exception type only, and raised again.
4. A published build is recorded as succeeded, with any pruning warnings.

Logs name documents, sizes, counts and outcomes, never document or chunk text.
"""

import asyncio
import dataclasses
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from app.domain.ingestion import (
    AvgLenSetting,
    Bm25Params,
    BuildStatus,
    BuildWarning,
    Chunk,
    Chunker,
    CorpusDocument,
    CorpusSource,
    DocumentParser,
    EmbeddedChunks,
    Embedder,
    IndexBuild,
    ManifestStore,
    ModelFiles,
    ModelSource,
    SparseLengths,
    VectorIndex,
    VerifiedModel,
    mean_length,
)
from app.ingestion.cleaning import CleaningStats, clean
from app.ingestion.freshness import Expected, StaleReason, stale_reason
from app.ingestion.lines import group_lines
from app.ingestion.publish import collection_name, publish
from app.ingestion.sections import build_sections

logger = logging.getLogger(__name__)

_BUILD_ID_FORMAT = "%Y%m%dt%H%M%Sz"  # 20261112t143005z: sorts by time, valid in names


@dataclass(frozen=True)
class ProcessedDocument:
    """One document's chunks and the statistics of how they were made."""

    document: CorpusDocument
    pages: int
    chunks: tuple[Chunk, ...]
    stats: CleaningStats

    def drop_stats(self) -> dict[str, int]:
        """Flat counters for the manifest's `drop_stats` column."""
        return {
            **{f"lines_{reason.value}": n for reason, n in self.stats.lines.items()},
            **{f"pages_{reason.value}": n for reason, n in self.stats.pages.items()},
            "sections_empty": self.stats.empty_sections,
        }


class DocumentProcessor(Protocol):
    def process(self, path: Path, document: CorpusDocument) -> ProcessedDocument: ...


class StructureProcessor:
    """Parse, clean, section and chunk one PDF. CPU-bound: run it in a worker thread."""

    def __init__(self, parser: DocumentParser, chunker: Chunker) -> None:
        self.parser, self.chunker = parser, chunker

    def process(self, path: Path, document: CorpusDocument) -> ProcessedDocument:
        parsed = self.parser.parse(path, document)
        lines = group_lines(parsed.words, parsed.images)
        cleaned = clean(lines, parsed.outline, parsed.page_heights, parsed.rules)
        sectioned = build_sections(document.doc_id, cleaned, parsed.outline, parsed.rules)
        chunks = self.chunker.chunk(sectioned.sections, document)
        return ProcessedDocument(document, parsed.page_count, tuple(chunks), cleaned.stats)


@dataclass(frozen=True)
class IngestConfig:
    alias: str
    keep_previous: bool
    dense_model: str
    sparse_model: str
    chunking_version: str
    bm25_avg_len: AvgLenSetting


@dataclass(frozen=True)
class IngestPorts:
    """The run's collaborators. The processor and embedder are built from verified
    models, so they are factories: the skip path never builds either.
    """

    corpus: CorpusSource
    models: ModelSource
    index: VectorIndex
    manifest: ManifestStore
    processor: Callable[[VerifiedModel], DocumentProcessor]
    embedder: Callable[[VerifiedModel, VerifiedModel, Bm25Params], Embedder]
    sparse_lengths: Callable[[VerifiedModel], SparseLengths]
    clock: Callable[[], datetime]


@dataclass(frozen=True)
class IngestResult:
    skipped: bool
    collection: str
    build_id: str
    reason: StaleReason | None  # why it rebuilt; None when skipped or forced
    warnings: tuple[BuildWarning, ...] = ()


@dataclass
class _Progress:
    """What a build had done when it stopped, for its manifest row."""

    chunk_counts: dict[str, int]
    drop_stats: dict[str, dict[str, int]]
    sparse_params: Bm25Params | None = None


async def _live_build(
    documents: Sequence[CorpusDocument],
    pins: Mapping[str, ModelFiles],
    config: IngestConfig,
    ports: IngestPorts,
) -> tuple[IndexBuild | None, StaleReason | None]:
    """The live build when it can be reused, else the reason to rebuild."""
    target = await ports.index.alias_target(config.alias)
    if target is None:
        return None, StaleReason.NO_LIVE_INDEX
    live = await ports.manifest.succeeded_build(target)
    if live is None:
        return None, StaleReason.NO_MANIFEST
    expected = Expected(
        document_hashes={d.doc_id: d.sha256 for d in documents},
        chunking_version=config.chunking_version,
        dense_model=config.dense_model,
        sparse_model=config.sparse_model,
        dense_pin=pins.get(config.dense_model),
        sparse_pin=pins.get(config.sparse_model),
        bm25_avg_len=config.bm25_avg_len,
    )
    reason = stale_reason(live, expected, await ports.index.count(target))
    return (live, None) if reason is None else (None, reason)


async def _process(
    document: CorpusDocument, ports: IngestPorts, processor: DocumentProcessor
) -> tuple[ProcessedDocument, int]:
    path = await ports.corpus.fetch(document)
    size = await asyncio.to_thread(lambda: path.stat().st_size)
    return await asyncio.to_thread(processor.process, path, document), size


async def process_documents(
    documents: Sequence[CorpusDocument], dense_model: str, ports: IngestPorts
) -> list[ProcessedDocument]:
    """Chunk every document without embedding or publishing (for `inspect-chunks`).

    Only the dense model is verified, because only its tokenizer is needed.
    """
    processor = ports.processor(await ports.models.fetch(dense_model))
    return [(await _process(document, ports, processor))[0] for document in documents]


def _seconds(start: float) -> float:
    return round(time.perf_counter() - start, 3)


def _stage_done(stage: str, items: int, start: float) -> None:
    """One line per stage, so a slow build shows where the time went (ADR 0008)."""
    logger.info("stage done", extra={"stage": stage, "items": items, "seconds": _seconds(start)})


async def _chunk_all(
    documents: Sequence[CorpusDocument],
    ports: IngestPorts,
    processor: DocumentProcessor,
    progress: _Progress,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        start = time.perf_counter()
        processed, size = await _process(document, ports, processor)
        chunks.extend(processed.chunks)
        progress.chunk_counts[document.doc_id] = len(processed.chunks)
        progress.drop_stats[document.doc_id] = processed.drop_stats()
        logger.info(
            "document chunked",
            extra={
                "doc_id": document.doc_id,
                "bytes": size,
                "pages": processed.pages,
                "chunks": len(processed.chunks),
                "seconds": _seconds(start),  # fetch, parse, clean and chunk
            },
        )
    return chunks


async def _embed(chunks: Sequence[Chunk], embedder: Embedder) -> EmbeddedChunks:
    texts = [chunk.text for chunk in chunks]
    start = time.perf_counter()
    dense = await embedder.embed_dense(texts)
    _stage_done("embed_dense", len(texts), start)
    start = time.perf_counter()
    sparse = await embedder.embed_sparse(texts)
    _stage_done("embed_sparse", len(texts), start)
    return EmbeddedChunks(chunks=chunks, dense=dense, sparse=sparse)


async def _measured_bm25(
    chunks: Sequence[Chunk], sparse: VerifiedModel, ports: IngestPorts
) -> Bm25Params:
    """`auto`: avg_len is the mean BM25 length of the chunks just built."""
    start = time.perf_counter()
    measure = ports.sparse_lengths(sparse)
    lengths = await asyncio.to_thread(measure.lengths, [chunk.text for chunk in chunks])
    params = Bm25Params(avg_len=mean_length(lengths), avg_len_auto=True)
    _stage_done("measure_bm25", len(lengths), start)
    return params


async def _build(
    documents: Sequence[CorpusDocument],
    config: IngestConfig,
    ports: IngestPorts,
    reason: StaleReason | None,
) -> IngestResult:
    dense = await ports.models.fetch(config.dense_model)
    sparse = await ports.models.fetch(config.sparse_model)

    created_at = ports.clock().astimezone(UTC)
    build_id = created_at.strftime(_BUILD_ID_FORMAT)
    collection = collection_name(config.chunking_version, build_id)
    # A fixed avg_len is known now, so even a build that fails early records it.
    fixed = None if isinstance(config.bm25_avg_len, str) else Bm25Params(config.bm25_avg_len)
    progress = _Progress(chunk_counts={}, drop_stats={}, sparse_params=fixed)

    def manifest_row(
        status: BuildStatus,
        warnings: tuple[BuildWarning, ...] = (),
        error_type: str | None = None,
    ) -> IndexBuild:
        return IndexBuild(
            build_id=build_id,
            created_at=created_at,
            collection=collection,
            status=status,
            document_hashes={d.doc_id: d.sha256 for d in documents},
            chunking_version=config.chunking_version,
            dense=dense.files,
            sparse=sparse.files,
            chunk_counts=dict(progress.chunk_counts),
            drop_stats=dict(progress.drop_stats),
            warnings=warnings,
            error_type=error_type,
            sparse_params=progress.sparse_params,
        )

    logger.info("index build started", extra={"build_id": build_id})
    try:
        processor = ports.processor(dense)
        chunks = await _chunk_all(documents, ports, processor, progress)
        bm25 = fixed or await _measured_bm25(chunks, sparse, ports)
        progress.sparse_params = bm25
        logger.info("bm25 params resolved", extra=dataclasses.asdict(bm25))
        embedded = await _embed(chunks, ports.embedder(dense, sparse, bm25))
        start = time.perf_counter()
        published = await publish(
            ports.index, config.alias, build_id, embedded, keep_previous=config.keep_previous
        )
        _stage_done("publish", len(chunks), start)
    except Exception as exc:
        await ports.manifest.record(manifest_row(BuildStatus.FAILED, error_type=type(exc).__name__))
        raise
    await ports.manifest.record(manifest_row(BuildStatus.SUCCEEDED, warnings=published.warnings))
    logger.info(
        "index build published",
        extra={
            "build_id": build_id,
            "collection": published.collection,
            "chunks": len(chunks),
            "pruned": len(published.pruned),
            "warnings": len(published.warnings),
        },
    )
    return IngestResult(False, published.collection, build_id, reason, published.warnings)


async def ingest(
    documents: Sequence[CorpusDocument],
    pins: Mapping[str, ModelFiles],
    config: IngestConfig,
    ports: IngestPorts,
    *,
    force: bool,
) -> IngestResult:
    """Build and publish the index, or with `force=False` reuse a live build that is current."""
    reason: StaleReason | None = None
    if not force:
        live, reason = await _live_build(documents, pins, config, ports)
        if live is not None:
            logger.info(
                "index is current; build skipped",
                extra={"build_id": live.build_id, "collection": live.collection},
            )
            return IngestResult(True, live.collection, live.build_id, None)
        logger.info("index build needed", extra={"reason": reason})
    return await _build(documents, config, ports, reason)

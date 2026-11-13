"""Ingest CLI: `python -m app.ingestion (--if-needed | --force | --inspect-chunks PATH)`.

This is the composition root of ingestion: the only module that builds the real
adapters. Exit codes: 0 when the index is live (built or reused) or the report is
written, 1 on any failure, 2 for bad arguments. A model failure is logged as one
structured line (model, file, error type) and nothing else (ADR 0006); any other
failure as one line with its error type. Neither includes exception messages.
"""

import argparse
import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import AsyncQdrantClient

from app.core.logging import configure_logging
from app.core.settings import Settings, get_settings
from app.domain.ingestion import (
    Bm25Params,
    CorpusDocument,
    ModelError,
    ModelFiles,
    VerifiedModel,
)
from app.infra.corpus_http import HttpCorpusSource, create_corpus_http_client
from app.infra.db import create_engine, create_session_factory
from app.infra.fastembed_bm25 import FastembedBm25Lengths
from app.infra.fastembed_embedder import (
    DenseModelClass,
    FastembedEmbedder,
    SparseModelClass,
)
from app.infra.fastembed_tokens import FastembedTokenCounter, TokenizingModelClass
from app.infra.manifest_repository import SqlManifestStore
from app.infra.model_files import HuggingFaceHub, PinnedModelSource, load_pins
from app.infra.pdfplumber_parser import PdfplumberParser
from app.infra.qdrant_index import QdrantIndex
from app.ingestion.chunking import VERSION, StructureChunker
from app.ingestion.corpus import load_corpus
from app.ingestion.pipeline import (
    IngestConfig,
    IngestPorts,
    StructureProcessor,
    ingest,
    process_documents,
)
from app.ingestion.report import render_markdown

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Runtime:
    documents: tuple[CorpusDocument, ...]
    pins: Mapping[str, ModelFiles]
    ports: IngestPorts


RuntimeFactory = Callable[[Settings], AbstractAsyncContextManager[Runtime]]


@asynccontextmanager
async def open_runtime(
    settings: Settings,
    *,
    qdrant_class: Callable[..., AsyncQdrantClient] = AsyncQdrantClient,
    dense_class: DenseModelClass = TextEmbedding,
    sparse_class: SparseModelClass = SparseTextEmbedding,
    token_class: TokenizingModelClass = TextEmbedding,
) -> AsyncIterator[Runtime]:
    """The real adapters, closed on exit. Nothing here connects until first used."""
    documents = load_corpus()

    def processor(dense: VerifiedModel) -> StructureProcessor:
        tokens = FastembedTokenCounter.from_settings(settings, dense, model_class=token_class)
        chunker = StructureChunker(
            tokens, max_tokens=settings.chunk_max_tokens, min_tokens=settings.chunk_min_tokens
        )
        return StructureProcessor(PdfplumberParser(), chunker)

    def embedder(
        dense: VerifiedModel, sparse: VerifiedModel, bm25: Bm25Params
    ) -> FastembedEmbedder:
        return FastembedEmbedder.from_settings(
            settings, dense, sparse, bm25=bm25, dense_class=dense_class, sparse_class=sparse_class
        )

    def sparse_lengths(sparse: VerifiedModel) -> FastembedBm25Lengths:
        return FastembedBm25Lengths(sparse, settings.embedding_cache_dir)

    # The server image and the client are pinned together and an integration test checks
    # them; the client's own check is a background request that runs even when the
    # index is never used (inspect-chunks) and reports outside the JSON logs.
    qdrant = qdrant_class(url=str(settings.qdrant_url), check_compatibility=False)
    engine = create_engine(settings)
    try:
        async with create_corpus_http_client(settings) as http:
            yield Runtime(
                documents=documents,
                pins=load_pins(),
                ports=IngestPorts(
                    corpus=HttpCorpusSource.from_settings(http, documents, settings),
                    models=PinnedModelSource(
                        load_pins(), settings.embedding_cache_dir, HuggingFaceHub()
                    ),
                    index=QdrantIndex(qdrant),
                    manifest=SqlManifestStore(create_session_factory(engine)),
                    processor=processor,
                    embedder=embedder,
                    sparse_lengths=sparse_lengths,
                    clock=lambda: datetime.now(UTC),
                ),
            )
    finally:
        await qdrant.close()
        await engine.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.ingestion", description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--if-needed", action="store_true", help="build only when the live index is not current"
    )
    mode.add_argument("--force", action="store_true", help="always build a new index")
    mode.add_argument(
        "--inspect-chunks",
        type=Path,
        metavar="PATH",
        help="write every chunk to a Markdown review file; no index, no database",
    )
    return parser


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


async def _run(args: argparse.Namespace, settings: Settings, runtime: RuntimeFactory) -> None:
    async with runtime(settings) as rt:
        if args.inspect_chunks is not None:
            processed = await process_documents(
                rt.documents, settings.embedding_dense_model, rt.ports
            )
            await asyncio.to_thread(_write, args.inspect_chunks, render_markdown(processed))
            logger.info("chunk review written", extra={"documents": len(processed)})
            return
        config = IngestConfig(
            alias=settings.qdrant_collection_alias,
            keep_previous=settings.index_keep_previous,
            dense_model=settings.embedding_dense_model,
            sparse_model=settings.embedding_sparse_model,
            chunking_version=VERSION,
            bm25_avg_len=settings.embedding_bm25_avg_len,
        )
        await ingest(rt.documents, rt.pins, config, rt.ports, force=args.force)


def main(argv: Sequence[str] | None = None, *, runtime: RuntimeFactory = open_runtime) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    configure_logging(
        settings.log_level, include_exception_messages=settings.log_include_exception_messages
    )
    try:
        asyncio.run(_run(args, settings, runtime))
    except ModelError as exc:
        logger.error(  # noqa: TRY400 - one line by design (ADR 0006), no traceback
            "embedding model check failed",
            extra={
                "model": getattr(exc, "model", None),
                "file": getattr(exc, "file", None),
                "error_type": type(exc).__name__,
            },
        )
        return 1
    except Exception as exc:
        # The JSON formatter keeps the traceback's frames and drops messages (CLAUDE.md).
        logger.exception("ingestion failed", extra={"error_type": type(exc).__name__})
        return 1
    return 0

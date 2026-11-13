"""Benchmark CLI: `python -m app.evaluation (preflight | run) [options]`.

The composition root of the benchmark: the only evaluation module that builds real
adapters. It runs in the backend image against the running stack's index, in
process, with the models baked into the image (ADR 0008); `make eval-retrieval`
mounts `eval/` at `/eval`.

- `preflight` resolves every golden anchor against the indexed chunks and logs ids
  and counts only.
- `run` runs the preflight, then each named configuration over one split, and writes
  `reports/runs/<utc>-<sha>/` under the eval directory.

Exit codes: 0 done; 1 the command ran and failed (an invalid golden set or matrix,
an unusable index or models, preflight problems, any other failure); 2 refused before
any work (bad arguments, the held-out split without --allow-heldout, a missing or
malformed GIT_SHA, an unknown config name). Logs carry ids, counts, scores and
timings, never question wording, chunk text or exception messages.
"""

import argparse
import asyncio
import logging
import os
import platform
import re
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from concurrent.futures import Executor
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.logging import configure_logging
from app.core.settings import Settings, get_settings
from app.domain.evaluation import GoldenQuestion, GoldenSet, GoldenSetError, IndexedChunk, Split
from app.domain.ingestion import IndexBuild, ManifestStore, ModelFiles
from app.evaluation.golden import load_golden
from app.evaluation.matrix import MatrixError, NamedConfig, load_matrix
from app.evaluation.report import RunHeader, RunOutcome, write_run
from app.evaluation.runner import (
    AnchorProblem,
    HeldoutRefusedError,
    IndexInfo,
    Preflight,
    QuestionRun,
    Retriever,
    preflight,
    run_config,
    select_best,
    select_questions,
    summarize_config,
)
from app.infra.db import create_engine, create_session_factory
from app.infra.inference import CpuAllowance, cpu_allowance, create_inference_executor
from app.infra.manifest_repository import SqlManifestStore
from app.infra.model_files import load_pins
from app.infra.qdrant_chunks import scroll_chunks
from app.infra.qdrant_index import QdrantIndex
from app.infra.qdrant_search import QdrantSearchStore, create_qdrant_client
from app.infra.query_models import ModelsState, QueryModels
from app.ingestion.corpus import load_corpus
from app.services.retrieval import RetrievalService, product_names

logger = logging.getLogger(__name__)

GOLDEN_FILE = Path("golden") / "retrieval-v1.toml"
MATRIX_FILE = Path("retrieval-matrix.toml")
RUNS_DIR = Path("reports") / "runs"
DEFAULT_EVAL_DIR = Path("/eval")
_GIT_SHA = re.compile(r"[0-9a-f]{7,40}(-dirty)?")


class Failure(StrEnum):
    NO_LIVE_INDEX = "no_live_index"
    NO_SUCCEEDED_BUILD = "no_succeeded_build"
    MODELS_FAILED = "models_failed"
    MODEL_FILES_MISMATCH = "model_files_mismatch"


class EvaluationError(Exception):
    """The index or the models cannot be benchmarked; `failure` says why."""

    def __init__(self, failure: Failure) -> None:
        super().__init__(failure.value)
        self.failure = failure


class _RefusedError(Exception):
    """Refused before any work (exit 2); the message is a fixed string, safe to log."""


@dataclass(frozen=True)
class Runtime:
    index: IndexInfo
    chunks: Callable[[], Awaitable[list[IndexedChunk]]]
    retriever: Callable[[str], Awaitable[Retriever]]  # by reranker model name
    revisions: Mapping[str, str]  # pinned model name -> hub revision


RuntimeFactory = Callable[[Settings, str | None], AbstractAsyncContextManager[Runtime]]
ModelsFactory = Callable[[Settings, Executor], QueryModels]


async def _resolve(
    qdrant: AsyncQdrantClient, manifest: ManifestStore, alias: str, collection: str | None
) -> tuple[IndexInfo, IndexBuild]:
    """An explicit collection (a kept earlier build) or the alias's live one."""
    target = collection or await QdrantIndex(qdrant).alias_target(alias)
    if target is None:
        raise EvaluationError(Failure.NO_LIVE_INDEX)
    build = await manifest.succeeded_build(target)
    if build is None:
        raise EvaluationError(Failure.NO_SUCCEEDED_BUILD)
    info = IndexInfo(
        alias=alias if collection is None else None,
        collection=target,
        build_id=build.build_id,
        chunk_count=build.chunk_count,
        chunking_version=build.chunking_version,
    )
    return info, build


async def _loaded(models: QueryModels, build: IndexBuild) -> QueryModels:
    """Query vectors from other model files than the build's would be silently wrong."""
    await models.load()
    if models.state is not ModelsState.READY:
        raise EvaluationError(Failure.MODELS_FAILED)
    if models.verified_files() != (build.dense, build.sparse):
        raise EvaluationError(Failure.MODEL_FILES_MISMATCH)
    return models


@asynccontextmanager
async def open_runtime(  # noqa: PLR0913 - settings, the target, and injectable factories
    settings: Settings,
    collection: str | None,
    *,
    qdrant_factory: Callable[[Settings], AsyncQdrantClient] = create_qdrant_client,
    engine_factory: Callable[[Settings], AsyncEngine] = create_engine,
    models_factory: ModelsFactory = QueryModels.from_settings,
    pins: Callable[[], Mapping[str, ModelFiles]] = load_pins,
) -> AsyncIterator[Runtime]:
    """The real adapters, closed on exit; the models load once, before the first run."""
    qdrant = qdrant_factory(settings)
    engine = engine_factory(settings)
    executor = create_inference_executor(settings.inference_workers)
    try:
        manifest = SqlManifestStore(create_session_factory(engine))
        index, build = await _resolve(
            qdrant, manifest, settings.qdrant_collection_alias, collection
        )
        models = await _loaded(models_factory(settings, executor), build)
        store = QdrantSearchStore(
            qdrant, index.collection, timeout_seconds=settings.qdrant_search_timeout_seconds
        )
        products = product_names(load_corpus())

        async def chunks() -> list[IndexedChunk]:
            return await scroll_chunks(qdrant, index.collection)

        async def retriever(name: str) -> Retriever:
            reranker = await models.reranker(name)
            return RetrievalService(models.query_embedder(), store, reranker, products=products)

        yield Runtime(
            index=index,
            chunks=chunks,
            retriever=retriever,
            revisions={name: files.revision for name, files in pins().items()},
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        await qdrant.close()
        await engine.dispose()


# ---------------------------------------------------------------- commands


@dataclass(frozen=True)
class _Context:
    runtime: RuntimeFactory
    environ: Mapping[str, str]
    clock: Callable[[], datetime]
    cpu: Callable[[], CpuAllowance]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.evaluation", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--eval-dir", type=Path, default=DEFAULT_EVAL_DIR, help="the mounted eval/ directory"
    )
    common.add_argument(
        "--collection", help="read this collection (a kept earlier build) instead of the alias"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", parents=[common], help="resolve every golden anchor")
    run = commands.add_parser("run", parents=[common], help="benchmark the named configs")
    run.add_argument("--split", choices=[s.value for s in Split], default=Split.DEV.value)
    run.add_argument(
        "--allow-heldout", action="store_true", help="score the held-out split (Phase 5 only)"
    )
    run.add_argument(
        "--config", action="append", default=[], metavar="NAME", help="repeatable; default all"
    )
    return parser


def _log_preflight(report: Preflight) -> None:
    for problem in report.problems:
        logger.warning(
            "anchor does not resolve",
            extra={
                "question_id": problem.question_id,
                "split": problem.split.value,
                "anchor": problem.anchor,
                "matches": problem.matches,
                "problem": str(problem.problem),
            },
        )
    counts = Counter(str(r.matches) for r in report.resolutions)
    extra = {
        "chunks": report.chunks,
        "anchors": len(report.resolutions),
        "match_counts": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
    }
    if report.ok:
        logger.info("preflight passed", extra=extra)
        return
    kinds = Counter(problem.problem for problem in report.problems)
    logger.error(
        "preflight failed",
        extra={
            **extra,
            "unresolved": kinds[AnchorProblem.UNRESOLVED],
            "too_generic": kinds[AnchorProblem.TOO_GENERIC],
        },
    )


async def _preflight(golden: GoldenSet, runtime: Runtime) -> Preflight:
    report = preflight(golden, await runtime.chunks())
    _log_preflight(report)
    return report


def _git_sha(environ: Mapping[str, str]) -> str:
    sha = environ.get("GIT_SHA", "")
    if not _GIT_SHA.fullmatch(sha):
        raise _RefusedError("GIT_SHA must be a commit SHA, optionally with -dirty")
    return sha


def _configs(
    path: Path, settings: Settings, names: Sequence[str]
) -> tuple[int, int, tuple[NamedConfig, ...]]:
    matrix = load_matrix(path, settings)
    try:
        selected = matrix.select(names)
    except MatrixError as exc:
        raise _RefusedError("unknown config name; see retrieval-matrix.toml") from exc
    return matrix.seed, matrix.resamples, selected


async def _run_all(
    configs: Sequence[NamedConfig], questions: Sequence[GoldenQuestion], runtime: Runtime
) -> dict[str, tuple[QuestionRun, ...]]:
    """Every reranker loads before the first config runs, so a bad pin fails early."""
    retrievers = {
        name: await runtime.retriever(name) for name in dict.fromkeys(c.reranker for c in configs)
    }
    runs: dict[str, tuple[QuestionRun, ...]] = {}
    for named in configs:
        runs[named.name] = await run_config(named, questions, retrievers[named.reranker])
        logger.info("config done", extra={"config": named.name, "questions": len(questions)})
    return runs


def _hardware(settings: Settings, allowance: CpuAllowance) -> dict[str, object]:
    return {
        "machine": platform.machine(),
        "cpus": allowance.cpus,
        "cpu_source": allowance.source.value,
        "inference_workers": settings.inference_workers,
        "onnx_threads": settings.inference_onnx_threads,
    }


def _models(
    settings: Settings, configs: Sequence[NamedConfig], revisions: Mapping[str, str]
) -> dict[str, str | None]:
    """The models this run used, each with its pinned revision."""
    rerankers = dict.fromkeys(c.reranker for c in configs if c.config.rerank)
    names = [settings.embedding_dense_model, settings.embedding_sparse_model, *rerankers]
    return {name: revisions.get(name) for name in names}


async def _run(
    args: argparse.Namespace, settings: Settings, golden: GoldenSet, ctx: _Context
) -> int:
    try:
        questions = select_questions(golden, Split(args.split), allow_heldout=args.allow_heldout)
    except HeldoutRefusedError as exc:
        raise _RefusedError("held-out split refused; pass --allow-heldout") from exc
    git_sha = _git_sha(ctx.environ)
    seed, resamples, configs = _configs(args.eval_dir / MATRIX_FILE, settings, args.config)
    created_at = ctx.clock()
    async with ctx.runtime(settings, args.collection) as runtime:
        report = await _preflight(golden, runtime)
        if not report.ok:
            return 1
        runs = await _run_all(configs, questions, runtime)
        header = RunHeader(
            created_at=created_at,
            git_sha=git_sha,
            golden_version=golden.version,
            golden_sha256=golden.sha256,
            split=Split(args.split),
            questions=len(questions),
            index=runtime.index,
            models=_models(settings, configs, runtime.revisions),
            hardware=_hardware(settings, ctx.cpu()),
            seed=seed,
            resamples=resamples,
        )
    summaries = tuple(
        summarize_config(c, runs[c.name], seed=seed, resamples=resamples) for c in configs
    )
    outcome = RunOutcome(
        configs=configs,
        summaries=summaries,
        selection=select_best(summaries),
        preflight=report,
        runs=tuple(run for c in configs for run in runs[c.name]),
    )
    path = await asyncio.to_thread(write_run, args.eval_dir / RUNS_DIR, header, outcome)
    logger.info("benchmark written", extra={"run": path.name, "configs": len(configs)})
    return 0


async def _command(args: argparse.Namespace, settings: Settings, ctx: _Context) -> int:
    doc_ids = {document.doc_id for document in load_corpus()}
    golden = load_golden(args.eval_dir / GOLDEN_FILE, doc_ids)
    if args.command == "run":
        return await _run(args, settings, golden, ctx)
    async with ctx.runtime(settings, args.collection) as runtime:
        report = await _preflight(golden, runtime)
    return 0 if report.ok else 1


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime: RuntimeFactory = open_runtime,
    environ: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    cpu: Callable[[], CpuAllowance] = cpu_allowance,
) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    configure_logging(
        settings.log_level, include_exception_messages=settings.log_include_exception_messages
    )
    try:
        return asyncio.run(_command(args, settings, _Context(runtime, environ, clock, cpu)))
    except _RefusedError as exc:
        logger.error(str(exc))  # noqa: TRY400 - a fixed message, no traceback needed
        return 2
    except GoldenSetError as exc:
        logger.error("golden set is invalid", extra={"rule": exc.rule.value})  # noqa: TRY400
        return 1
    except MatrixError:
        logger.error("retrieval matrix is invalid")  # noqa: TRY400 - see the matrix tests
        return 1
    except EvaluationError as exc:
        logger.error("evaluation failed", extra={"reason": exc.failure.value})  # noqa: TRY400
        return 1
    except Exception as exc:
        # The JSON formatter keeps the traceback's frames and drops messages (CLAUDE.md).
        logger.exception("evaluation failed", extra={"error_type": type(exc).__name__})
        return 1

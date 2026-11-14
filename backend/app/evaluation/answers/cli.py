"""Answer benchmark CLI: `python -m app.evaluation.answers <command> [options]`.

- `estimate` prints the budget plan (`estimate.render`): each run's estimate in
  priority order, which runs the cap still funds, and the spend so far split into
  measured and estimated charges. It reads files only (matrix, frozen datasets,
  ledger), so `make eval-estimate` runs it on the host with no stack and no key.
- `ask --run NAME` asks one matrix run (`ask.ask_run`) inside the backend image on the
  compose networks, after the host-side driver restarted the backend with the run's
  settings. `--limit N` asks a spread of N pending items (the pilot), `--resume`
  continues a run directory, `--retry-errors` re-asks errored items, and
  `--allow-heldout` is accepted for the heldout run only. Exit 1 also when the run is
  invalid (more than 5% errors).
- `drive --run NAME --eval-dir DIR` is the host side (`make eval-answers`): it restarts
  the backend for the run, checks its env, runs `ask` in a harness container with the
  same run options and restores the defaults (`driver.drive`). It never opens a
  database, so it needs no settings.

Run files live in `answer-runs/` under the eval directory, which is git-ignored: the
ledger, and per run the config record, the results journal and the traces (stored
rows and HP excerpt text). Exit codes: 0 done; 1 the command ran and failed (an
invalid matrix, dataset or ledger, a stopped or invalid run); 2 bad arguments. Logs
carry fixed messages and the error's own text, which names files, keys, ids and line
numbers (for the retrieval set, its rule only), never values or question wording; an
infrastructure failure logs its type only.
"""

import argparse
import asyncio
import logging
import os
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import TextIO

from app.core.logging import configure_logging
from app.core.settings import Settings, get_settings
from app.domain.answer_dataset import Dataset, DatasetError, DatasetNotFrozenError, EvalItem
from app.domain.evaluation import GoldenSetError
from app.evaluation.answers.answer_matrix import (
    AnswerMatrix,
    ItemSet,
    MatrixError,
    RunSpec,
    load_answer_matrix,
)
from app.evaluation.answers.ask import AskError, AskOptions, AskPorts, RunFiles, ask_run
from app.evaluation.answers.budget import (
    BudgetExceededError,
    Ledger,
    LedgerError,
    RunNotFundedError,
    UnsoundBoundError,
)
from app.evaluation.answers.config_check import ConfigMismatchError
from app.evaluation.answers.dataset import load_dataset
from app.evaluation.answers.driver import (
    CommandRunner,
    DriveRequest,
    DriverError,
    drive,
    subprocess_runner,
)
from app.evaluation.answers.estimate import plan, render, typical_tokens
from app.evaluation.answers.excerpts import ExcerptError
from app.evaluation.answers.freeze import run_items
from app.evaluation.answers.journal import JournalError, read_journal, retried_items
from app.evaluation.answers.runtime import AskRuntime, open_ask_runtime
from app.evaluation.answers.settle import StoreMismatchError
from app.evaluation.runner import HeldoutRefusedError
from app.ingestion.corpus import load_corpus

_STOPS = (
    AskError,
    BudgetExceededError,
    ConfigMismatchError,
    DriverError,
    ExcerptError,
    JournalError,
    RunNotFundedError,
    StoreMismatchError,
    UnsoundBoundError,
)
AskRuntimeFactory = Callable[[Settings, str], AbstractAsyncContextManager[AskRuntime]]

logger = logging.getLogger(__name__)

MATRIX_FILE = Path("answer-matrix.toml")
RUNS_DIR = Path("answer-runs")
LEDGER_FILE = RUNS_DIR / "ledger.jsonl"
DEFAULT_EVAL_DIR = Path("/eval")
DEFAULT_API_URL = "http://backend:8000"
# The API's message limit as compose sets it by default; turns longer than this fail
# dataset validation instead of a 422 in the middle of a paid run.
MAX_TURN_CHARS: int = Settings.model_fields["chat_message_max_chars"].default


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.evaluation.answers", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    estimate = commands.add_parser("estimate", help="print the budget plan and the spend")
    estimate.add_argument(
        "--eval-dir", type=Path, default=DEFAULT_EVAL_DIR, help="the eval/ directory"
    )
    estimate.add_argument("--ledger", type=Path, help="default: answer-runs/ledger.jsonl")
    run_options = argparse.ArgumentParser(add_help=False)
    run_options.add_argument("--run", required=True, help="the matrix run name")
    run_options.add_argument("--resume", action="store_true", help="continue a run directory")
    run_options.add_argument("--retry-errors", action="store_true", help="re-ask errored items")
    run_options.add_argument("--limit", type=int, help="ask a spread of N pending items (pilot)")
    run_options.add_argument("--allow-heldout", action="store_true", help="the heldout run only")
    ask = commands.add_parser(
        "ask", parents=[run_options], help="ask one run of the matrix (inside the backend image)"
    )
    ask.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR, help="the eval/ directory")
    ask.add_argument("--api-url", default=DEFAULT_API_URL, help="the backend API base URL")
    drive = commands.add_parser(
        "drive", parents=[run_options], help="drive one run from the host (restart, check, ask)"
    )
    drive.add_argument("--eval-dir", type=Path, required=True, help="the eval/ directory")
    return parser


def _estimate(args: argparse.Namespace, stdout: TextIO) -> int:
    matrix = load_answer_matrix(args.eval_dir / MATRIX_FILE)
    doc_ids = {document.doc_id for document in load_corpus()}
    dataset = load_dataset(args.eval_dir, doc_ids, max_turn_chars=MAX_TURN_CHARS)
    # Counting the held-out items for the plan is not scoring them (rule 8).
    items = {run.name: run_items(dataset, run.items, allow_heldout=True) for run in matrix.runs}
    charges = Ledger(args.ledger or args.eval_dir / LEDGER_FILE).charges()
    typical = typical_tokens(matrix, charges)
    stdout.write(render(matrix, plan(matrix, items, charges, typical.tokens), typical, charges))
    return 0


class _RefusedError(Exception):
    """Arguments the parser accepts but the run does not; exit 2."""


def _items_by_run(
    matrix: AnswerMatrix, dataset: Dataset, run: RunSpec, *, allow_heldout: bool
) -> dict[str, tuple[EvalItem, ...]]:
    """The asked run's items through the held-out guard; the other runs' items only
    to hold back the reserved runs' estimates (counting is not scoring, rule 8).
    """
    if allow_heldout and run.items is not ItemSet.HELDOUT:
        raise _RefusedError("--allow-heldout is for the heldout run only")
    try:
        asked = run_items(dataset, run.items, allow_heldout=allow_heldout)
    except HeldoutRefusedError as exc:
        raise _RefusedError(str(exc)) from None
    counted = {r.name: run_items(dataset, r.items, allow_heldout=True) for r in matrix.runs}
    return counted | {run.name: asked}


async def _ask(args: argparse.Namespace, settings: Settings, ask_runtime: AskRuntimeFactory) -> int:
    matrix = load_answer_matrix(args.eval_dir / MATRIX_FILE)
    run = matrix.run(args.run)
    doc_ids = {document.doc_id for document in load_corpus()}
    dataset = load_dataset(args.eval_dir, doc_ids, max_turn_chars=MAX_TURN_CHARS)
    items = _items_by_run(matrix, dataset, run, allow_heldout=args.allow_heldout)
    titles = {document.doc_id: document.title for document in load_corpus()}
    files = RunFiles(args.eval_dir / RUNS_DIR / run.name, args.eval_dir / LEDGER_FILE)
    options = AskOptions(resume=args.resume, retry_errors=args.retry_errors, limit=args.limit)
    async with ask_runtime(settings, args.api_url) as runtime:
        ports = AskPorts(
            backend=runtime.backend,
            index=runtime.index,
            rows=runtime.rows,
            chunks=runtime.chunks,
            titles=titles,
            new_client_id=lambda: str(uuid.uuid4()),
        )
        outcome = await ask_run(matrix, run, items, files, ports, options)
    health = outcome.health
    extra: dict[str, object] = {
        "run": run.name,
        "asked": outcome.asked,
        "recorded": len(outcome.results),
        "retried_items": len(retried_items(read_journal(files.results))),
    }
    if health is not None:
        extra |= {
            "valid": health.valid,
            "errors_by_code": dict(health.errors_by_code),
            "retries": health.retries,
        }
    if health is not None and not health.valid:
        logger.error("answer run is invalid", extra=extra)
        return 1
    logger.info("answer run done", extra=extra)
    return 0


def _ask_args(args: argparse.Namespace) -> tuple[str, ...]:
    """The run options given to `drive`, passed on to `ask` unchanged."""
    flags = [
        ("--limit", str(args.limit)) if args.limit is not None else (),
        ("--resume",) if args.resume else (),
        ("--retry-errors",) if args.retry_errors else (),
        ("--allow-heldout",) if args.allow_heldout else (),
    ]
    return tuple(part for flag in flags for part in flag)


def _drive(
    args: argparse.Namespace,
    runner_for: Callable[[Path], CommandRunner],
    environ: Mapping[str, str],
) -> int:
    eval_dir = args.eval_dir.resolve()
    matrix = load_answer_matrix(eval_dir / MATRIX_FILE)
    request = DriveRequest(
        matrix.run(args.run), eval_dir, _ask_args(args), uid=os.getuid(), gid=os.getgid()
    )
    return drive(matrix, request, runner_for(eval_dir.parent), environ)


def _failure(exc: Exception) -> tuple[int, str, dict[str, str]]:
    """Exit code, fixed message and log fields for a failed command."""
    if isinstance(exc, _RefusedError):
        return 2, str(exc), {}
    if isinstance(exc, _STOPS):
        # These messages name runs, settings, files, ids, models and amounts only.
        return 1, "answer run stopped", {"error": type(exc).__name__, "detail": str(exc)}
    if isinstance(exc, GoldenSetError):
        # As in the retrieval CLI: the rule only, the message may quote the file.
        return 1, "dataset is invalid", {"rule": exc.rule.value}
    for kinds, message in _INVALID_FILES:
        if isinstance(exc, kinds):
            return 1, message, {"detail": str(exc)}
    # Infrastructure failures (API, Qdrant, Postgres): the type only, the message could
    # carry request or row content.
    return 1, "answer run failed", {"error": type(exc).__name__}


_INVALID_FILES: tuple[tuple[type[Exception] | tuple[type[Exception], ...], str], ...] = (
    (MatrixError, "answer matrix is invalid"),
    ((DatasetError, DatasetNotFrozenError), "dataset is invalid"),
    (LedgerError, "ledger is invalid"),
)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    ask_runtime: AskRuntimeFactory = open_ask_runtime,
    drive_runner: Callable[[Path], CommandRunner] = subprocess_runner,
    environ: Mapping[str, str] = os.environ,
) -> int:
    args = _parser().parse_args(argv)
    configure_logging("INFO")
    try:
        if args.command == "ask":
            return asyncio.run(_ask(args, get_settings(), ask_runtime))
        if args.command == "drive":
            return _drive(args, drive_runner, environ)
        return _estimate(args, stdout)
    except Exception as exc:  # every failure maps to an exit code and one log line
        code, message, extra = _failure(exc)
        logger.error(message, extra=extra)  # noqa: TRY400 - fixed messages, no traceback
        return code

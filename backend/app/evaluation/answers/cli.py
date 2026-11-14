"""Answer benchmark CLI: `python -m app.evaluation.answers <command> [options]`.

- `estimate` prints the budget plan (`estimate.render`): each run's estimate in
  priority order, which runs the cap still funds, and the spend so far split into
  measured and estimated charges. It reads files only (matrix, frozen datasets,
  ledger), so `make eval-estimate` runs it on the host with no stack and no key.

Run files live in `answer-runs/` under the eval directory, which is git-ignored: the
ledger, and per run the results journal and the traces (stored rows and HP excerpt
text). Exit codes: 0 done; 1 the command ran and failed (an invalid matrix, dataset
or ledger); 2 bad arguments. Logs carry fixed messages and the error's own text,
which names files, keys, ids and line numbers (for the retrieval set, its rule only),
never values or question wording.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from app.core.logging import configure_logging
from app.core.settings import Settings
from app.domain.answer_dataset import DatasetError, DatasetNotFrozenError
from app.domain.evaluation import GoldenSetError
from app.evaluation.answers.answer_matrix import MatrixError, load_answer_matrix
from app.evaluation.answers.budget import Ledger, LedgerError
from app.evaluation.answers.dataset import load_dataset
from app.evaluation.answers.estimate import plan, render, typical_tokens
from app.evaluation.answers.freeze import run_items
from app.ingestion.corpus import load_corpus

logger = logging.getLogger(__name__)

MATRIX_FILE = Path("answer-matrix.toml")
RUNS_DIR = Path("answer-runs")
LEDGER_FILE = RUNS_DIR / "ledger.jsonl"
DEFAULT_EVAL_DIR = Path("/eval")
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


def _failed(message: str, **extra: str) -> int:
    logger.error(message, extra=extra)
    return 1


def main(argv: Sequence[str] | None = None, *, stdout: TextIO = sys.stdout) -> int:
    args = _parser().parse_args(argv)
    configure_logging("INFO")
    try:
        return _estimate(args, stdout)
    except MatrixError as exc:
        return _failed("answer matrix is invalid", detail=str(exc))
    except (DatasetError, DatasetNotFrozenError) as exc:
        return _failed("dataset is invalid", detail=str(exc))
    except GoldenSetError as exc:
        # As in the retrieval CLI: the rule only, the message may quote the file.
        return _failed("dataset is invalid", rule=exc.rule.value)
    except LedgerError as exc:
        return _failed("ledger is invalid", detail=str(exc))

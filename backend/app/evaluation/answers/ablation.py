"""The ablation sweep (`ablate` command, on the host): matrix rows in priority order.

For each row, in the matrix's order:
1. A finished row (every item has a final result) gets its report written again and
   nothing else, so a sweep can be repeated after an interruption.
2. A row the budget plan cuts, recomputed from the ledger right before the row, is
   "not run: budget" (the summary marks it from the same plan) and the sweep goes on:
   a later row may still fit.
3. Otherwise the row's index is built if it needs one, the row is driven (resuming a
   run directory that exists), and its report is written.

A driven run that exits non-zero (invalid, stopped, refused) stops the sweep: the next
rows would meet the same cause. The held-out row is never swept: it runs once, for the
final default, after every decision (rule 8).
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from app.evaluation.answers.answer_matrix import AnswerMatrix, ItemSet, RunSpec

logger = logging.getLogger(__name__)


class AblationError(Exception):
    """A row the sweep must not run; the message names the row."""


class Outcome(StrEnum):
    FINISHED = "finished"
    NOT_RUN_BUDGET = "not run: budget"
    DONE = "done"
    STOPPED = "stopped"


@dataclass(frozen=True)
class RowResult:
    run: str
    outcome: Outcome
    code: int = 0  # the driven run's exit code when it stopped


class SweepPorts(Protocol):
    def finished(self, run: RunSpec) -> bool: ...

    def started(self, run: RunSpec) -> bool: ...

    def funded(self, run: RunSpec) -> bool: ...

    def build_index(self, run: RunSpec) -> None: ...

    def drive(self, run: RunSpec, *, resume: bool) -> int: ...

    def report(self, run: RunSpec) -> None: ...


def sweep_runs(matrix: AnswerMatrix, names: Sequence[str]) -> tuple[RunSpec, ...]:
    """The named rows, or every unreserved row, always in the matrix's priority order."""
    if not names:
        return tuple(run for run in matrix.runs if not run.reserved)
    wanted = {matrix.run(name).name for name in names}
    runs = tuple(run for run in matrix.runs if run.name in wanted)
    heldout = [run.name for run in runs if run.items is ItemSet.HELDOUT]
    if heldout:
        raise AblationError(f"{heldout[0]} is never swept: drive it once, for the final default")
    return runs


def _row(run: RunSpec, ports: SweepPorts) -> RowResult:
    if ports.finished(run):
        ports.report(run)
        return RowResult(run.name, Outcome.FINISHED)
    if not ports.funded(run):
        return RowResult(run.name, Outcome.NOT_RUN_BUDGET)
    ports.build_index(run)
    code = ports.drive(run, resume=ports.started(run))
    if code != 0:
        return RowResult(run.name, Outcome.STOPPED, code)
    ports.report(run)
    return RowResult(run.name, Outcome.DONE)


def sweep(runs: Sequence[RunSpec], ports: SweepPorts) -> list[RowResult]:
    results: list[RowResult] = []
    for run in runs:
        result = _row(run, ports)
        results.append(result)
        logger.info(
            "ablation row",
            extra={"run": result.run, "outcome": result.outcome.value, "code": result.code},
        )
        if result.outcome is Outcome.STOPPED:
            break
    return results


def exit_code(results: Sequence[RowResult]) -> int:
    return next((r.code for r in results if r.outcome is Outcome.STOPPED), 0)

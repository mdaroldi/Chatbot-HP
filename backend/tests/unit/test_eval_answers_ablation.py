"""The ablation sweep: the matrix rows in priority order, each one finished, cut by the
budget, or built, driven and reported; a stopped run stops the sweep. Fake ports record
each step in order.
"""

from collections.abc import Iterable

import pytest

from app.evaluation.answers.ablation import (
    AblationError,
    Outcome,
    RowResult,
    exit_code,
    sweep,
    sweep_runs,
)
from app.evaluation.answers.answer_matrix import MatrixError, RunSpec, load_answer_matrix
from tests.unit.test_eval_answers_matrix import REPO_FILE

MATRIX = load_answer_matrix(REPO_FILE)


class FakePorts:
    def __init__(
        self,
        *,
        finished: Iterable[str] = (),
        started: Iterable[str] = (),
        cut: Iterable[str] = (),
        codes: dict[str, int] | None = None,
    ) -> None:
        self._finished, self._started, self._cut = set(finished), set(started), set(cut)
        self._codes = codes or {}
        self.log: list[tuple[str, str]] = []

    def finished(self, run: RunSpec) -> bool:
        self.log.append(("finished?", run.name))
        return run.name in self._finished

    def started(self, run: RunSpec) -> bool:
        return run.name in self._started

    def funded(self, run: RunSpec) -> bool:
        self.log.append(("funded?", run.name))
        return run.name not in self._cut

    def build_index(self, run: RunSpec) -> None:
        self.log.append(("build", run.name))

    def drive(self, run: RunSpec, *, resume: bool) -> int:
        self.log.append(("drive --resume" if resume else "drive", run.name))
        return self._codes.get(run.name, 0)

    def report(self, run: RunSpec) -> None:
        self.log.append(("report", run.name))


def runs(*names: str) -> tuple[RunSpec, ...]:
    return tuple(MATRIX.run(name) for name in names)


# ------------------------------------------------------------------ which rows


def test_by_default_the_sweep_takes_every_unreserved_row_in_priority_order() -> None:
    assert [run.name for run in sweep_runs(MATRIX, ())] == [
        "reference-1",
        "reference-2",
        "reference-3",
        "hybrid-no-rerank",
        "dense",
        "naive",
        "rewrite-off",
        "k3",
        "sparse",
        "k8",
    ]


def test_named_rows_keep_the_matrix_order_and_may_include_the_improvement_rerun() -> None:
    found = sweep_runs(MATRIX, ("k8", "improvement", "dense", "k8"))

    assert [run.name for run in found] == ["improvement", "dense", "k8"]


def test_an_unknown_row_is_refused() -> None:
    with pytest.raises(MatrixError, match="k4"):
        sweep_runs(MATRIX, ("dense", "k4"))


def test_the_heldout_row_is_never_swept() -> None:
    """It runs once, last, for the final default (rule 8), through `drive` with its flag."""
    with pytest.raises(AblationError, match="heldout"):
        sweep_runs(MATRIX, ("heldout",))


# ------------------------------------------------------------------ each row


def test_a_pending_funded_row_is_built_driven_and_reported_in_that_order() -> None:
    ports = FakePorts()

    found = sweep(runs("naive"), ports)

    assert found == [RowResult("naive", Outcome.DONE)]
    assert ports.log == [
        ("finished?", "naive"),
        ("funded?", "naive"),
        ("build", "naive"),
        ("drive", "naive"),
        ("report", "naive"),
    ]


def test_a_row_with_a_run_directory_resumes() -> None:
    ports = FakePorts(started=["dense"])

    sweep(runs("dense"), ports)

    assert ("drive --resume", "dense") in ports.log


def test_a_finished_row_gets_its_report_and_nothing_else() -> None:
    ports = FakePorts(finished=["reference-1"])

    found = sweep(runs("reference-1"), ports)

    assert found == [RowResult("reference-1", Outcome.FINISHED)]
    assert ports.log == [("finished?", "reference-1"), ("report", "reference-1")]


def test_a_row_the_budget_cuts_is_not_run_and_the_sweep_goes_on() -> None:
    ports = FakePorts(cut=["k3"])

    found = sweep(runs("k3", "sparse"), ports)

    assert found == [
        RowResult("k3", Outcome.NOT_RUN_BUDGET),
        RowResult("sparse", Outcome.DONE),
    ]
    assert ("build", "k3") not in ports.log
    assert ("drive", "k3") not in ports.log
    assert ("report", "k3") not in ports.log


def test_the_budget_is_checked_right_before_each_row_after_the_previous_one_ran() -> None:
    ports = FakePorts()

    sweep(runs("dense", "naive"), ports)

    assert ports.log.index(("funded?", "naive")) > ports.log.index(("report", "dense"))


def test_a_stopped_run_stops_the_sweep_without_a_report() -> None:
    ports = FakePorts(codes={"dense": 1})

    found = sweep(runs("hybrid-no-rerank", "dense", "naive"), ports)

    assert found == [
        RowResult("hybrid-no-rerank", Outcome.DONE),
        RowResult("dense", Outcome.STOPPED, 1),
    ]
    assert ("report", "dense") not in ports.log
    assert all(name != "naive" for _, name in ports.log)


# ------------------------------------------------------------------ exit code


def test_the_exit_code_is_zero_unless_a_run_stopped() -> None:
    assert exit_code([]) == 0
    assert (
        exit_code(
            [
                RowResult("a", Outcome.FINISHED),
                RowResult("b", Outcome.NOT_RUN_BUDGET),
                RowResult("c", Outcome.DONE),
            ]
        )
        == 0
    )
    assert exit_code([RowResult("a", Outcome.DONE), RowResult("b", Outcome.STOPPED, 2)]) == 2


def test_outcomes_read_as_the_summary_writes_them() -> None:
    assert [outcome.value for outcome in Outcome] == [
        "finished",
        "not run: budget",
        "done",
        "stopped",
    ]

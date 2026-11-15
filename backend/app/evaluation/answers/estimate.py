"""The budget plan `make eval-estimate` prints: what each run should cost, which runs the
cap still funds, and what has been spent, split into measured and estimated charges.

Typical tokens per call are the measured means of the ledger's answer and judge calls
once the pilot has run (the matrix estimate before). Runs are walked in priority order
with the same start check the harness applies before a run: reserved runs are funded
up front, and an unreserved run that no longer fits after the earlier funded ones is
marked cut. A run's need is what is left of its estimate after what it already spent.

Spend is split so the ledger can be compared with the Anthropic console: `measured` is
priced from stored token counts, `estimated` is the worst-case bounds charged where the
backend stores no usage (rewrite calls, turns removed before their first token). The
console bill should land between measured and measured + estimated.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.domain.answer_dataset import EvalItem
from app.evaluation.answers.answer_matrix import AnswerMatrix, RunSpec, Tokens
from app.evaluation.answers.budget import (
    Charge,
    Kind,
    Spend,
    StartCheck,
    charge,
    judge_requests,
    run_estimate,
    split_by_run,
    split_spend,
    start_check,
)

# Owner rule: above this measured cost per answer, k3, sparse and k8 become the last,
# conditional rows (2026-11-15: kept, not dropped; each runs only if its start check passes).
PILOT_ANSWER_LIMIT_USD = 0.008


class Status(StrEnum):
    RESERVED = "reserved"
    FUNDED = "funded"
    CUT = "cut"


@dataclass(frozen=True)
class Typical:
    tokens: Tokens
    answer_calls: int  # measured answer calls behind the answer means (0: matrix estimate)
    judge_calls: int

    def answer_cost(self, matrix: AnswerMatrix) -> float:
        tokens = self.tokens
        return charge(
            matrix,
            "",
            "",
            Kind.ANSWER,
            matrix.models.answer,
            tokens.answer_input_tokens,
            tokens.answer_output_tokens,
        ).cost_usd


@dataclass(frozen=True)
class PlanRow:
    run: str
    items: int
    turns: int
    judge_requests: int
    estimate: float  # the whole run at typical tokens
    needed: float  # what is left of the estimate, with the start margin
    spent: Spend
    status: Status


def _mean(values: Sequence[int]) -> int:
    return math.ceil(sum(values) / len(values))


def typical_tokens(matrix: AnswerMatrix, charges: Sequence[Charge]) -> Typical:
    """Measured means, rounded up, where measured calls exist; bounds are not measurements."""
    measured = [made for made in charges if not made.estimated]
    answers = [made for made in measured if made.kind is Kind.ANSWER]
    judged = [made for made in measured if made.kind is Kind.JUDGE]
    tokens = matrix.estimate
    if answers:
        tokens = tokens.model_copy(
            update={
                "answer_input_tokens": _mean([made.input_tokens for made in answers]),
                "answer_output_tokens": _mean([made.output_tokens for made in answers]),
            }
        )
    if judged:
        tokens = tokens.model_copy(
            update={
                "judge_input_tokens": _mean([made.input_tokens for made in judged]),
                "judge_output_tokens": _mean([made.output_tokens for made in judged]),
            }
        )
    return Typical(tokens, len(answers), len(judged))


def _reserved(
    matrix: AnswerMatrix, items: Mapping[str, Sequence[EvalItem]], tokens: Tokens
) -> dict[str, float]:
    return {
        run.name: run_estimate(matrix, run, items[run.name], tokens)
        for run in matrix.runs
        if run.reserved
    }


def plan(
    matrix: AnswerMatrix,
    items: Mapping[str, Sequence[EvalItem]],
    charges: Sequence[Charge],
    tokens: Tokens,
) -> list[PlanRow]:
    """`items` holds every run's items by run name."""
    reserved = _reserved(matrix, items, tokens)
    spent = split_by_run(charges)
    committed = {name: paid.total for name, paid in spent.items()}
    rows = []
    for run in matrix.runs:
        run_items = items[run.name]
        estimate = run_estimate(matrix, run, run_items, tokens)
        paid = spent.get(run.name, Spend(0.0, 0.0))
        to_go = max(0.0, estimate - paid.total)
        status = Status.RESERVED
        if not run.reserved:
            check = start_check(
                matrix, run, estimate=to_go, spent_by_run=committed, reserved=reserved
            )
            status = Status.FUNDED if check.ok else Status.CUT
            if check.ok:
                committed[run.name] = paid.total + check.needed
        rows.append(
            PlanRow(
                run=run.name,
                items=len(run_items),
                turns=sum(len(item.turns) for item in run_items),
                judge_requests=judge_requests(run, run_items),
                estimate=estimate,
                needed=to_go * matrix.start_margin,
                spent=paid,
                status=status,
            )
        )
    return rows


def _usd(value: float) -> str:
    return f"${value:.4f}"


def _source(matrix: AnswerMatrix, typical: Typical) -> str:
    cost = _usd(typical.answer_cost(matrix))
    if not typical.answer_calls:
        return f"matrix estimate (no measured answer calls yet), {cost} per answer"
    source = (
        f"means of {typical.answer_calls} measured answer calls and "
        f"{typical.judge_calls} measured judge calls, {cost} per answer"
    )
    if typical.answer_cost(matrix) > PILOT_ANSWER_LIMIT_USD:
        source += (
            f" (above ${PILOT_ANSWER_LIMIT_USD:.4f}: k3, sparse and k8 run only if their "
            "start check passes)"
        )
    return source


def render(
    matrix: AnswerMatrix, rows: Sequence[PlanRow], typical: Typical, charges: Sequence[Charge]
) -> str:
    total = split_spend(charges)
    lines = [
        "# Answer benchmark budget plan",
        "",
        f"Typical tokens per call: {_source(matrix, typical)}.",
        f"Matrix prices as of {matrix.prices_as_of} ({matrix.prices_source}); "
        f"start margin x{matrix.start_margin}.",
        "",
        "| run | items | turns | judge requests | estimate | needed with margin "
        "| spent measured | spent estimated | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {row.run} | {row.items} | {row.turns} | {row.judge_requests} "
        f"| {_usd(row.estimate)} | {_usd(row.needed)} | {_usd(row.spent.measured)} "
        f"| {_usd(row.spent.estimated)} | {row.status.value} |"
        for row in rows
    ]
    lines += [
        "",
        f"Spent: measured {_usd(total.measured)} + estimated (worst-case bounds) "
        f"{_usd(total.estimated)} = {_usd(total.total)} of cap ${matrix.cap_usd:.2f}; "
        f"{_usd(matrix.cap_usd - total.total)} left.",
        "Measured is priced from stored token counts; estimated is rewrite calls and "
        "unstored turns charged at their token bounds, so the Anthropic console should "
        "show between measured and measured + estimated.",
        "",
    ]
    return "\n".join(lines)


def run_start(
    matrix: AnswerMatrix,
    run: RunSpec,
    items: Mapping[str, Sequence[EvalItem]],
    charges: Sequence[Charge],
) -> StartCheck:
    """The start check a run must pass now, on the numbers `make eval-estimate` shows:
    the rest of the run at typical tokens (measured once the pilot ran), against what
    the cap leaves after all spending and the other reserved runs' reservations.
    """
    tokens = typical_tokens(matrix, charges).tokens
    spent = {name: paid.total for name, paid in split_by_run(charges).items()}
    estimate = run_estimate(matrix, run, items[run.name], tokens)
    return start_check(
        matrix,
        run,
        estimate=max(0.0, estimate - spent.get(run.name, 0.0)),
        spent_by_run=spent,
        reserved=_reserved(matrix, items, tokens),
    )

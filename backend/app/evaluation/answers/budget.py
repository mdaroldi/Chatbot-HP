"""The phase budget (owner's hard cap): an append-only ledger and the guards on it.

Three guards, from coarse to fine:
- Run start: the run's estimate x `start_margin` must fit in what is left after the
  reserved runs (held-out, one improvement re-run) keep their own estimate x margin,
  less what they already spent. A run that does not fit does not start, so cuts fall
  on the late, unreserved rows of the priority list.
- Item start: spent + the item's worst case (every turn's answer and rewrite at their
  token bounds) must fit under the cap. This is the hard stop: it holds even when the
  estimates are wrong.
- Measured call: a charge above its kind's worst-case bound means the hard stop was
  computed from a false bound, so the run stops for the bound to be raised.

Charges are measured tokens priced at the dated matrix prices, or token bounds marked
`estimated` where the backend stores no usage (the rewrite call). Each attempt of an
item is charged (`attempt` counts `--retry-errors` re-asks), so nothing paid is lost.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from app.domain.answer_dataset import EvalItem
from app.evaluation.answers.answer_matrix import AnswerMatrix, RunSpec, Tokens
from app.evaluation.answers.jsonl import Json, append_object, read_objects

_PER_MTOK = 1_000_000
# Float sums of cents must not turn an exact fit into an overrun.
_TOLERANCE_USD = 1e-9


class Kind(StrEnum):
    ANSWER = "answer"
    REWRITE = "rewrite"
    JUDGE = "judge"


class LedgerError(Exception):
    """The ledger cannot be trusted or a charge cannot be priced; nothing may be spent."""


class BudgetExceededError(Exception):
    """The next item could cross the cap at its worst case."""


class UnsoundBoundError(Exception):
    """A measured call exceeded its worst-case bound; the hard stop is not safe."""


class RunNotFundedError(Exception):
    """The run's estimate with margin does not fit in the unreserved budget."""


@dataclass(frozen=True)
class Charge:
    run: str
    item_id: str
    kind: Kind
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    estimated: bool = False  # tokens are a bound, not a measurement
    attempt: int = 1


@dataclass(frozen=True)
class StartCheck:
    run: str
    remaining: float  # unreserved budget left before the run
    needed: float  # the run's estimate x start margin

    @property
    def ok(self) -> bool:
        return self.needed <= self.remaining + _TOLERANCE_USD


def _price(matrix: AnswerMatrix, model: str, input_tokens: int, output_tokens: int) -> float:
    price = matrix.prices.get(model)
    if price is None:
        raise LedgerError(f"no price for {model}")
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / _PER_MTOK


def charge(  # noqa: PLR0913, PLR0917 - every field of a charge is named at the call site
    matrix: AnswerMatrix,
    run: str,
    item_id: str,
    kind: Kind,
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    estimated: bool = False,
    attempt: int = 1,
) -> Charge:
    """Price one call; judge calls go through the Batch API at its discount."""
    cost = _price(matrix, model, input_tokens, output_tokens)
    if kind is Kind.JUDGE:
        cost *= matrix.batch_discount
    return Charge(run, item_id, kind, model, input_tokens, output_tokens, cost, estimated, attempt)


def judge_cost(matrix: AnswerMatrix, requests: int, tokens: Tokens) -> float:
    per = _price(matrix, matrix.models.judge, tokens.judge_input_tokens, tokens.judge_output_tokens)
    return requests * per * matrix.batch_discount


# ---------------------------------------------------------------- ledger


class _FieldError(Exception):
    pass


def _field[T](obj: Json, key: str, kind: type[T]) -> T:
    value = obj.get(key)
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool):
        raise _FieldError(f"field {key} is missing or has the wrong type")
    return value


def _number(obj: Json, key: str) -> float:
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _FieldError(f"field {key} is missing or has the wrong type")
    return float(value)


def _charge_of(obj: Json) -> Charge:
    try:
        kind = Kind(_field(obj, "kind", str))
    except ValueError:
        raise _FieldError("field kind has an unknown value") from None
    made = Charge(
        run=_field(obj, "run", str),
        item_id=_field(obj, "item_id", str),
        kind=kind,
        model=_field(obj, "model", str),
        input_tokens=_field(obj, "input_tokens", int),
        output_tokens=_field(obj, "output_tokens", int),
        cost_usd=_number(obj, "cost_usd"),
        estimated=_field(obj, "estimated", bool),
        attempt=_field(obj, "attempt", int),
    )
    if min(made.input_tokens, made.output_tokens, made.cost_usd) < 0 or made.attempt < 1:
        raise _FieldError("a count, the cost or the attempt is out of range")
    return made


class Ledger:
    """Every paid call of the phase, across runs; git-ignored, append-only."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, made: Charge) -> None:
        append_object(self.path, cast(Json, asdict(made)))

    def charges(self) -> list[Charge]:
        charges = []
        for number, obj in read_objects(self.path, LedgerError):
            try:
                charges.append(_charge_of(obj))
            except _FieldError as exc:
                raise LedgerError(f"line {number}: {exc}") from None
        return charges

    def spent(self) -> float:
        return sum(made.cost_usd for made in self.charges())

    def spent_by_run(self) -> dict[str, float]:
        totals: defaultdict[str, float] = defaultdict(float)
        for made in self.charges():
            totals[made.run] += made.cost_usd
        return dict(totals)


# ---------------------------------------------------------------- per item


def _answer_cost(matrix: AnswerMatrix, tokens: Tokens) -> float:
    return _price(
        matrix, matrix.models.answer, tokens.answer_input_tokens, tokens.answer_output_tokens
    )


def _rewrite_worst(matrix: AnswerMatrix) -> float:
    worst = matrix.worst_case
    return _price(
        matrix, matrix.models.rewrite, worst.rewrite_input_tokens, worst.rewrite_output_tokens
    )


def item_worst_case(matrix: AnswerMatrix, item: EvalItem) -> float:
    """Every turn answered at the token bounds, and every turn after the first rewritten."""
    turns = len(item.turns)
    return turns * _answer_cost(matrix, matrix.worst_case) + (turns - 1) * _rewrite_worst(matrix)


def guard_item(matrix: AnswerMatrix, *, spent: float, worst: float) -> None:
    if spent + worst > matrix.cap_usd + _TOLERANCE_USD:
        raise BudgetExceededError(
            f"the next item could cross the cap: spent ${spent:.4f} + worst case "
            f"${worst:.4f} > ${matrix.cap_usd:.2f}"
        )


def check_measured(matrix: AnswerMatrix, made: Charge) -> None:
    worst = matrix.worst_case
    bounds = {
        Kind.ANSWER: (worst.answer_input_tokens, worst.answer_output_tokens),
        Kind.REWRITE: (worst.rewrite_input_tokens, worst.rewrite_output_tokens),
        Kind.JUDGE: (worst.judge_input_tokens, worst.judge_output_tokens),
    }
    max_in, max_out = bounds[made.kind]
    if made.input_tokens > max_in or made.output_tokens > max_out:
        raise UnsoundBoundError(
            f"a {made.kind.value} call used {made.input_tokens} input and "
            f"{made.output_tokens} output tokens, above its bound ({max_in}, {max_out}); "
            "raise worst_case in answer-matrix.toml before going on"
        )


# ---------------------------------------------------------------- per run


def run_estimate(
    matrix: AnswerMatrix,
    run: RunSpec,
    items: Sequence[EvalItem],
    tokens: Tokens | None = None,
) -> float:
    """Typical cost of a run; `tokens` are measured means (the pilot's) when known."""
    typical = tokens or matrix.estimate
    turns = sum(len(item.turns) for item in items)
    cost = turns * _answer_cost(matrix, typical) + (turns - len(items)) * _rewrite_worst(matrix)
    if run.judge:
        cost += judge_cost(matrix, len(items) + run.judge_repeat, typical)
    return cost


def start_check(
    matrix: AnswerMatrix,
    run: RunSpec,
    *,
    estimate: float,
    spent_by_run: Mapping[str, float],
    reserved: Mapping[str, float],
) -> StartCheck:
    """`reserved` holds each reserved run's estimate; its own spending comes off it."""
    held = sum(
        max(0.0, cost * matrix.start_margin - spent_by_run.get(name, 0.0))
        for name, cost in reserved.items()
        if name != run.name
    )
    remaining = matrix.cap_usd - sum(spent_by_run.values()) - held
    return StartCheck(run.name, remaining, estimate * matrix.start_margin)


def require_start(check: StartCheck) -> None:
    if not check.ok:
        raise RunNotFundedError(
            f"run {check.run} needs ${check.needed:.4f} with margin, "
            f"${check.remaining:.4f} is left after reservations"
        )

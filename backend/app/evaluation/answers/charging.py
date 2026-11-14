"""What one asked item cost, from the rows the backend stored for its conversation.

Measured where the backend stores usage, bounded where it does not, so the ledger
never holds less than was paid:
- An answer call is charged its stored tokens at the model that served it. A layer 1
  refusal made no answer call. A call with no stored usage (a stream that failed
  before reporting it) is charged at the answer bound.
- The rewrite call stores no usage: each row with `rewrite_ms` (set exactly when the
  rewrite model ran) is charged at the rewrite bound.
- A turn that failed before its first token leaves no row (the backend removes it),
  yet its rewrite and the model's thinking may have been paid: each such turn is
  charged at the turn bound (the rewrite bound unless it is the first turn, plus the
  answer bound). An answered item missing rows is treated the same way.
- Each retry (429/503 before a first token) is charged one rewrite bound, the part of
  a removed attempt that can have finished before it failed.
- With no conversation the item never reached a model: nothing is charged.

Bounded charges are marked `estimated`; every charge carries the attempt number.
"""

from collections.abc import Sequence

from app.domain.answer_dataset import EvalItem
from app.domain.chat import Outcome, RefusalReason
from app.evaluation.answers.answer_matrix import AnswerMatrix
from app.evaluation.answers.budget import Charge, Kind, charge
from app.evaluation.answers.client import ItemResult
from app.evaluation.answers.store import StoredAnswer

# Outcomes stored for a turn that failed after its first token: that turn has its row.
_STOPPED = frozenset({Outcome.FAILED, Outcome.CANCELLED})


class _Charger:
    def __init__(self, matrix: AnswerMatrix, run: str, item_id: str, attempt: int) -> None:
        self._matrix, self._run, self._item_id, self._attempt = matrix, run, item_id, attempt

    def measured_answer(self, model: str, input_tokens: int, output_tokens: int) -> Charge:
        return self._charge(Kind.ANSWER, model, input_tokens, output_tokens, estimated=False)

    def worst_answer(self) -> Charge:
        worst = self._matrix.worst_case
        return self._charge(
            Kind.ANSWER,
            self._matrix.models.answer,
            worst.answer_input_tokens,
            worst.answer_output_tokens,
            estimated=True,
        )

    def worst_rewrite(self) -> Charge:
        worst = self._matrix.worst_case
        return self._charge(
            Kind.REWRITE,
            self._matrix.models.rewrite,
            worst.rewrite_input_tokens,
            worst.rewrite_output_tokens,
            estimated=True,
        )

    def _charge(
        self, kind: Kind, model: str, input_tokens: int, output_tokens: int, *, estimated: bool
    ) -> Charge:
        return charge(
            self._matrix,
            self._run,
            self._item_id,
            kind,
            model,
            input_tokens,
            output_tokens,
            estimated=estimated,
            attempt=self._attempt,
        )


def _row_charges(charger: _Charger, row: StoredAnswer) -> list[Charge]:
    charges = [] if row.rewrite_ms is None else [charger.worst_rewrite()]
    if row.refusal_reason is RefusalReason.BELOW_THRESHOLD:
        return charges
    if row.model is None or row.input_tokens is None or row.output_tokens is None:
        return [*charges, charger.worst_answer()]
    return [*charges, charger.measured_answer(row.model, row.input_tokens, row.output_tokens)]


def _attempted_turns(item: EvalItem, result: ItemResult, rows: Sequence[StoredAnswer]) -> int:
    if result.ok:
        return len(item.turns)
    failed_with_row = bool(rows) and rows[-1].outcome in _STOPPED
    return len(rows) if failed_with_row else min(len(item.turns), len(rows) + 1)


def item_charges(  # noqa: PLR0913 - the item, what the API said and what the store holds
    matrix: AnswerMatrix,
    run: str,
    item: EvalItem,
    result: ItemResult,
    rows: Sequence[StoredAnswer],
    *,
    attempt: int,
) -> list[Charge]:
    if result.conversation_id is None:
        return []
    charger = _Charger(matrix, run, item.id, attempt)
    charges = [made for row in rows for made in _row_charges(charger, row)]
    for turn in range(len(rows), _attempted_turns(item, result, rows)):
        if turn > 0:
            charges.append(charger.worst_rewrite())
        charges.append(charger.worst_answer())
    charges.extend(charger.worst_rewrite() for _ in range(result.retries))
    return charges

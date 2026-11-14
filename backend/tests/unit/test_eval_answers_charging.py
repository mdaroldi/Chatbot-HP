"""What one asked item cost: measured charges from its stored rows, and bounds where the
backend stores nothing (the rewrite call, turns removed before their first token).
"""

import dataclasses

import pytest

from app.domain.chat import Outcome, RefusalReason
from app.evaluation.answers.budget import Charge, Kind, LedgerError, charge
from app.evaluation.answers.charging import item_charges
from app.evaluation.answers.client import ItemResult
from app.evaluation.answers.store import StoredAnswer
from tests.unit.test_eval_answers_budget import HAIKU, MATRIX, SONNET, item

RUN = "reference-1"
WORST = MATRIX.worst_case
SERVED = SONNET  # the run's config check holds the stored model to the matrix model


def row(n: int = 1, **overrides: object) -> StoredAnswer:
    fields: dict[str, object] = {
        "message_id": f"m-{n}",
        "outcome": Outcome.COMPLETED,
        "refusal_reason": None,
        "model": SERVED,
        "input_tokens": 4000 + n,
        "output_tokens": 500 + n,
        "chunk_ids": ("c-1", "c-2"),
        "rewritten_query": None,
        "rewrite_ms": None,
        "index_build_id": "20261113t173108z",
        "relevance_signal": -1.0,
        "retrieval_ms": 40.0,
        "ttft_ms": 600.0,
        "generation_ms": 2000.0,
        "total_ms": 2700.0,
    }
    fields.update(overrides)
    return StoredAnswer(**fields)  # type: ignore[arg-type]  # fields typed per test


def result(
    *, conversation: str | None = "conv-1", error: str | None = None, retries: int = 0
) -> ItemResult:
    return ItemResult("q-1", "client-1", conversation, retries, None, error)


def measured(stored: StoredAnswer, attempt: int = 1) -> Charge:
    assert stored.input_tokens is not None
    assert stored.output_tokens is not None
    return charge(
        MATRIX,
        RUN,
        "q-1",
        Kind.ANSWER,
        SERVED,
        stored.input_tokens,
        stored.output_tokens,
        attempt=attempt,
    )


def worst_answer(attempt: int = 1) -> Charge:
    return charge(
        MATRIX,
        RUN,
        "q-1",
        Kind.ANSWER,
        SONNET,
        WORST.answer_input_tokens,
        WORST.answer_output_tokens,
        estimated=True,
        attempt=attempt,
    )


def worst_rewrite(attempt: int = 1) -> Charge:
    return charge(
        MATRIX,
        RUN,
        "q-1",
        Kind.REWRITE,
        HAIKU,
        WORST.rewrite_input_tokens,
        WORST.rewrite_output_tokens,
        estimated=True,
        attempt=attempt,
    )


def test_an_answered_turn_is_charged_its_measured_tokens_at_the_served_model() -> None:
    stored = row()

    assert item_charges(MATRIX, RUN, item(), result(), [stored], attempt=1) == [measured(stored)]


def test_a_served_model_with_no_price_cannot_be_charged_silently() -> None:
    with pytest.raises(LedgerError, match="no price for other-model"):
        item_charges(MATRIX, RUN, item(), result(), [row(model="other-model")], attempt=1)


def test_a_rewrite_that_ran_is_charged_at_its_bound_and_marked_estimated() -> None:
    first, second = row(1), row(2, rewrite_ms=150.0, rewritten_query="standalone")

    charges = item_charges(MATRIX, RUN, item(turns=2), result(), [first, second], attempt=2)

    assert charges == [measured(first, 2), worst_rewrite(2), measured(second, 2)]


def test_a_layer_one_refusal_costs_no_answer_call_but_its_rewrite_is_charged() -> None:
    refused = row(
        2,
        outcome=Outcome.REFUSED,
        refusal_reason=RefusalReason.BELOW_THRESHOLD,
        model=None,
        input_tokens=None,
        output_tokens=None,
        rewrite_ms=90.0,
    )

    charges = item_charges(MATRIX, RUN, item(turns=2), result(), [row(1), refused], attempt=1)

    assert charges == [measured(row(1)), worst_rewrite()]


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (Outcome.REFUSED, RefusalReason.MODEL_REFUSED),
        (Outcome.REFUSED, RefusalReason.PROVIDER_REFUSED),
        (Outcome.COMPLETED, None),
    ],
)
def test_a_model_refusal_is_a_paid_answer_call(
    outcome: Outcome, reason: RefusalReason | None
) -> None:
    stored = row(outcome=outcome, refusal_reason=reason)

    assert item_charges(MATRIX, RUN, item(), result(), [stored], attempt=1) == [measured(stored)]


@pytest.mark.parametrize("missing", ["input_tokens", "output_tokens", "model"])
def test_an_answer_call_with_no_stored_usage_is_charged_at_its_bound(missing: str) -> None:
    stored = row(1, outcome=Outcome.FAILED, **{missing: None})

    charges = item_charges(MATRIX, RUN, item(), result(error="stream_error"), [stored], attempt=1)

    assert charges == [worst_answer()]


def test_without_a_conversation_no_model_was_called_but_a_first_turn_error_was_paid() -> None:
    no_conversation = result(conversation=None, error="http_error", retries=2)
    first_turn_error = result(error="read_timeout")

    assert item_charges(MATRIX, RUN, item(turns=3), no_conversation, [], attempt=1) == []
    assert item_charges(MATRIX, RUN, item(turns=3), first_turn_error, [], attempt=1) == [
        worst_answer()
    ]


def test_a_turn_removed_before_its_first_token_is_charged_at_the_turn_bound() -> None:
    """The backend keeps nothing of such a turn, yet its rewrite and the model's
    thinking may have been paid.
    """
    charges = item_charges(
        MATRIX, RUN, item(turns=3), result(error="read_timeout"), [row(1)], attempt=1
    )

    assert charges == [measured(row(1)), worst_rewrite(), worst_answer()]


@pytest.mark.parametrize("outcome", [Outcome.FAILED, Outcome.CANCELLED])
def test_a_turn_stored_after_its_first_token_is_not_charged_twice(outcome: Outcome) -> None:
    stopped = row(2, outcome=outcome)

    charges = item_charges(
        MATRIX, RUN, item(turns=3), result(error="read_timeout"), [row(1), stopped], attempt=1
    )

    assert charges == [measured(row(1)), measured(stopped)]


def test_an_error_on_the_first_turn_before_a_token_costs_no_rewrite() -> None:
    charges = item_charges(
        MATRIX, RUN, item(turns=2), result(error="transport_error"), [], attempt=1
    )

    assert charges == [worst_answer()]


def test_an_answered_item_missing_rows_has_them_charged_at_the_bound() -> None:
    charges = item_charges(MATRIX, RUN, item(turns=2), result(), [row(1)], attempt=1)

    assert charges == [measured(row(1)), worst_rewrite(), worst_answer()]


def test_each_retry_before_a_first_token_is_charged_one_rewrite_bound() -> None:
    charges = item_charges(MATRIX, RUN, item(), result(retries=2), [row()], attempt=1)

    assert charges == [measured(row()), worst_rewrite(), worst_rewrite()]


def test_every_charge_names_the_run_the_item_and_the_attempt() -> None:
    charges = item_charges(
        MATRIX, "dense", item(turns=2, item_id="mt-7"), result(retries=1), [row(1)], attempt=3
    )

    assert {(c.run, c.item_id, c.attempt) for c in charges} == {("dense", "mt-7", 3)}
    assert [dataclasses.replace(c, run=RUN, item_id="q-1", attempt=1) for c in charges] == [
        measured(row(1)),
        worst_rewrite(),
        worst_answer(),
        worst_rewrite(),
    ]

"""The budget plan behind `make eval-estimate`: typical tokens (measured once the pilot
has run), each run's estimate in priority order, which runs the cap still funds, and
the spend so far split into measured and estimated (worst-case) charges.
"""

import pytest

from app.evaluation.answers.answer_matrix import Tokens, parse_answer_matrix
from app.evaluation.answers.budget import Charge, Kind, Spend, charge, run_estimate
from app.evaluation.answers.estimate import (
    PlanRow,
    Status,
    Typical,
    plan,
    render,
    run_start,
    typical_tokens,
)
from tests.unit.test_eval_answers_budget import HAIKU, MATRIX, OPUS, SONNET, item
from tests.unit.test_eval_answers_matrix import BASE, DENSE, HELDOUT, REFERENCE

ITEMS = {
    "reference-1": [item(1, "q-1"), item(2, "q-2")],
    "heldout": [item(1, "h-1")],
    "dense": [item(1, "q-1")],
}


def answer(run: str, tokens_in: int, tokens_out: int, *, estimated: bool = False) -> Charge:
    return charge(
        MATRIX, run, "q-1", Kind.ANSWER, SONNET, tokens_in, tokens_out, estimated=estimated
    )


# ------------------------------------------------------------------ typical tokens


def test_without_measured_calls_the_matrix_estimate_is_typical() -> None:
    assert typical_tokens(MATRIX, []) == Typical(MATRIX.estimate, answer_calls=0, judge_calls=0)


def test_measured_answer_and_judge_means_replace_the_estimate_rounded_up() -> None:
    charges = [
        answer("reference-1", 3000, 400),
        answer("reference-1", 4001, 501),
        answer("reference-1", 12000, 2048, estimated=True),  # a bound, not a measurement
        charge(MATRIX, "reference-1", "q-1", Kind.REWRITE, HAIKU, 4000, 100, estimated=True),
        charge(MATRIX, "reference-1", "q-1", Kind.JUDGE, OPUS, 6000, 1000),
    ]
    assert typical_tokens(MATRIX, charges) == Typical(
        Tokens(
            answer_input_tokens=3501,
            answer_output_tokens=451,
            judge_input_tokens=6000,
            judge_output_tokens=1000,
            rewrite_input_tokens=MATRIX.estimate.rewrite_input_tokens,
            rewrite_output_tokens=MATRIX.estimate.rewrite_output_tokens,
        ),
        answer_calls=2,
        judge_calls=1,
    )


def test_the_mean_answer_cost_is_reported_for_the_pilot_rule() -> None:
    typical = typical_tokens(MATRIX, [answer("reference-1", 3000, 400)])
    assert typical.answer_cost(MATRIX) == pytest.approx((3000 * 2.0 + 400 * 10.0) / 1e6)


# ------------------------------------------------------------------ plan


THREE = parse_answer_matrix(BASE + REFERENCE + HELDOUT + DENSE)


def test_rows_follow_the_matrix_order_with_their_estimates() -> None:
    reference = [*ITEMS["reference-1"], item(1, "q-3", in_scope=False)]
    rows = plan(THREE, {**ITEMS, "reference-1": reference}, [], THREE.estimate)
    assert [row.run for row in rows] == ["reference-1", "heldout", "dense"]
    first = rows[0]
    expected = run_estimate(THREE, THREE.run("reference-1"), reference)
    assert first == PlanRow(
        run="reference-1",
        items=3,
        turns=4,
        judge_requests=2 + 20,  # the out-of-scope item is not judged
        estimate=pytest.approx(expected),  # type: ignore[arg-type]
        needed=pytest.approx(expected * 1.25),  # type: ignore[arg-type]
        spent=Spend(0.0, 0.0),
        status=Status.FUNDED,
    )
    assert rows[1].status is Status.RESERVED
    assert rows[2].judge_requests == 0


def test_spent_money_is_shown_per_run_split_measured_and_estimated() -> None:
    measured = answer("dense", 3000, 400)
    bound = answer("dense", 12000, 2048, estimated=True)
    (_, _, dense) = plan(THREE, ITEMS, [measured, bound], THREE.estimate)
    assert dense.spent == Spend(measured.cost_usd, bound.cost_usd)


def test_a_late_run_the_cap_cannot_fund_after_earlier_runs_is_cut() -> None:
    poor = parse_answer_matrix(
        (BASE + REFERENCE + HELDOUT + DENSE).replace("cap_usd = 10.0", "cap_usd = 0.80")
    )
    reference, heldout, dense = plan(poor, ITEMS, [], poor.estimate)
    # held-out is funded first; reference-1 fits after it, dense no longer does.
    assert (reference.status, heldout.status, dense.status) == (
        Status.FUNDED,
        Status.RESERVED,
        Status.CUT,
    )


def test_a_run_already_paid_for_needs_only_what_is_left() -> None:
    estimate = run_estimate(THREE, THREE.run("dense"), ITEMS["dense"])
    paid = answer("dense", 4500, 600)  # exactly the estimate of its one item
    (_, _, dense) = plan(THREE, ITEMS, [paid], THREE.estimate)
    assert paid.cost_usd == pytest.approx(estimate)
    assert dense.needed == pytest.approx(0.0)
    assert dense.status is Status.FUNDED


# ------------------------------------------------------------------ run start


def test_a_run_start_holds_back_the_reserved_runs_and_counts_all_spending() -> None:
    paid = answer("dense", 4500, 600)
    check = run_start(THREE, THREE.run("reference-1"), ITEMS, [paid])
    heldout = run_estimate(THREE, THREE.run("heldout"), ITEMS["heldout"])
    reference = run_estimate(THREE, THREE.run("reference-1"), ITEMS["reference-1"])
    assert check.run == "reference-1"
    assert check.remaining == pytest.approx(10.0 - paid.cost_usd - heldout * 1.25)
    assert check.needed == pytest.approx(reference * 1.25)


def test_a_run_start_prices_the_rest_of_the_run_at_measured_tokens() -> None:
    paid = answer("reference-1", 3000, 300)
    measured = THREE.estimate.model_copy(
        update={"answer_input_tokens": 3000, "answer_output_tokens": 300}
    )
    estimate = run_estimate(THREE, THREE.run("reference-1"), ITEMS["reference-1"], measured)
    check = run_start(THREE, THREE.run("reference-1"), ITEMS, [paid])
    assert check.needed == pytest.approx((estimate - paid.cost_usd) * 1.25)


def test_a_reserved_run_does_not_hold_back_its_own_reservation() -> None:
    check = run_start(THREE, THREE.run("heldout"), ITEMS, [])
    heldout = run_estimate(THREE, THREE.run("heldout"), ITEMS["heldout"])
    assert check.remaining == pytest.approx(10.0)
    assert check.needed == pytest.approx(heldout * 1.25)


# ------------------------------------------------------------------ render


def test_the_report_shows_rows_spend_split_and_the_token_source() -> None:
    measured = answer("dense", 3000, 400)
    bound = charge(THREE, "dense", "q-1", Kind.REWRITE, HAIKU, 4000, 100, estimated=True)
    charges = [measured, bound]
    typical = typical_tokens(THREE, charges)
    text = render(THREE, plan(THREE, ITEMS, charges, typical.tokens), typical, charges)

    assert "| reference-1 | 2 | 3 | 22 |" in text
    assert "| heldout |" in text
    assert "reserved" in text
    assert f"measured ${measured.cost_usd:.4f}" in text
    assert f"estimated (worst-case bounds) ${bound.cost_usd:.4f}" in text
    assert "cap $10.00" in text
    assert "1 measured answer calls" in text
    assert f"${typical.answer_cost(THREE):.4f} per answer" in text
    assert "prices as of 2026-11-03" in text


CONDITIONAL_NOTE = "k3, sparse and k8 run only if their start check passes"


@pytest.mark.parametrize(
    ("tokens_in", "tokens_out", "noted"),
    [(3000, 400, True), (1000, 100, False)],  # $0.0100 and $0.0030 per answer
)
def test_the_conditional_rows_are_noted_only_above_the_pilot_limit(
    tokens_in: int, tokens_out: int, noted: bool
) -> None:
    charges = [answer("dense", tokens_in, tokens_out)]
    typical = typical_tokens(THREE, charges)
    text = render(THREE, plan(THREE, ITEMS, charges, typical.tokens), typical, charges)
    assert (CONDITIONAL_NOTE in text) is noted
    assert "drop" not in text


def test_before_the_pilot_the_report_says_tokens_are_the_matrix_estimate() -> None:
    typical = typical_tokens(THREE, [])
    text = render(THREE, plan(THREE, ITEMS, [], typical.tokens), typical, [])
    assert "matrix estimate (no measured answer calls yet)" in text
    assert "measured $0.0000" in text

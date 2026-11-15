"""The phase budget: an append-only ledger of charges, the per-item hard stop, the
measured-versus-bound check, run estimates and the start check with reserved runs.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from app.domain.answer_dataset import EvalItem
from app.domain.evaluation import Split
from app.evaluation.answers.answer_matrix import (
    AnswerMatrix,
    ItemSet,
    RunSpec,
    Tokens,
    parse_answer_matrix,
)
from app.evaluation.answers.budget import (
    BudgetExceededError,
    Charge,
    Kind,
    Ledger,
    LedgerError,
    RunNotFundedError,
    Spend,
    UnsoundBoundError,
    charge,
    check_measured,
    guard_item,
    item_worst_case,
    judge_cost,
    judge_requests,
    require_start,
    run_estimate,
    split_by_run,
    split_spend,
    start_check,
)
from tests.unit.test_eval_answers_matrix import BASE, HELDOUT, REFERENCE

MATRIX: AnswerMatrix = parse_answer_matrix(BASE + REFERENCE + HELDOUT)
SONNET, HAIKU, OPUS = "claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"


def item(turns: int = 1, item_id: str = "q-1", *, in_scope: bool = True) -> EvalItem:
    return EvalItem(
        id=item_id,
        split=Split.DEV,
        turns=tuple(f"turn {n}" for n in range(turns)),
        in_scope=in_scope,
    )


def answer_charge(run: str = "reference-1", cost_in: int = 1000, cost_out: int = 100) -> Charge:
    return charge(MATRIX, run, "q-1", Kind.ANSWER, SONNET, cost_in, cost_out)


# ------------------------------------------------------------------ costs


def test_a_charge_prices_measured_tokens_at_the_model_rates() -> None:
    made = charge(MATRIX, "reference-1", "q-1", Kind.ANSWER, SONNET, 4000, 500)

    # 4000 x $2/M + 500 x $10/M
    assert made.cost_usd == pytest.approx(0.008 + 0.005)
    assert (made.run, made.item_id, made.kind, made.model) == (
        "reference-1",
        "q-1",
        "answer",
        SONNET,
    )
    assert not made.estimated
    assert made.attempt == 1


def test_a_judge_charge_gets_the_batch_discount() -> None:
    made = charge(MATRIX, "reference-1", "q-1", Kind.JUDGE, OPUS, 5000, 1000)

    assert made.cost_usd == pytest.approx((5000 * 4 + 1000 * 20) / 1e6 * 0.5)


def test_an_estimated_charge_and_a_second_attempt_are_marked() -> None:
    made = charge(MATRIX, "r", "q-1", Kind.REWRITE, HAIKU, 4000, 100, estimated=True, attempt=2)

    assert made.estimated
    assert made.attempt == 2
    assert made.cost_usd == pytest.approx((4000 * 0.1 + 100 * 0.5) / 1e6)


def test_a_model_without_a_price_cannot_be_charged() -> None:
    with pytest.raises(LedgerError, match="no price for claude-x"):
        charge(MATRIX, "r", "q-1", Kind.ANSWER, "claude-x", 1, 1)


def test_the_judge_cost_of_n_requests_uses_the_estimate_and_the_discount() -> None:
    per = (5000 * 4 + 1500 * 20) / 1e6 * 0.5
    assert judge_cost(MATRIX, 10, MATRIX.estimate) == pytest.approx(10 * per)


# ------------------------------------------------------------------ ledger


def test_charges_are_appended_and_summed(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    first, second = answer_charge(), answer_charge(run="reference-2")

    ledger.append(first)
    ledger.append(second)

    assert ledger.charges() == [first, second]
    assert ledger.spent() == pytest.approx(first.cost_usd + second.cost_usd)
    assert ledger.spent_by_run() == {
        "reference-1": pytest.approx(first.cost_usd),
        "reference-2": pytest.approx(second.cost_usd),
    }
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 2


def test_a_missing_ledger_has_spent_nothing(tmp_path: Path) -> None:
    assert Ledger(tmp_path / "ledger.jsonl").spent() == 0.0


def _raw() -> dict[str, object]:
    return json.loads(json.dumps(dataclasses.asdict(answer_charge())))  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        "[]",
        '{"run": "r"}',
        json.dumps({**_raw(), "cost_usd": -1}),
        json.dumps({**_raw(), "cost_usd": "0.01"}),
        json.dumps({**_raw(), "input_tokens": True}),
        json.dumps({**_raw(), "kind": "lunch"}),
        json.dumps({**_raw(), "attempt": 0}),
    ],
)
def test_a_bad_ledger_line_stops_everything(tmp_path: Path, line: str) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(LedgerError, match="line 1"):
        Ledger(path).spent()


def test_a_partial_last_ledger_line_stops_everything(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    Ledger(path).append(answer_charge())
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"run": ')

    with pytest.raises(LedgerError, match="line 2: partial"):
        Ledger(path).spent()


# ------------------------------------------------------------------ per item


def test_the_worst_case_of_an_item_covers_every_turn_and_rewrite() -> None:
    worst = MATRIX.worst_case
    answer = (worst.answer_input_tokens * 2 + worst.answer_output_tokens * 10) / 1e6
    rewrite = (worst.rewrite_input_tokens * 0.1 + worst.rewrite_output_tokens * 0.5) / 1e6

    assert item_worst_case(MATRIX, item(1)) == pytest.approx(answer)
    assert item_worst_case(MATRIX, item(3)) == pytest.approx(3 * answer + 2 * rewrite)


def test_an_item_that_fits_under_the_cap_may_start() -> None:
    guard_item(MATRIX, spent=9.90, worst=0.10)


def test_an_item_that_could_cross_the_cap_does_not_start() -> None:
    with pytest.raises(BudgetExceededError, match="cap"):
        guard_item(MATRIX, spent=9.95, worst=0.06)


@pytest.mark.parametrize(
    ("kind", "model", "tokens_in", "tokens_out"),
    [
        (Kind.ANSWER, SONNET, 12001, 10),
        (Kind.ANSWER, SONNET, 10, 2049),
        (Kind.REWRITE, HAIKU, 4001, 10),
        (Kind.JUDGE, OPUS, 10, 4001),
    ],
)
def test_a_call_above_its_bound_makes_the_hard_stop_unsound(
    kind: Kind, model: str, tokens_in: int, tokens_out: int
) -> None:
    made = charge(MATRIX, "r", "q-1", kind, model, tokens_in, tokens_out)

    with pytest.raises(UnsoundBoundError, match=kind.value):
        check_measured(MATRIX, made)


def test_a_call_at_its_bound_is_fine() -> None:
    check_measured(MATRIX, charge(MATRIX, "r", "q-1", Kind.ANSWER, SONNET, 12000, 2048))


# ------------------------------------------------------------------ per run


def spec(name: str, items: ItemSet = ItemSet.DEV, **fields: object) -> RunSpec:
    return RunSpec.model_validate({"name": name, "items": items.value, **fields})


def test_a_run_estimate_counts_answers_rewrites_and_judging() -> None:
    est = MATRIX.estimate
    answer = (est.answer_input_tokens * 2 + est.answer_output_tokens * 10) / 1e6
    worst = MATRIX.worst_case
    rewrite = (worst.rewrite_input_tokens * 0.1 + worst.rewrite_output_tokens * 0.5) / 1e6
    judged = (est.judge_input_tokens * 4 + est.judge_output_tokens * 20) / 1e6 * 0.5
    items = [item(1, "a"), item(2, "b")]

    plain = run_estimate(MATRIX, spec("plain"), items)
    judged_run = run_estimate(MATRIX, spec("j", judge=True, judge_repeat=1), items)

    assert plain == pytest.approx(3 * answer + rewrite)
    assert judged_run == pytest.approx(plain + 3 * judged)


def test_only_in_scope_items_are_judged_plus_the_repeat_sample() -> None:
    # Lever 1a: refusals are scored in code, so out-of-scope items never reach the judge.
    items = [item(1, "a"), item(1, "b", in_scope=False), item(1, "c")]
    est = MATRIX.estimate
    judged = (est.judge_input_tokens * 4 + est.judge_output_tokens * 20) / 1e6 * 0.5

    assert judge_requests(spec("j", judge=True, judge_repeat=1), items) == 2 + 1
    assert judge_requests(spec("plain"), items) == 0
    assert run_estimate(MATRIX, spec("j", judge=True, judge_repeat=1), items) == pytest.approx(
        run_estimate(MATRIX, spec("plain"), items) + 3 * judged
    )


def test_measured_means_can_replace_the_matrix_estimate() -> None:
    measured = Tokens(
        answer_input_tokens=1000,
        answer_output_tokens=100,
        judge_input_tokens=1,
        judge_output_tokens=1,
    )

    cost = run_estimate(MATRIX, spec("plain"), [item()], measured)

    assert cost == pytest.approx((1000 * 2 + 100 * 10) / 1e6)


def test_a_run_starts_when_its_estimate_with_margin_fits_after_reservations() -> None:
    reserved = {"heldout": 1.0}

    check = start_check(
        MATRIX, spec("dense"), estimate=6.0, spent_by_run={"pilot": 1.0}, reserved=reserved
    )

    # 10 - 1 spent - 1.0 x 1.25 reserved = 7.75 left; 6.0 x 1.25 = 7.5 needed.
    assert check.remaining == pytest.approx(7.75)
    assert check.needed == pytest.approx(7.5)
    assert check.ok
    require_start(check)


def test_a_run_that_does_not_fit_with_margin_is_not_started() -> None:
    check = start_check(
        MATRIX, spec("dense"), estimate=6.3, spent_by_run={"pilot": 1.0}, reserved={"heldout": 1.0}
    )

    assert not check.ok
    with pytest.raises(RunNotFundedError, match="dense"):
        require_start(check)


def test_a_reserved_run_does_not_reserve_against_itself() -> None:
    heldout = spec("heldout", ItemSet.HELDOUT, judge=True, reserved=True)

    check = start_check(
        MATRIX, heldout, estimate=7.0, spent_by_run={"pilot": 1.0}, reserved={"heldout": 7.0}
    )

    assert check.remaining == pytest.approx(9.0)
    assert check.needed == pytest.approx(8.75)
    assert check.ok


def test_what_a_reserved_run_already_spent_comes_off_its_reservation() -> None:
    check = start_check(
        MATRIX,
        spec("dense"),
        estimate=1.0,
        spent_by_run={"pilot": 1.0, "heldout": 1.0},
        reserved={"heldout": 2.0},
    )

    # Reservation left: 2.0 x 1.25 - 1.0 = 1.5; remaining 10 - 2 - 1.5 = 6.5.
    assert check.remaining == pytest.approx(6.5)


# ------------------------------------------------------------------ measured vs estimated


def test_spend_is_split_into_measured_and_estimated_charges() -> None:
    measured = answer_charge()
    bound = charge(MATRIX, "reference-1", "q-1", Kind.REWRITE, HAIKU, 4000, 100, estimated=True)
    other = charge(MATRIX, "reference-2", "q-1", Kind.ANSWER, SONNET, 2000, 50)

    total = split_spend([measured, bound, other])

    assert total == Spend(
        measured=pytest.approx(measured.cost_usd + other.cost_usd),  # type: ignore[arg-type]
        estimated=pytest.approx(bound.cost_usd),  # type: ignore[arg-type]
    )
    assert total.total == pytest.approx(measured.cost_usd + other.cost_usd + bound.cost_usd)
    assert split_by_run([measured, bound, other]) == {
        "reference-1": Spend(measured.cost_usd, bound.cost_usd),
        "reference-2": Spend(other.cost_usd, 0.0),
    }


def test_no_charges_spend_nothing() -> None:
    assert split_spend([]) == Spend(0.0, 0.0)
    assert split_by_run([]) == {}

"""Profile B's spend guard: a question is sent only when its worst-case provider charge
still fits the cap and the question count is under the hard stop; the run is told to
stop exactly once."""

import pathlib
import tomllib

import pytest

from hpdocs_load.budget import (
    ANSWER_CALL_MICRO_USD,
    PROFILE_B,
    REWRITE_CALL_MICRO_USD,
    Budget,
    SpendGuard,
    admission,
    admit_all,
)

MATRIX = pathlib.Path(__file__).resolve().parents[2] / "eval" / "answer-matrix.toml"


def guard(**kwargs: int) -> tuple[SpendGuard, list[str]]:
    values = {
        "max_questions": 90,
        "cap_micro_usd": 1_000,
        "answer_call_micro_usd": 100,
        "rewrite_call_micro_usd": 10,
    } | kwargs
    stops: list[str] = []
    return SpendGuard(Budget(**values), lambda: stops.append("stop")), stops


# ---------------------------------------------------------------- the binding numbers


def test_profile_b_is_capped_at_three_dollars_and_ninety_questions() -> None:
    assert PROFILE_B.max_questions == 90
    assert PROFILE_B.cap_micro_usd == 3_000_000


def test_each_call_is_charged_the_phase_5_worst_case_at_the_dated_prices() -> None:
    matrix = tomllib.loads(MATRIX.read_text())
    worst, prices, models = matrix["worst_case"], matrix["prices"], matrix["models"]

    def micro_usd(model: str, kind: str) -> float:
        price = prices[models[model]]
        return float(
            worst[f"{kind}_input_tokens"] * price["input_per_mtok"]
            + worst[f"{kind}_output_tokens"] * price["output_per_mtok"]
        )

    assert micro_usd("answer", "answer") == pytest.approx(ANSWER_CALL_MICRO_USD)
    assert micro_usd("rewrite", "rewrite") == pytest.approx(REWRITE_CALL_MICRO_USD)
    assert PROFILE_B.answer_call_micro_usd == ANSWER_CALL_MICRO_USD
    assert PROFILE_B.rewrite_call_micro_usd == REWRITE_CALL_MICRO_USD


def test_the_cap_binds_before_the_question_stop_at_worst_case() -> None:
    # 3,000,000 / 44,930 per follow-up: the dollar cap, not the 90, ends a real run.
    spend = SpendGuard(PROFILE_B, lambda: None)
    admitted = sum(spend.admit(follow_up=True) for _ in range(200))

    assert admitted == 66
    assert spend.charged_micro_usd <= 3_000_000


# ---------------------------------------------------------------- charging


def test_a_first_question_is_one_answer_call() -> None:
    spend, _ = guard()

    assert spend.admit(follow_up=False) is True
    assert (spend.questions, spend.calls, spend.charged_micro_usd) == (1, 1, 100)


def test_a_follow_up_adds_its_rewrite_call() -> None:
    spend, _ = guard()

    assert spend.admit(follow_up=True) is True
    assert (spend.questions, spend.calls, spend.charged_micro_usd) == (1, 2, 110)


def test_charges_add_up_over_questions() -> None:
    spend, _ = guard()

    spend.admit(follow_up=False)
    spend.admit(follow_up=True)
    spend.admit(follow_up=True)

    assert (spend.questions, spend.calls, spend.charged_micro_usd) == (3, 5, 320)


# ---------------------------------------------------------------- the stops


def test_a_charge_that_lands_exactly_on_the_cap_is_admitted() -> None:
    spend, stops = guard(cap_micro_usd=200)

    assert spend.admit(follow_up=False) is True
    assert spend.admit(follow_up=False) is True
    assert spend.charged_micro_usd == 200
    assert stops == []


def test_a_charge_that_would_pass_the_cap_is_refused_and_not_charged() -> None:
    spend, stops = guard(cap_micro_usd=209)

    assert spend.admit(follow_up=False) is True
    assert spend.admit(follow_up=True) is False
    assert (spend.questions, spend.calls, spend.charged_micro_usd) == (1, 1, 100)
    assert stops == ["stop"]


def test_the_question_stop_refuses_the_next_question() -> None:
    spend, stops = guard(max_questions=2)

    assert spend.admit(follow_up=False) is True
    assert spend.admit(follow_up=False) is True
    assert spend.admit(follow_up=False) is False
    assert spend.questions == 2
    assert stops == ["stop"]


def test_once_stopped_nothing_more_is_admitted_even_if_it_would_fit() -> None:
    # A follow-up was refused at the cap; a cheaper first question must not slip in.
    spend, stops = guard(cap_micro_usd=205)

    assert spend.admit(follow_up=False) is True
    assert spend.admit(follow_up=True) is False
    assert spend.admit(follow_up=False) is False
    assert spend.questions == 1
    assert stops == ["stop"]


def test_the_run_is_told_to_stop_exactly_once() -> None:
    spend, stops = guard(max_questions=1)

    spend.admit(follow_up=False)
    for _ in range(5):
        spend.admit(follow_up=False)

    assert stops == ["stop"]
    assert spend.stopped is True


def test_a_fresh_guard_has_not_stopped() -> None:
    spend, stops = guard()

    assert spend.stopped is False
    assert stops == []


# ---------------------------------------------------------------- record and switch


def test_the_summary_records_counts_charge_cap_and_stop() -> None:
    spend, _ = guard(max_questions=1)
    spend.admit(follow_up=True)
    spend.admit(follow_up=False)

    assert spend.summary() == {
        "questions": 1,
        "provider_calls": 2,
        "charged_worst_case_usd": 0.00011,
        "cap_usd": 0.001,
        "max_questions": 1,
        "stopped": True,
    }


def test_admission_is_the_guard_only_when_the_budget_is_on() -> None:
    spend, _ = guard()

    assert admission(enabled=True, guard=spend) == spend.admit
    assert admission(enabled=False, guard=spend) is admit_all


def test_without_the_budget_every_question_is_admitted() -> None:
    assert admit_all(follow_up=False) is True
    assert admit_all(follow_up=True) is True

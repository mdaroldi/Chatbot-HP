"""Profile B's spend guard: a question is sent only when its worst-case provider charge
still fits the cap and the question count is under the hard stop. Once it refuses, the
run's shape ends the run on Locust's normal stop path."""

import ast
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
    until_spent,
)

MATRIX = pathlib.Path(__file__).resolve().parents[2] / "eval" / "answer-matrix.toml"
LOCUSTFILE = pathlib.Path(__file__).resolve().parents[1] / "locustfile.py"


def guard(**kwargs: int) -> SpendGuard:
    values = {
        "max_questions": 90,
        "cap_micro_usd": 1_000,
        "answer_call_micro_usd": 100,
        "rewrite_call_micro_usd": 10,
    } | kwargs
    return SpendGuard(Budget(**values))


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
    spend = SpendGuard(PROFILE_B)
    admitted = sum(spend.admit(follow_up=True) for _ in range(200))

    assert admitted == 66
    assert spend.charged_micro_usd <= 3_000_000


# ---------------------------------------------------------------- charging


def test_a_first_question_is_one_answer_call() -> None:
    spend = guard()

    assert spend.admit(follow_up=False) is True
    assert (spend.questions, spend.calls, spend.charged_micro_usd) == (1, 1, 100)


def test_a_follow_up_adds_its_rewrite_call() -> None:
    spend = guard()

    assert spend.admit(follow_up=True) is True
    assert (spend.questions, spend.calls, spend.charged_micro_usd) == (1, 2, 110)


def test_charges_add_up_over_questions() -> None:
    spend = guard()

    spend.admit(follow_up=False)
    spend.admit(follow_up=True)
    spend.admit(follow_up=True)

    assert (spend.questions, spend.calls, spend.charged_micro_usd) == (3, 5, 320)


# ---------------------------------------------------------------- the stops


def test_a_charge_that_lands_exactly_on_the_cap_is_admitted() -> None:
    spend = guard(cap_micro_usd=200)

    assert spend.admit(follow_up=False) is True
    assert spend.admit(follow_up=False) is True
    assert spend.charged_micro_usd == 200
    assert spend.stopped is False


def test_a_charge_that_would_pass_the_cap_is_refused_and_not_charged() -> None:
    spend = guard(cap_micro_usd=209)

    assert spend.admit(follow_up=False) is True
    assert spend.admit(follow_up=True) is False
    assert (spend.questions, spend.calls, spend.charged_micro_usd) == (1, 1, 100)
    assert spend.stopped is True


def test_the_question_stop_refuses_the_next_question() -> None:
    spend = guard(max_questions=2)

    assert spend.admit(follow_up=False) is True
    assert spend.admit(follow_up=False) is True
    assert spend.admit(follow_up=False) is False
    assert spend.questions == 2
    assert spend.stopped is True


def test_once_stopped_nothing_more_is_admitted_even_if_it_would_fit() -> None:
    # A follow-up was refused at the cap; a cheaper first question must not slip in.
    spend = guard(cap_micro_usd=205)

    assert spend.admit(follow_up=False) is True
    assert spend.admit(follow_up=True) is False
    assert spend.admit(follow_up=False) is False
    assert spend.questions == 1
    assert spend.stopped is True


def test_once_stopped_the_guard_stays_stopped() -> None:
    spend = guard(max_questions=1)

    spend.admit(follow_up=False)
    for _ in range(5):
        assert spend.admit(follow_up=False) is False

    assert spend.stopped is True
    assert spend.questions == 1


def test_a_fresh_guard_has_not_stopped() -> None:
    spend = guard()

    assert spend.stopped is False
    assert spend.stopped is False


# ---------------------------------------------------------------- record and switch


def test_the_summary_records_counts_charge_cap_and_stop() -> None:
    spend = guard(max_questions=1)
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
    spend = guard()

    assert admission(enabled=True, guard=spend) == spend.admit
    assert admission(enabled=False, guard=spend) is admit_all


def test_without_the_budget_every_question_is_admitted() -> None:
    assert admit_all(follow_up=False) is True
    assert admit_all(follow_up=True) is True


# ---------------------------------------------------------------- ending the run


def test_the_shape_runs_on_until_the_guard_stops() -> None:
    spend = guard(max_questions=1)

    assert until_spent((16, 4.0), spend) == (16, 4.0)
    spend.admit(follow_up=False)
    assert until_spent((16, 4.0), spend) == (16, 4.0)


def test_the_shape_ends_the_run_once_the_guard_has_stopped() -> None:
    spend = guard(max_questions=1)
    spend.admit(follow_up=False)
    spend.admit(follow_up=False)

    assert until_spent((16, 4.0), spend) is None


def test_a_shape_that_has_ended_stays_ended() -> None:
    assert until_spent(None, guard()) is None


# ---------------------------------------------------------------- the glue (not measured)


def _glue() -> ast.Module:
    return ast.parse(LOCUSTFILE.read_text())


def _named(name: str) -> str:
    return ast.unparse(
        next(
            n
            for n in _glue().body
            if isinstance(n, ast.ClassDef | ast.FunctionDef) and n.name == name
        )
    )


def test_the_glue_ends_a_spent_run_through_the_shape_only() -> None:
    # runner.quit() from a user greenlet fired the stop event twice and lost meta.json.
    assert "until_spent(shape_users(tick(CONFIG.plan, now), burst=CONFIG.burst), GUARD)" in (
        _named("StepShape")
    )
    assert ".quit" not in LOCUSTFILE.read_text()


def test_every_user_asks_through_the_switched_guard() -> None:
    source = ast.unparse(_glue())

    assert "GUARD = SpendGuard(PROFILE_B)" in source
    assert "ADMIT = admission(enabled=CONFIG.budget, guard=GUARD)" in source
    for user in ("Browser", "Burst"):
        assert "admit=ADMIT" in _named(user), user


def test_the_run_files_record_the_switch_and_the_guard_summary() -> None:
    assert "'budget': CONFIG.budget" in _named("_meta")
    assert "'spend_guard': GUARD.summary()" in _named("_stop")

"""Rule 6 (no evidence within the noise floor) and rule 7 (when a default changes),
fixed before any run, with the boundary cases that must keep the default.
"""

from dataclasses import replace

import pytest

from app.evaluation.answers.decision import (
    Comparison,
    Evidence,
    Reason,
    Verdict,
    decide,
    evidence,
)
from app.evaluation.answers.stats import PairedDifference, Proportion

FLOOR = 0.05


def diff(mean: float, low: float, high: float, n: int = 48) -> PairedDifference:
    return PairedDifference(n=n, excluded=0, mean=mean, low=low, high=high)


# ------------------------------------------------------------------ rule 6


@pytest.mark.parametrize(
    ("difference", "expected", "floor"),
    [
        (diff(0.10, 0.02, 0.18), Evidence.BETTER, FLOOR),
        (diff(-0.10, -0.18, -0.02), Evidence.WORSE, FLOOR),
        (diff(0.10, -0.01, 0.20), Evidence.NONE, FLOOR),  # the interval holds zero
        (diff(-0.10, -0.20, 0.01), Evidence.NONE, FLOOR),
        (diff(0.05, 0.01, 0.09), Evidence.NONE, FLOOR),  # a gap equal to the floor
        (diff(-0.05, -0.09, -0.01), Evidence.NONE, FLOOR),
        (diff(0.10, 0.0, 0.20), Evidence.NONE, FLOOR),  # an interval touching zero
        (diff(0.8 - 0.7, 0.02, 0.18), Evidence.NONE, 0.1),  # equal to a 0.1 floor up to rounding
        (PairedDifference(0, 3, None, None, None), Evidence.NONE, FLOOR),  # nothing paired
    ],
)
def test_a_difference_counts_only_outside_zero_and_beyond_the_noise_floor(
    difference: PairedDifference, expected: Evidence, floor: float
) -> None:
    assert evidence(difference, floor) is expected


# ------------------------------------------------------------------ rule 7

GOOD = Comparison(
    primary=diff(0.10, 0.02, 0.18),
    primary_floor=FLOOR,
    false_refusals=(Proportion(4, 48), Proportion(4, 48)),
    false_refusal_floor=FLOOR,
    near_domain=(Proportion(8, 10), Proportion(8, 10)),
    near_domain_floor=0.1,
)


def test_a_clear_win_that_keeps_both_refusal_rates_changes_the_default() -> None:
    decision = decide(GOOD)

    assert decision.verdict is Verdict.CHANGE
    assert decision.reasons == ()


def test_a_primary_interval_that_holds_zero_keeps_the_default() -> None:
    decision = decide(replace(GOOD, primary=diff(0.10, -0.01, 0.20)))

    assert decision.verdict is Verdict.KEEP
    assert decision.reasons == (Reason.PRIMARY_INTERVAL_NOT_ABOVE_ZERO,)


def test_a_primary_gap_no_larger_than_the_noise_floor_keeps_the_default() -> None:
    decision = decide(replace(GOOD, primary=diff(0.05, 0.01, 0.09)))

    assert decision.verdict is Verdict.KEEP
    assert decision.reasons == (Reason.PRIMARY_GAP_WITHIN_NOISE,)


def test_a_primary_gap_equal_to_the_floor_up_to_rounding_keeps_the_default() -> None:
    decision = decide(replace(GOOD, primary=diff(0.8 - 0.7, 0.02, 0.18), primary_floor=0.1))

    assert decision.reasons == (Reason.PRIMARY_GAP_WITHIN_NOISE,)


def test_a_worse_alternative_keeps_the_default_for_both_primary_reasons() -> None:
    decision = decide(replace(GOOD, primary=diff(-0.10, -0.18, -0.02)))

    assert decision.verdict is Verdict.KEEP
    assert decision.reasons == (
        Reason.PRIMARY_INTERVAL_NOT_ABOVE_ZERO,
        Reason.PRIMARY_GAP_WITHIN_NOISE,
    )


def test_more_false_refusals_than_the_noise_floor_keep_the_default() -> None:
    # 4/48 to 7/48 is +0.0625, above the 0.05 floor.
    decision = decide(replace(GOOD, false_refusals=(Proportion(4, 48), Proportion(7, 48))))

    assert decision.verdict is Verdict.KEEP
    assert decision.reasons == (Reason.FALSE_REFUSALS_WORSE,)


def test_false_refusals_rising_by_at_most_the_floor_are_tolerated() -> None:
    # 4/40 to 6/40 is +0.05: equal to the floor, so not worse "by more than" it.
    comparison = replace(GOOD, false_refusals=(Proportion(4, 40), Proportion(6, 40)))

    assert decide(comparison).verdict is Verdict.CHANGE


def test_fewer_near_domain_refusals_than_the_floor_allows_keep_the_default() -> None:
    decision = decide(replace(GOOD, near_domain=(Proportion(8, 10), Proportion(6, 10))))

    assert decision.verdict is Verdict.KEEP
    assert decision.reasons == (Reason.NEAR_DOMAIN_REFUSALS_WORSE,)


def test_near_domain_refusals_falling_by_at_most_the_floor_are_tolerated() -> None:
    comparison = replace(GOOD, near_domain=(Proportion(8, 10), Proportion(7, 10)))

    assert decide(comparison).verdict is Verdict.CHANGE


def test_better_refusal_rates_never_block_a_change() -> None:
    comparison = replace(
        GOOD,
        false_refusals=(Proportion(8, 48), Proportion(0, 48)),
        near_domain=(Proportion(5, 10), Proportion(10, 10)),
    )

    assert decide(comparison).verdict is Verdict.CHANGE


def test_a_refusal_rate_that_cannot_be_computed_keeps_the_default() -> None:
    comparison = replace(GOOD, near_domain=(Proportion(8, 10), Proportion(0, 0)))

    decision = decide(comparison)

    assert decision.verdict is Verdict.KEEP
    assert decision.reasons == (Reason.REFUSAL_RATE_NOT_MEASURED,)


def test_no_paired_items_keep_the_default() -> None:
    decision = decide(replace(GOOD, primary=PairedDifference(0, 5, None, None, None)))

    assert decision.verdict is Verdict.KEEP
    assert decision.reasons == (Reason.NO_PAIRED_ITEMS,)


def test_every_failed_condition_is_reported() -> None:
    comparison = replace(
        GOOD,
        primary=diff(0.02, -0.03, 0.07),
        false_refusals=(Proportion(0, 48), Proportion(10, 48)),
        near_domain=(Proportion(10, 10), Proportion(2, 10)),
    )

    assert decide(comparison).reasons == (
        Reason.PRIMARY_INTERVAL_NOT_ABOVE_ZERO,
        Reason.PRIMARY_GAP_WITHIN_NOISE,
        Reason.FALSE_REFUSALS_WORSE,
        Reason.NEAR_DOMAIN_REFUSALS_WORSE,
    )


def test_dropping_the_reranker_is_deferred_to_the_joint_decision_even_on_a_win() -> None:
    decision = decide(replace(GOOD, drops_reranker=True))

    assert decision.verdict is Verdict.DEFER
    assert decision.reasons == (Reason.RERANKER_JOINT_DECISION,)


def test_a_reranker_drop_that_fails_the_rule_is_kept_not_deferred() -> None:
    decision = decide(replace(GOOD, primary=diff(0.0, -0.1, 0.1), drops_reranker=True))

    assert decision.verdict is Verdict.KEEP
    assert Reason.RERANKER_JOINT_DECISION not in decision.reasons

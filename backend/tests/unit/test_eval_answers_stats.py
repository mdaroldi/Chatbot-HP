"""Statistics of the answer benchmark (Phase 5 rules 5 and 6), with published or
hand-computed values: Wilson intervals, the paired bootstrap and the noise floor.
"""

import pytest

from app.evaluation.answers.stats import (
    PairedDifference,
    Proportion,
    noise_floor,
    paired_bootstrap,
    wilson,
)
from app.evaluation.metrics import bootstrap_interval

# ------------------------------------------------------------------ Wilson interval


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        # Published Wilson score intervals (z = 1.96), to four places.
        (0, 10, 0.0, 0.2775),
        (5, 10, 0.2366, 0.7634),
        (10, 10, 0.7225, 1.0),
        (1, 48, 0.0037, 0.1090),
        (24, 48, 0.3639, 0.6361),
    ],
)
def test_the_wilson_interval_matches_published_values(
    k: int, n: int, low: float, high: float
) -> None:
    interval = wilson(k, n)

    assert interval is not None
    assert interval[0] == pytest.approx(low, abs=1e-4)
    assert interval[1] == pytest.approx(high, abs=1e-4)


def test_the_wilson_interval_never_leaves_zero_to_one() -> None:
    for n in (1, 2, 7, 48):
        for k in range(n + 1):
            interval = wilson(k, n)
            assert interval is not None
            assert 0.0 <= interval[0] <= k / n <= interval[1] <= 1.0


def test_an_empty_group_has_no_interval() -> None:
    assert wilson(0, 0) is None


@pytest.mark.parametrize(("k", "n"), [(-1, 5), (6, 5), (0, -1)])
def test_impossible_counts_are_rejected(k: int, n: int) -> None:
    with pytest.raises(ValueError, match="count"):
        wilson(k, n)


def test_a_proportion_shows_its_counts_rate_and_interval() -> None:
    share = Proportion(5, 10)

    assert share.rate == 0.5
    assert share.interval == wilson(5, 10)


def test_an_empty_proportion_has_no_rate_and_no_interval() -> None:
    share = Proportion(0, 0)

    assert share.rate is None
    assert share.interval is None


def test_a_proportion_with_impossible_counts_is_rejected() -> None:
    with pytest.raises(ValueError, match="count"):
        Proportion(3, 2)


def test_a_proportion_counts_true_values_and_ignores_unscored_ones() -> None:
    assert Proportion.of([True, False, None, True]) == Proportion(2, 3)
    assert Proportion.of([None, None]) == Proportion(0, 0)


# ------------------------------------------------------------------ paired bootstrap


def test_the_difference_is_the_alternative_minus_the_reference() -> None:
    reference = {"q1": 1.0, "q2": 1.0, "q3": 0.0}
    alternative = {"q1": 0.0, "q2": 0.0, "q3": 0.0}

    difference = paired_bootstrap(reference, alternative, resamples=200, seed=3)

    assert difference.mean == pytest.approx(-2 / 3)
    assert difference.high is not None
    assert difference.high <= 0.0


def test_a_constant_gain_has_a_zero_width_interval_at_that_gain() -> None:
    reference = {"q1": 0.0, "q2": 0.0, "q3": 0.0}
    alternative = {"q1": 1.0, "q2": 1.0, "q3": 1.0}

    difference = paired_bootstrap(reference, alternative, resamples=100, seed=1)

    assert difference == PairedDifference(n=3, excluded=0, mean=1.0, low=1.0, high=1.0)


def test_only_items_valid_in_both_runs_are_paired_and_the_rest_are_counted() -> None:
    reference = {"q1": 1.0, "q2": 0.0, "only-ref": 1.0}
    alternative = {"q1": 1.0, "q2": 1.0, "only-alt-a": 0.0, "only-alt-b": 0.0}

    difference = paired_bootstrap(reference, alternative, resamples=50, seed=1)

    assert difference.n == 2
    assert difference.excluded == 3
    assert difference.mean == pytest.approx(0.5)


def test_the_bootstrap_resamples_per_item_differences_in_id_order() -> None:
    """The paired bootstrap is the percentile bootstrap of per-item differences,
    taken in sorted id order, so insertion order cannot change a report.
    """
    reference = {"b": 0.0, "a": 1.0, "d": 0.0, "c": 1.0, "e": 0.0}
    alternative = {"e": 1.0, "d": 1.0, "c": 1.0, "b": 0.0, "a": 0.0}
    diffs = [-1.0, 0.0, 0.0, 1.0, 1.0]  # a, b, c, d, e

    difference = paired_bootstrap(reference, alternative, resamples=500, seed=11)

    assert (difference.low, difference.high) == bootstrap_interval(diffs, resamples=500, seed=11)


def test_the_seed_fixes_the_interval() -> None:
    # Continuous differences: binary ones put resampled means on a coarse grid,
    # where two seeds can give the same interval.
    reference = {f"q{i}": (i * 0.37) % 1 for i in range(20)}
    alternative = {f"q{i}": (i * 0.61) % 1 for i in range(20)}

    first = paired_bootstrap(reference, alternative, resamples=300, seed=5)
    again = paired_bootstrap(dict(reversed(reference.items())), alternative, resamples=300, seed=5)
    other = paired_bootstrap(reference, alternative, resamples=300, seed=6)

    assert first == again
    assert first != other


def test_no_shared_items_gives_no_difference() -> None:
    difference = paired_bootstrap({"a": 1.0}, {"b": 1.0}, resamples=10, seed=1)

    assert difference == PairedDifference(n=0, excluded=2, mean=None, low=None, high=None)


# ------------------------------------------------------------------ noise floor (rule 6)


def test_the_noise_floor_is_the_spread_of_the_reference_runs() -> None:
    assert noise_floor([0.70, 0.75, 0.65], n=48) == pytest.approx(0.10)


def test_the_noise_floor_is_never_less_than_one_item() -> None:
    assert noise_floor([0.5, 0.5, 0.5], n=48) == pytest.approx(1 / 48)
    assert noise_floor([0.50, 0.51, 0.50], n=10) == pytest.approx(0.1)


def test_the_noise_floor_needs_runs_and_items() -> None:
    with pytest.raises(ValueError, match="run"):
        noise_floor([], n=10)
    with pytest.raises(ValueError, match="item"):
        noise_floor([0.5], n=0)

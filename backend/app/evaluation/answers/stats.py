"""Statistics of the answer benchmark (Phase 5 rules 5 and 6, fixed before any run).

Proportions carry a Wilson 95% interval, which stays inside [0, 1] and behaves at
0/n and n/n where the normal approximation does not. Differences between two runs are
paired by item: the percentile bootstrap of per-item differences, with a fixed seed so
a report can be reproduced. The noise floor is the spread of the three reference runs,
never less than one item, because a single answer flipping is always possible.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from app.evaluation.metrics import bootstrap_interval

_Z = 1.959963984540054  # two-sided 95%


def _check_counts(k: int, n: int) -> None:
    if n < 0 or not 0 <= k <= n:
        raise ValueError(f"impossible count: {k} of {n}")


def wilson(k: int, n: int) -> tuple[float, float] | None:
    """Wilson score 95% interval for k successes in n; None for an empty group."""
    _check_counts(k, n)
    if n == 0:
        return None
    p = k / n
    z2 = _Z * _Z
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = _Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    # Exact bounds at 0/n and n/n, where rounding would leave a stray 1e-17.
    low = 0.0 if k == 0 else centre - half
    high = 1.0 if k == n else centre + half
    return low, high


@dataclass(frozen=True)
class Proportion:
    """k of n, shown with its rate and interval so a reader always sees n."""

    k: int
    n: int

    def __post_init__(self) -> None:
        _check_counts(self.k, self.n)

    @property
    def rate(self) -> float | None:
        return self.k / self.n if self.n else None

    @property
    def interval(self) -> tuple[float, float] | None:
        return wilson(self.k, self.n)

    @classmethod
    def of(cls, verdicts: Iterable[bool | None]) -> "Proportion":
        """Counts True among the scored verdicts; None means not scored for this metric."""
        scored = [verdict for verdict in verdicts if verdict is not None]
        return cls(sum(scored), len(scored))


@dataclass(frozen=True)
class PairedDifference:
    """Alternative minus reference over the items valid in both runs.

    `excluded` counts items present in only one run (errors, or a different item set),
    so a reader sees how many pairs the interval rests on.
    """

    n: int
    excluded: int
    mean: float | None
    low: float | None
    high: float | None


def paired_bootstrap(
    reference: Mapping[str, float], alternative: Mapping[str, float], *, resamples: int, seed: int
) -> PairedDifference:
    shared = sorted(reference.keys() & alternative.keys())
    excluded = len(reference.keys() ^ alternative.keys())
    diffs = [alternative[item] - reference[item] for item in shared]
    interval = bootstrap_interval(diffs, resamples=resamples, seed=seed)
    if interval is None:
        return PairedDifference(0, excluded, None, None, None)
    return PairedDifference(len(diffs), excluded, sum(diffs) / len(diffs), *interval)


def noise_floor(values: Sequence[float], *, n: int) -> float:
    """Largest minus smallest value over the reference runs, at least 1/n (rule 6)."""
    if not values:
        raise ValueError("the noise floor needs at least one run")
    if n < 1:
        raise ValueError("the noise floor needs at least one item")
    return max(max(values) - min(values), 1 / n)

"""Rules 6 and 7 of the answer benchmark, fixed before any run.

Rule 6: a paired difference is evidence only when its 95% interval excludes zero and
its mean is larger than the noise floor; otherwise the report says "no evidence of a
difference". Boundaries keep the default: an interval touching zero or a gap equal to
the floor (up to rounding) is not evidence.

Rule 7: a default changes only when the alternative beats reference run 1 on the
primary metric under rule 6, and neither the false-refusal rate rises nor the
near-domain refusal rate falls by more than its own noise floor. Every failed condition
is reported, so a row says all of why it kept the default. A rate that cannot be
computed (an empty group) cannot show it did not worsen, so it keeps the default.
Dropping the reranker is a joint Phase 5 and Phase 6 decision: a reranker drop that
passes the rule is deferred, never applied here.
"""

from dataclasses import dataclass
from enum import StrEnum

from app.evaluation.answers.stats import PairedDifference, Proportion

# Rates and floors are both computed from k/n in floating point, so a gap equal to the
# floor can come out a few ulps above it (0.8 - 0.7 > 0.1); ties must keep the default.
_TOLERANCE = 1e-9


def _beyond(gap: float, floor: float) -> bool:
    return gap > floor + _TOLERANCE


class Evidence(StrEnum):
    BETTER = "better"
    WORSE = "worse"
    NONE = "no_evidence"


class Verdict(StrEnum):
    KEEP = "keep"
    CHANGE = "change"
    DEFER = "defer"  # passes rule 7, but the decision is joint with Phase 6


class Reason(StrEnum):
    NO_PAIRED_ITEMS = "no_paired_items"
    PRIMARY_INTERVAL_NOT_ABOVE_ZERO = "primary_interval_not_above_zero"
    PRIMARY_GAP_WITHIN_NOISE = "primary_gap_within_noise"
    FALSE_REFUSALS_WORSE = "false_refusals_worse"
    NEAR_DOMAIN_REFUSALS_WORSE = "near_domain_refusals_worse"
    REFUSAL_RATE_NOT_MEASURED = "refusal_rate_not_measured"
    RERANKER_JOINT_DECISION = "reranker_joint_decision"


@dataclass(frozen=True)
class Comparison:
    """An alternative against reference run 1; rate pairs are (reference, alternative)."""

    primary: PairedDifference  # alternative minus reference, paired by item
    primary_floor: float
    false_refusals: tuple[Proportion, Proportion]
    false_refusal_floor: float
    near_domain: tuple[Proportion, Proportion]  # near-domain refusal recall
    near_domain_floor: float
    drops_reranker: bool = False


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reasons: tuple[Reason, ...]


def evidence(difference: PairedDifference, floor: float) -> Evidence:
    mean, low, high = difference.mean, difference.low, difference.high
    if mean is None or low is None or high is None:
        return Evidence.NONE
    if low > 0 and _beyond(mean, floor):
        return Evidence.BETTER
    if high < 0 and _beyond(-mean, floor):
        return Evidence.WORSE
    return Evidence.NONE


def _primary_reasons(difference: PairedDifference, floor: float) -> list[Reason]:
    if difference.mean is None or difference.low is None:
        return [Reason.NO_PAIRED_ITEMS]
    reasons = []
    if difference.low <= 0:
        reasons.append(Reason.PRIMARY_INTERVAL_NOT_ABOVE_ZERO)
    if not _beyond(difference.mean, floor):
        reasons.append(Reason.PRIMARY_GAP_WITHIN_NOISE)
    return reasons


def _rise(rates: tuple[Proportion, Proportion]) -> float | None:
    reference, alternative = rates[0].rate, rates[1].rate
    if reference is None or alternative is None:
        return None
    return alternative - reference


def _guard_reasons(comparison: Comparison) -> list[Reason]:
    false_rise = _rise(comparison.false_refusals)
    near_rise = _rise(comparison.near_domain)
    if false_rise is None or near_rise is None:
        return [Reason.REFUSAL_RATE_NOT_MEASURED]
    reasons = []
    if _beyond(false_rise, comparison.false_refusal_floor):
        reasons.append(Reason.FALSE_REFUSALS_WORSE)
    if _beyond(-near_rise, comparison.near_domain_floor):
        reasons.append(Reason.NEAR_DOMAIN_REFUSALS_WORSE)
    return reasons


def decide(comparison: Comparison) -> Decision:
    reasons = _primary_reasons(comparison.primary, comparison.primary_floor)
    reasons += _guard_reasons(comparison)
    if reasons:
        return Decision(Verdict.KEEP, tuple(reasons))
    if comparison.drops_reranker:
        return Decision(Verdict.DEFER, (Reason.RERANKER_JOINT_DECISION,))
    return Decision(Verdict.CHANGE, ())

"""Run health: whether a run's results count as evidence (Phase 5 rule 3).

A failed item is not scored, so failures bias every metric toward the items that
happened to work; above `MAX_ERROR_RATE` the run is reported as invalid instead of
compared. The limit is an exact fraction, so exactly 5% is never lost to rounding.
Errors are counted by code, and setup failures also by the earlier turn's code, so a
report can say why without reading any answer.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from app.evaluation.answers.client import HarnessCode, ItemResult

MAX_ERROR_RATE = Fraction(1, 20)


@dataclass(frozen=True)
class RunHealth:
    items: int
    errors: int
    error_rate: float
    errors_by_code: Mapping[str, int]  # sorted by code
    setup_causes: Mapping[str, int]  # sorted by code
    failed_ids: tuple[str, ...]
    retries: int
    items_retried: int
    valid: bool


def _sorted(counts: Counter[str]) -> dict[str, int]:
    return dict(sorted(counts.items()))


def run_health(results: Sequence[ItemResult]) -> RunHealth:
    if not results:
        raise ValueError("no results: a run without items has no health")
    failed = [result for result in results if result.error is not None]
    causes = Counter(
        str(result.cause) for result in failed if result.error == HarnessCode.SETUP_TURN_FAILED
    )
    return RunHealth(
        items=len(results),
        errors=len(failed),
        error_rate=len(failed) / len(results),
        errors_by_code=_sorted(Counter(str(result.error) for result in failed)),
        setup_causes=_sorted(causes),
        failed_ids=tuple(result.item_id for result in failed),
        retries=sum(result.retries for result in results),
        items_retried=sum(result.retries > 0 for result in results),
        valid=Fraction(len(failed), len(results)) <= MAX_ERROR_RATE,
    )

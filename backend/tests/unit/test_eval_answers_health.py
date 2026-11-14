"""Run health: how many items failed and why, how many retries were spent, and whether
the run counts as evidence (at most 5% of items may fail).
"""

from fractions import Fraction

import pytest

from app.domain.chat import AnswerFlags
from app.evaluation.answers.client import Answered, HarnessCode, ItemResult
from app.evaluation.answers.health import MAX_ERROR_RATE, run_health
from app.evaluation.answers.sse import Done


def ok(item_id: str, retries: int = 0) -> ItemResult:
    done = Done(f"c-{item_id}", f"m-{item_id}", "Answer [S1].", None, ("S1",), AnswerFlags())
    return ItemResult(item_id, "client", f"c-{item_id}", retries, Answered(done, 0, 1.0, 2.0))


def error(item_id: str, code: str, cause: str | None = None, retries: int = 0) -> ItemResult:
    return ItemResult(item_id, "client", None, retries, None, code, cause, 502)


def run(oks: int, errors: int) -> list[ItemResult]:
    return [ok(f"ok-{n}") for n in range(oks)] + [
        error(f"err-{n}", "llm_error") for n in range(errors)
    ]


def test_a_clean_run_is_valid() -> None:
    health = run_health(run(10, 0))

    assert health.items == 10
    assert health.errors == 0
    assert health.error_rate == 0.0
    assert health.valid


@pytest.mark.parametrize(
    ("oks", "errors", "valid"),
    [
        (100, 5, True),  # 4.8% of the 105 dev items
        (99, 6, False),  # 5.7%
        (19, 1, True),  # exactly 5% is still allowed
        (18, 2, False),
        (0, 1, False),
    ],
)
def test_a_run_with_more_than_five_percent_errors_is_invalid(
    oks: int, errors: int, valid: bool
) -> None:
    health = run_health(run(oks, errors))

    assert health.errors == errors
    assert health.error_rate == pytest.approx(errors / (oks + errors))
    assert health.valid is valid


def test_the_threshold_is_five_percent() -> None:
    assert Fraction(1, 20) == MAX_ERROR_RATE


def test_errors_are_counted_by_code_and_setup_failures_by_cause() -> None:
    results = [
        ok("a"),
        error("b", "llm_error"),
        error("c", "llm_error"),
        error("d", HarnessCode.READ_TIMEOUT),
        error("e", HarnessCode.SETUP_TURN_FAILED, cause="llm_unavailable"),
        error("f", HarnessCode.SETUP_TURN_FAILED, cause="llm_unavailable"),
    ]

    health = run_health(results)

    assert health.errors_by_code == {"llm_error": 2, "read_timeout": 1, "setup_turn_failed": 2}
    assert list(health.errors_by_code) == ["llm_error", "read_timeout", "setup_turn_failed"]
    assert health.setup_causes == {"llm_unavailable": 2}
    assert health.failed_ids == ("b", "c", "d", "e", "f")


def test_retries_are_totalled_over_every_item_including_failed_ones() -> None:
    results = [ok("a", retries=2), ok("b"), error("c", "rate_limited", retries=2)]

    health = run_health(results)

    assert health.retries == 4
    assert health.items_retried == 2


def test_a_run_without_items_has_no_health() -> None:
    with pytest.raises(ValueError, match="no results"):
        run_health([])

"""Which conversations the retention period has expired."""

from datetime import UTC, datetime, timedelta

import pytest

from app.domain.retention import PurgeCounts, retention_cutoff

NOW = datetime(2026, 11, 14, 9, 0, tzinfo=UTC)


def test_the_cutoff_is_the_retention_period_before_now() -> None:
    assert retention_cutoff(NOW, days=30) == NOW - timedelta(days=30)
    assert retention_cutoff(NOW, days=1) == NOW - timedelta(days=1)


@pytest.mark.parametrize("days", [0, -1])
def test_a_period_shorter_than_a_day_is_rejected(days: int) -> None:
    """Settings already forbid it; a zero period would purge every conversation."""
    with pytest.raises(ValueError, match="at least one day"):
        retention_cutoff(NOW, days=days)


def test_the_cutoff_needs_an_aware_time() -> None:
    with pytest.raises(ValueError, match="timezone"):
        retention_cutoff(datetime(2026, 11, 14, 9, 0), days=30)


def test_counts_add_up_across_batches() -> None:
    assert PurgeCounts(2, 7) + PurgeCounts(1, 3) == PurgeCounts(3, 10)
    assert PurgeCounts.none() == PurgeCounts(0, 0)


@pytest.mark.parametrize(("conversations", "messages"), [(-1, 0), (0, -1)])
def test_counts_are_never_negative(conversations: int, messages: int) -> None:
    with pytest.raises(ValueError, match="negative"):
        PurgeCounts(conversations, messages)

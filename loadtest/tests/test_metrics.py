"""Metric extraction from the per-request records: windows, percentiles, the unit of
capacity (a message whose stream reached `done`) and error accounting."""

import json
import pathlib

import pytest

from hpdocs_load.metrics import (
    MESSAGE,
    RequestRecord,
    minute_buckets,
    percentile,
    read_records,
    summarize_window,
    write_record,
)


def rec(  # noqa: PLR0913 - a record has these fields; keywords keep tests readable
    end: float,
    *,
    name: str = MESSAGE,
    status: int = 200,
    duration_ms: float = 1000.0,
    ttft_ms: float | None = 300.0,
    reached_done: bool = True,
    error_code: str | None = None,
) -> RequestRecord:
    return RequestRecord(
        name=name,
        start=end - duration_ms / 1000,
        duration_ms=duration_ms,
        status=status,
        ttft_ms=ttft_ms if name == MESSAGE else None,
        reached_done=reached_done if name == MESSAGE else False,
        error_code=error_code,
    )


# ---------------------------------------------------------------- percentile


def test_percentile_uses_the_nearest_rank() -> None:
    values = [float(v) for v in range(1, 101)]

    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0
    assert percentile(values, 100) == 100.0


def test_percentile_ignores_input_order() -> None:
    assert percentile([5.0, 1.0, 3.0], 50) == 3.0


def test_percentile_of_one_value_is_that_value() -> None:
    assert percentile([7.0], 95) == 7.0


def test_percentile_of_nothing_is_none() -> None:
    assert percentile([], 95) is None


@pytest.mark.parametrize("q", [0, -1, 101])
def test_percentile_rejects_out_of_range_ranks(q: float) -> None:
    with pytest.raises(ValueError, match="percentile"):
        percentile([1.0], q)


# ---------------------------------------------------------------- window summary


def test_completed_questions_are_messages_that_reached_done() -> None:
    records = [
        rec(10.0),
        rec(20.0),
        rec(30.0, reached_done=False, error_code="llm_timeout"),
        rec(40.0, name="list"),
    ]

    summary = summarize_window(records, start=0.0, end=60.0)

    assert summary.questions_completed == 2
    assert summary.questions_per_minute == pytest.approx(2.0)
    assert summary.requests == 4
    assert summary.requests_per_minute == pytest.approx(4.0)


def test_records_count_in_the_window_where_they_end() -> None:
    inside = rec(59.5, duration_ms=5000.0)
    ends_after = rec(60.5, duration_ms=500.0)
    ends_at_start = rec(10.0, duration_ms=500.0)
    ends_at_end = rec(60.0, duration_ms=500.0)
    records = [inside, ends_after, ends_at_start, ends_at_end]

    summary = summarize_window(records, start=10.0, end=60.0)

    # Half-open (start, end]: inside and ends_at_end count; the other two do not.
    assert summary.requests == 2


def test_rates_are_per_minute_of_window() -> None:
    records = [rec(float(t)) for t in range(1, 181)]

    summary = summarize_window(records, start=0.0, end=180.0)

    assert summary.minutes == pytest.approx(3.0)
    assert summary.questions_per_minute == pytest.approx(60.0)


def test_errors_count_5xx_4xx_and_stream_failures_but_not_429() -> None:
    records = [
        rec(1.0),
        rec(2.0, status=503, ttft_ms=None, reached_done=False, error_code="llm_unavailable"),
        rec(3.0, name="list", status=500, error_code="internal_error"),
        rec(4.0, status=200, reached_done=False, error_code="no_done"),
        rec(5.0, status=429, ttft_ms=None, reached_done=False, error_code="rate_limited"),
        rec(6.0, name="list", status=409, error_code="conversation_busy"),
        rec(7.0, name="list", status=0, error_code="connection_error"),
    ]

    summary = summarize_window(records, start=0.0, end=60.0)

    assert summary.errors == 5
    assert summary.rate_limited == 1
    assert summary.error_rate == pytest.approx(5 / 7)
    assert summary.error_codes == {
        "conversation_busy": 1,
        "connection_error": 1,
        "internal_error": 1,
        "llm_unavailable": 1,
        "no_done": 1,
    }


def test_a_successful_request_with_no_error_code_is_not_an_error() -> None:
    summary = summarize_window([rec(1.0, name="delete", status=204)], start=0.0, end=60.0)

    assert summary.errors == 0
    assert summary.error_rate == 0.0


def test_latency_percentiles_use_messages_that_reached_done_only() -> None:
    records = [
        rec(1.0, ttft_ms=100.0, duration_ms=1000.0),
        rec(2.0, ttft_ms=200.0, duration_ms=2000.0),
        rec(3.0, ttft_ms=9000.0, duration_ms=9500.0, reached_done=False, error_code="llm_error"),
        rec(4.0, name="list", duration_ms=50.0),
    ]

    summary = summarize_window(records, start=0.0, end=60.0)

    assert summary.ttft_p50_ms == 100.0
    assert summary.ttft_p95_ms == 200.0
    assert summary.done_p50_ms == 1000.0
    assert summary.done_p95_ms == 2000.0


def test_an_empty_window_has_zero_rates_and_no_percentiles() -> None:
    summary = summarize_window([], start=0.0, end=60.0)

    assert summary.requests == 0
    assert summary.error_rate == 0.0
    assert summary.ttft_p95_ms is None
    assert summary.done_p95_ms is None


@pytest.mark.parametrize(("start", "end"), [(10.0, 10.0), (10.0, 5.0)])
def test_a_window_must_have_positive_length(start: float, end: float) -> None:
    with pytest.raises(ValueError, match="window"):
        summarize_window([], start=start, end=end)


# ---------------------------------------------------------------- minute buckets


def test_minute_buckets_split_the_window_into_whole_minutes() -> None:
    records = [rec(30.0), rec(90.0), rec(100.0), rec(170.0)]

    buckets = minute_buckets(records, start=0.0, end=180.0)

    assert [b.questions_completed for b in buckets] == [1, 2, 1]
    assert all(b.minutes == pytest.approx(1.0) for b in buckets)


def test_a_trailing_partial_minute_is_its_own_shorter_bucket() -> None:
    buckets = minute_buckets([rec(65.0)], start=0.0, end=90.0)

    assert [b.minutes for b in buckets] == pytest.approx([1.0, 0.5])
    assert buckets[1].questions_per_minute == pytest.approx(2.0)


# ---------------------------------------------------------------- JSONL round trip


def test_records_round_trip_through_jsonl(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "requests.jsonl"
    written = [rec(1.0), rec(2.0, name="list", status=429, error_code="rate_limited")]

    with path.open("w") as out:
        for record in written:
            write_record(out, record)

    assert read_records(path) == written


def test_a_record_line_holds_only_the_known_fields(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "requests.jsonl"
    with path.open("w") as out:
        write_record(out, rec(1.0))

    line = path.read_text()

    assert set(json.loads(line)) == {
        "name",
        "start",
        "duration_ms",
        "status",
        "ttft_ms",
        "reached_done",
        "error_code",
    }

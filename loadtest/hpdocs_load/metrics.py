"""Turning per-request records into the capacity table's numbers.

A record holds a route name, times, the status, an error code and, for a message, its
question class (first or follow-up), never message text.
A record counts in the window where it ends, so a step's throughput is what finished in
it. Latency percentiles use only messages whose stream reached `done`, the unit of
capacity; failures show up in the error rate instead.
"""

import json
import math
import pathlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import TextIO

MESSAGE = "message"
RATE_LIMITED = 429


@dataclass(frozen=True)
class RequestRecord:
    name: str
    start: float
    duration_ms: float
    status: int
    ttft_ms: float | None
    reached_done: bool
    error_code: str | None
    # A follow-up is rewritten into a standalone query before retrieval: its own SLO class.
    # Defaults to False so run files written before the field still read.
    follow_up: bool = False

    @property
    def end(self) -> float:
        return self.start + self.duration_ms / 1000

    @property
    def is_error(self) -> bool:
        """Any failure but 429, which is reported on its own (rule 3 of the brief)."""
        return self.status != RATE_LIMITED and self.error_code is not None


@dataclass(frozen=True)
class WindowSummary:
    minutes: float
    requests: int
    requests_per_minute: float
    questions_completed: int
    questions_per_minute: float
    errors: int
    rate_limited: int
    error_rate: float
    error_codes: dict[str, int]
    ttft_p50_ms: float | None
    ttft_p95_ms: float | None
    done_p50_ms: float | None
    done_p95_ms: float | None


def percentile(values: Iterable[float], q: float) -> float | None:
    """Nearest rank: always an observed value, which keeps small samples honest."""
    if not 0 < q <= 100:
        raise ValueError("percentile rank must be in (0, 100]")
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[math.ceil(q / 100 * len(ordered)) - 1]


def summarize_window(
    records: Iterable[RequestRecord], *, start: float, end: float
) -> WindowSummary:
    if end <= start:
        raise ValueError("a window needs a positive length")
    minutes = (end - start) / 60
    inside = [r for r in records if start < r.end <= end]
    done = [r for r in inside if r.name == MESSAGE and r.reached_done]
    errors = [r for r in inside if r.is_error]
    ttft = [r.ttft_ms for r in done if r.ttft_ms is not None]
    totals = [r.duration_ms for r in done]
    return WindowSummary(
        minutes=minutes,
        requests=len(inside),
        requests_per_minute=len(inside) / minutes,
        questions_completed=len(done),
        questions_per_minute=len(done) / minutes,
        errors=len(errors),
        rate_limited=sum(r.status == RATE_LIMITED for r in inside),
        error_rate=len(errors) / len(inside) if inside else 0.0,
        error_codes=dict(sorted(Counter(str(r.error_code) for r in errors).items())),
        ttft_p50_ms=percentile(ttft, 50),
        ttft_p95_ms=percentile(ttft, 95),
        done_p50_ms=percentile(totals, 50),
        done_p95_ms=percentile(totals, 95),
    )


def minute_buckets(
    records: Iterable[RequestRecord], *, start: float, end: float
) -> list[WindowSummary]:
    """Stability evidence for a step; pass or fail is judged on the whole window."""
    kept = list(records)
    edges = [*range(0, math.ceil(end - start), 60)]
    return [
        summarize_window(kept, start=start + edge, end=min(start + edge + 60, end))
        for edge in edges
    ]


def write_record(out: TextIO, record: RequestRecord) -> None:
    out.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def read_records(path: pathlib.Path) -> list[RequestRecord]:
    with path.open() as lines:
        return [RequestRecord(**json.loads(line)) for line in lines if line.strip()]

"""The capacity table of one run: `python -m hpdocs_load.report <run dir> [--slo SLO.md]`.

One row per step, judged on the whole measured window (per-minute values are stability
evidence, not the verdict). A step counts when its window is at least 3 minutes (rule 2),
the generator never reached Locust's own CPU warning level (rule 10) and the host did not
page out more than a few thousand pages a minute (the 8 GB host may start with residual
swap). A run whose swap grew by more than 500 MiB does not count at all. A gap of more
than 15 s between host samples (taken every 5 s) means the host slept or stalled: that
step does not count, and the report names the largest gap of the run.

First questions and follow-ups are two SLO classes: a follow-up is rewritten into a
standalone query before retrieval, which adds its own wait. Each class has its own p95
bounds; a step meets the SLO only when both classes do and the error rate over all
questions stays under the bound (429s apart). The SLO comes
from the first TOML block of `loadtest/SLO.md`; before it exists the table has no
verdict. The headline is the counting step with the most completed questions per minute
that met the SLO.
"""

import argparse
import itertools
import json
import pathlib
import re
import sys
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from hpdocs_load.metrics import MESSAGE, RequestRecord, WindowSummary, percentile, read_records
from hpdocs_load.metrics import summarize_window as summarize
from hpdocs_load.stages import StageRow, read_stages

GENERATOR_SATURATED_PERCENT = 90.0  # Locust warns at 90% CPU
MIN_WINDOW_SECONDS = 180.0
MAX_PAGEOUTS_PER_MINUTE = 3000.0  # "a few thousand", about 47 MiB/min at 16 KiB pages
MAX_RUN_SWAP_GROWTH_BYTES = 500 * 1024**2
MAX_SAMPLE_GAP_SECONDS = 15.0  # sampled every 5 s; one docker stats call takes about 2 s
DEFAULT_SLO = pathlib.Path(__file__).resolve().parents[1] / "SLO.md"
_CONTAINERS = {"backend": "backend", "postgres": "postgres", "qdrant": "qdrant"}
_TOML_BLOCK = re.compile(r"```toml\n(.*?)```", re.DOTALL)
_GIB = 1024**3
_MIB = 1024**2

Sample = Mapping[str, object]


@dataclass(frozen=True)
class Bound:
    ttft_p95_ms: float
    done_p95_ms: float

    def met_by(self, summary: WindowSummary) -> bool:
        """A class with no completed question in the window is no evidence of meeting it."""
        ttft, done = summary.ttft_p95_ms, summary.done_p95_ms
        return (
            ttft is not None
            and done is not None
            and ttft <= self.ttft_p95_ms
            and done <= self.done_p95_ms
        )


@dataclass(frozen=True)
class Slo:
    first: Bound
    follow_up: Bound
    max_error_rate: float


def _bound(values: Mapping[str, object]) -> Bound:
    return Bound(
        ttft_p95_ms=float(values["ttft_p95_ms"]),  # type: ignore[arg-type]  # TOML number
        done_p95_ms=float(values["done_p95_ms"]),  # type: ignore[arg-type]  # TOML number
    )


def parse_slo(text: str) -> Slo:
    match = _TOML_BLOCK.search(text)
    if match is None:
        raise ValueError("the SLO file has no TOML block")
    values = tomllib.loads(match.group(1))
    try:
        return Slo(
            first=_bound(values["first"]),
            follow_up=_bound(values["follow_up"]),
            max_error_rate=float(values["max_error_rate"]),
        )
    except KeyError as missing:
        raise ValueError(f"the SLO block lacks {missing}") from None


@dataclass(frozen=True)
class StepRow:
    users: int
    seconds: float
    summary: WindowSummary
    first: WindowSummary
    follow_up: WindowSummary
    generator_cpu_max: float | None
    cpu_mean: dict[str, float | None]
    cpu_max: dict[str, float | None]
    backend_mem_max_bytes: float | None
    host_mem_max_percent: float | None
    swap_max_bytes: float | None
    swap_growth_bytes: float | None
    pageouts_per_minute: float | None
    max_sample_gap_seconds: float | None
    stage_p50_p95: dict[str, tuple[float | None, float | None]]

    @property
    def long_enough(self) -> bool:
        return self.seconds >= MIN_WINDOW_SECONDS

    @property
    def generator_ok(self) -> bool:
        cpu = self.generator_cpu_max
        return cpu is None or cpu < GENERATOR_SATURATED_PERCENT

    @property
    def memory_ok(self) -> bool:
        rate = self.pageouts_per_minute
        return rate is None or rate <= MAX_PAGEOUTS_PER_MINUTE

    @property
    def sampling_ok(self) -> bool:
        gap = self.max_sample_gap_seconds
        return gap is None or gap <= MAX_SAMPLE_GAP_SECONDS

    @property
    def counts(self) -> bool:
        return (
            self.long_enough
            and self.generator_ok
            and self.generator_cpu_max is not None
            and self.memory_ok
            and self.sampling_ok
        )

    def slo_met(self, slo: Slo) -> bool:
        return (
            slo.first.met_by(self.first)
            and slo.follow_up.met_by(self.follow_up)
            and self.summary.error_rate < slo.max_error_rate
        )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _max(values: list[float]) -> float | None:
    return max(values) if values else None


def _numbers(samples: Iterable[Sample], key: str) -> list[float]:
    return [float(s[key]) for s in samples]  # type: ignore[arg-type]  # JSON numbers


def _p50_p95(values: list[float]) -> tuple[float | None, float | None]:
    return percentile(values, 50), percentile(values, 95)


def _ordered(samples: Iterable[Sample]) -> list[Sample]:
    return sorted(samples, key=lambda s: float(s["at"]))  # type: ignore[arg-type]


def _paging(host: Sequence[Sample]) -> tuple[float | None, float | None]:
    """Swap growth and page-outs per minute between the first and last host sample."""
    if len(host) < 2:
        return None, None
    first, last = host[0], host[-1]
    minutes = (float(last["at"]) - float(first["at"])) / 60  # type: ignore[arg-type]
    growth = float(last["swap_used_bytes"]) - float(first["swap_used_bytes"])  # type: ignore[arg-type]
    paged = float(last["pageouts"]) - float(first["pageouts"])  # type: ignore[arg-type]
    return growth, paged / minutes


def _max_gap(host: Sequence[Sample]) -> float | None:
    times = _numbers(host, "at")
    return _max([later - earlier for earlier, later in itertools.pairwise(times)])


def _stages(rows: list[StageRow]) -> dict[str, tuple[float | None, float | None]]:
    retrieval = {r.request_id: r.timings for r in rows if r.kind == "retrieval"}
    chat = [r.timings for r in rows if r.kind == "chat"]
    before_stream = [
        c["ttft_ms"] - retrieval[r.request_id]["total_ms"]
        for r in rows
        if r.kind == "chat"
        and "ttft_ms" in (c := r.timings)
        and "total_ms" in retrieval.get(r.request_id, {})
    ]

    def column(timings: Iterable[dict[str, float]], key: str) -> list[float]:
        return [t[key] for t in timings if key in t]

    return {
        "rerank_ms": _p50_p95(column(retrieval.values(), "rerank_ms")),
        "retrieval_ms": _p50_p95(column(retrieval.values(), "total_ms")),
        "chat_ttft_ms": _p50_p95(column(chat, "ttft_ms")),
        "ttft_minus_retrieval_ms": _p50_p95(before_stream),
    }


def _row(
    window: Mapping[str, float],
    records: list[RequestRecord],
    samples: Sequence[Sample],
    stages: list[StageRow],
) -> StepRow:
    start, end = window["start"], window["end"]
    inside = [s for s in samples if start < float(s["at"]) <= end]  # type: ignore[arg-type]
    host = _ordered(s for s in inside if s["kind"] == "host")
    swap_growth, pageouts_per_minute = _paging(host)
    containers = {
        short: [s for s in inside if s.get("name") == f"hpdocs-load-{name}-1"]
        for short, name in _CONTAINERS.items()
    }
    messages = [r for r in records if r.name == MESSAGE]
    return StepRow(
        users=int(window["users"]),
        seconds=end - start,
        summary=summarize(records, start=start, end=end),
        first=summarize([r for r in messages if not r.follow_up], start=start, end=end),
        follow_up=summarize([r for r in messages if r.follow_up], start=start, end=end),
        generator_cpu_max=_max(
            _numbers((s for s in inside if s["kind"] == "generator"), "cpu_percent")
        ),
        cpu_mean={k: _mean(_numbers(v, "cpu_percent")) for k, v in containers.items()},
        cpu_max={k: _max(_numbers(v, "cpu_percent")) for k, v in containers.items()},
        backend_mem_max_bytes=_max(_numbers(containers["backend"], "mem_bytes")),
        host_mem_max_percent=_max(_numbers(host, "mem_percent")),
        swap_max_bytes=_max(_numbers(host, "swap_used_bytes")),
        swap_growth_bytes=swap_growth,
        pageouts_per_minute=pageouts_per_minute,
        max_sample_gap_seconds=_max_gap(host),
        stage_p50_p95=_stages([r for r in stages if start < r.at <= end]),
    )


def build_rows(
    meta: Mapping[str, object],
    records: list[RequestRecord],
    samples: Sequence[Sample],
    stages: list[StageRow],
) -> list[StepRow]:
    windows: list[Mapping[str, float]] = meta["windows"]  # type: ignore[assignment]
    return [_row(w, records, samples, stages) for w in windows]


@dataclass(frozen=True)
class HostSpan:
    """The host from the first to the last sample of a run: the start and end record."""

    swap_start_bytes: float
    swap_end_bytes: float
    swap_growth_bytes: float  # peak minus start: growth that later receded still counts
    mem_percent_start: float
    mem_percent_end: float
    pageins: float
    pageouts: float
    max_gap_seconds: float | None

    @property
    def counts(self) -> bool:
        return self.swap_growth_bytes <= MAX_RUN_SWAP_GROWTH_BYTES


def host_span(samples: Iterable[Sample]) -> HostSpan | None:
    host = _ordered(s for s in samples if s["kind"] == "host")
    if not host:
        return None
    first, last = host[0], host[-1]
    swap = _numbers(host, "swap_used_bytes")

    def delta(key: str) -> float:
        return float(last[key]) - float(first[key])  # type: ignore[arg-type]

    return HostSpan(
        swap_start_bytes=swap[0],
        swap_end_bytes=swap[-1],
        swap_growth_bytes=max(swap) - swap[0],
        mem_percent_start=float(first["mem_percent"]),  # type: ignore[arg-type]
        mem_percent_end=float(last["mem_percent"]),  # type: ignore[arg-type]
        pageins=delta("pageins"),
        pageouts=delta("pageouts"),
        max_gap_seconds=_max_gap(host),
    )


def headline(rows: Sequence[StepRow], slo: Slo) -> StepRow | None:
    passing = [r for r in rows if r.counts and r.slo_met(slo)]
    return max(passing, key=lambda r: r.summary.questions_per_minute, default=None)


def _s(ms: float | None) -> str:
    return "-" if ms is None else f"{ms / 1000:.2f}"


def _pair(values: tuple[float | None, float | None]) -> str:
    return f"{_s(values[0])} / {_s(values[1])}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def _verdict(row: StepRow, slo: Slo | None) -> str:
    if slo is None:
        return "-"
    return "pass" if row.slo_met(slo) else "fail"


def _counts(row: StepRow) -> str:
    if not row.long_enough:
        return "no (short)"
    if not row.memory_ok:
        return "no (memory)"
    if not row.sampling_ok:
        return "no (stall)"
    return "yes" if row.counts else "no (generator)"


def _num(value: float | None, scale: float = 1.0) -> str:
    return "-" if value is None else f"{value / scale:.0f}"


_HEADER = (
    "| Users | Min | Q/min | Req/min | Err % | 429 | First TTFT p50 / p95 s "
    "| First done p50 / p95 s | Follow-up TTFT p50 / p95 s | Follow-up done p50 / p95 s "
    "| Rerank p50 / p95 s | TTFT - retrieval p50 / p95 s | Backend CPU mean / max % "
    "| PG CPU % | Qdrant CPU % | Locust CPU max % | Host mem max % | Swap max GiB "
    "| Swap growth MiB | Page-outs/min | Counts | SLO |"
)


def _line(row: StepRow, slo: Slo | None) -> str:
    s = row.summary
    swap = "-" if row.swap_max_bytes is None else f"{row.swap_max_bytes / _GIB:.1f}"
    cells = [
        str(row.users),
        f"{row.seconds / 60:.1f}",
        f"{s.questions_per_minute:.1f}",
        f"{s.requests_per_minute:.1f}",
        f"{s.error_rate * 100:.2f}",
        str(s.rate_limited),
        *(
            _pair(pair)
            for c in (row.first, row.follow_up)
            for pair in ((c.ttft_p50_ms, c.ttft_p95_ms), (c.done_p50_ms, c.done_p95_ms))
        ),
        _pair(row.stage_p50_p95["rerank_ms"]),
        _pair(row.stage_p50_p95["ttft_minus_retrieval_ms"]),
        f"{_pct(row.cpu_mean['backend'])} / {_pct(row.cpu_max['backend'])}",
        _pct(row.cpu_mean["postgres"]),
        _pct(row.cpu_mean["qdrant"]),
        _pct(row.generator_cpu_max),
        _pct(row.host_mem_max_percent),
        swap,
        _num(row.swap_growth_bytes, _MIB),
        _num(row.pageouts_per_minute),
        _counts(row),
        _verdict(row, slo),
    ]
    return "| " + " | ".join(cells) + " |"


def _host_line(span: HostSpan) -> str:
    return (
        f"Host: swap {span.swap_start_bytes / _MIB:.0f} -> {span.swap_end_bytes / _MIB:.0f} MiB "
        f"(peak growth {span.swap_growth_bytes / _MIB:.0f} MiB), memory used "
        f"{span.mem_percent_start:.0f}% -> {span.mem_percent_end:.0f}%, "
        f"page-ins +{span.pageins:.0f}, page-outs +{span.pageouts:.0f}, "
        f"largest sampler gap {_num(span.max_gap_seconds)} s."
    )


def render(rows: Sequence[StepRow], slo: Slo | None, span: HostSpan | None = None) -> str:
    separator = "|" + "---|" * _HEADER.count(" | ") + "---|"
    lines = [_HEADER, separator, *(_line(r, slo) for r in rows), ""]
    if span is not None:
        lines.append(_host_line(span))
    gap = None if span is None else span.max_gap_seconds
    if gap is not None and gap > MAX_SAMPLE_GAP_SECONDS:
        lines.append(
            f"Sleep or stall: a {gap:.0f} s gap between host samples "
            f"(limit {MAX_SAMPLE_GAP_SECONDS:.0f} s); check pmset -g log for this run."
        )
    if span is not None and not span.counts:
        lines.append(
            f"Run does not count: swap grew by {span.swap_growth_bytes / _MIB:.0f} MiB "
            f"(limit {MAX_RUN_SWAP_GROWTH_BYTES / _MIB:.0f} MiB)."
        )
    elif slo is None:
        lines.append("SLO: no SLO file yet, so no verdict.")
    else:
        best = headline(rows, slo)
        lines.append(
            "Sustained: no step met the SLO."
            if best is None
            else f"Sustained: {best.summary.questions_per_minute:.1f} completed questions/min "
            f"({best.summary.requests_per_minute:.1f} requests/min) at {best.users} users."
        )
    return "\n".join(lines) + "\n"


def _jsonl(path: pathlib.Path) -> list[Sample]:
    with path.open() as lines:
        return [json.loads(line) for line in lines if line.strip()]


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m hpdocs_load.report")
    parser.add_argument("run", type=pathlib.Path)
    parser.add_argument("--slo", type=pathlib.Path, default=DEFAULT_SLO)
    args = parser.parse_args(argv)
    run: pathlib.Path = args.run
    stages_file = run / "stages.jsonl"
    samples = _jsonl(run / "samples.jsonl")
    rows = build_rows(
        json.loads((run / "run.json").read_text()),
        read_records(run / "requests.jsonl"),
        samples,
        read_stages(stages_file) if stages_file.exists() else [],
    )
    slo = parse_slo(args.slo.read_text()) if args.slo.exists() else None
    text = render(rows, slo, host_span(samples))
    (run / "report.md").write_text(text)
    print(text, end="")  # noqa: T201 - the CLI's output is the table
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

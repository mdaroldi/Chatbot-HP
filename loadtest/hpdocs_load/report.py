"""The capacity table of one run: `python -m hpdocs_load.report <run dir> [--slo SLO.md]`.

One row per step, judged on the whole measured window (per-minute values are stability
evidence, not the verdict). A step counts when its window is at least 3 minutes (rule 2),
the generator never reached Locust's own CPU warning level (rule 10) and the host did not
page out more than a few thousand pages a minute (the 8 GB host may start with residual
swap). A run whose swap grew by more than 500 MiB does not count at all. The SLO comes
from the first TOML block of `loadtest/SLO.md`; before it exists the table has no
verdict. The headline is the counting step with the most completed questions per minute
that met the SLO.
"""

import argparse
import json
import pathlib
import re
import sys
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from hpdocs_load.metrics import RequestRecord, WindowSummary, percentile, read_records
from hpdocs_load.metrics import summarize_window as summarize
from hpdocs_load.stages import StageRow, read_stages

GENERATOR_SATURATED_PERCENT = 90.0  # Locust warns at 90% CPU
MIN_WINDOW_SECONDS = 180.0
MAX_PAGEOUTS_PER_MINUTE = 3000.0  # "a few thousand", about 47 MiB/min at 16 KiB pages
MAX_RUN_SWAP_GROWTH_BYTES = 500 * 1024**2
DEFAULT_SLO = pathlib.Path(__file__).resolve().parents[1] / "SLO.md"
_CONTAINERS = {"backend": "backend", "postgres": "postgres", "qdrant": "qdrant"}
_TOML_BLOCK = re.compile(r"```toml\n(.*?)```", re.DOTALL)
_GIB = 1024**3
_MIB = 1024**2

Sample = Mapping[str, object]


@dataclass(frozen=True)
class Slo:
    ttft_p95_ms: float
    done_p95_ms: float
    max_error_rate: float


def parse_slo(text: str) -> Slo:
    match = _TOML_BLOCK.search(text)
    if match is None:
        raise ValueError("the SLO file has no TOML block")
    values = tomllib.loads(match.group(1))
    try:
        return Slo(
            ttft_p95_ms=float(values["ttft_p95_ms"]),
            done_p95_ms=float(values["done_p95_ms"]),
            max_error_rate=float(values["max_error_rate"]),
        )
    except KeyError as missing:
        raise ValueError(f"the SLO block lacks {missing}") from None


@dataclass(frozen=True)
class StepRow:
    users: int
    seconds: float
    summary: WindowSummary
    generator_cpu_max: float | None
    cpu_mean: dict[str, float | None]
    cpu_max: dict[str, float | None]
    backend_mem_max_bytes: float | None
    host_mem_max_percent: float | None
    swap_max_bytes: float | None
    swap_growth_bytes: float | None
    pageouts_per_minute: float | None
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
    def counts(self) -> bool:
        return (
            self.long_enough
            and self.generator_ok
            and self.generator_cpu_max is not None
            and self.memory_ok
        )

    def slo_met(self, slo: Slo) -> bool:
        s = self.summary
        return (
            s.ttft_p95_ms is not None
            and s.done_p95_ms is not None
            and s.ttft_p95_ms <= slo.ttft_p95_ms
            and s.done_p95_ms <= slo.done_p95_ms
            and s.error_rate < slo.max_error_rate
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
    return StepRow(
        users=int(window["users"]),
        seconds=end - start,
        summary=summarize(records, start=start, end=end),
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
    return "yes" if row.counts else "no (generator)"


def _num(value: float | None, scale: float = 1.0) -> str:
    return "-" if value is None else f"{value / scale:.0f}"


_HEADER = (
    "| Users | Min | Q/min | Req/min | Err % | 429 | TTFT p50 / p95 s | Done p50 / p95 s "
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
        _pair((s.ttft_p50_ms, s.ttft_p95_ms)),
        _pair((s.done_p50_ms, s.done_p95_ms)),
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
        f"page-ins +{span.pageins:.0f}, page-outs +{span.pageouts:.0f}."
    )


def render(rows: Sequence[StepRow], slo: Slo | None, span: HostSpan | None = None) -> str:
    separator = "|" + "---|" * _HEADER.count(" | ") + "---|"
    lines = [_HEADER, separator, *(_line(r, slo) for r in rows), ""]
    if span is not None:
        lines.append(_host_line(span))
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

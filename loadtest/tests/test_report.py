"""The capacity table: one row per step, judged on the whole measured window, with the
generator and host columns that decide whether a step counts (rules 2 and 10)."""

import json
import pathlib
import runpy
import sys

import pytest

from hpdocs_load.metrics import RequestRecord, write_record
from hpdocs_load.report import (
    GENERATOR_SATURATED_PERCENT,
    MAX_PAGEOUTS_PER_MINUTE,
    MAX_RUN_SWAP_GROWTH_BYTES,
    MIN_WINDOW_SECONDS,
    Slo,
    build_rows,
    headline,
    host_span,
    main,
    parse_slo,
    render,
)
from hpdocs_load.stages import StageRow

SLO = Slo(ttft_p95_ms=4000.0, done_p95_ms=8000.0, max_error_rate=0.01)
SLO_TEXT = """# SLO

Prose first.

```toml
ttft_p95_ms = 4000
done_p95_ms = 8000
max_error_rate = 0.01
```
"""


def meta(*windows: tuple[int, float, float]) -> dict[str, object]:
    return {
        "run": "r",
        "windows": [{"users": u, "start": s, "end": e} for u, s, e in windows],
    }


def message(end: float, ttft: float = 1000.0, total: float = 3000.0, **kw: object) -> RequestRecord:
    fields: dict[str, object] = {
        "name": "message",
        "start": end - total / 1000,
        "duration_ms": total,
        "status": 200,
        "ttft_ms": ttft,
        "reached_done": True,
        "error_code": None,
    }
    return RequestRecord(**{**fields, **kw})  # type: ignore[arg-type]


def sample(at: float, kind: str, cpu: float = 0.0, **kw: object) -> dict[str, object]:
    return {"kind": kind, "at": at, "cpu_percent": cpu, **kw}


def host(at: float, swap: int = 0, pageouts: int = 0, **kw: object) -> dict[str, object]:
    fields = {"mem_percent": 50.0, "mem_available_bytes": 4 * 1024**3, "pageins": 0}
    return {
        "kind": "host",
        "at": at,
        "swap_used_bytes": swap,
        "pageouts": pageouts,
        **fields,
        **kw,
    }


WINDOW = (5, 100.0, 300.0)  # 200 s: long enough
_MIB = 1024**2


# ---------------------------------------------------------------- SLO


def test_the_slo_is_read_from_the_first_toml_block() -> None:
    assert parse_slo(SLO_TEXT) == SLO


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("# SLO\nno block", id="no-block"),
        pytest.param("```toml\nttft_p95_ms = 1\n```", id="missing-keys"),
    ],
)
def test_an_unreadable_slo_raises(text: str) -> None:
    with pytest.raises(ValueError, match="SLO"):
        parse_slo(text)


# ---------------------------------------------------------------- rows


def test_a_row_summarises_requests_resources_and_stages() -> None:
    records = [message(150.0, ttft=1000.0, total=3000.0), message(250.0, ttft=2000.0)]
    samples = [
        sample(150.0, "generator", 10.0),
        sample(250.0, "generator", 30.0),
        sample(150.0, "container", 300.0, name="hpdocs-load-backend-1", mem_bytes=1000),
        sample(250.0, "container", 400.0, name="hpdocs-load-backend-1", mem_bytes=3000),
        sample(250.0, "container", 20.0, name="hpdocs-load-postgres-1", mem_bytes=1),
        sample(250.0, "container", 40.0, name="hpdocs-load-qdrant-1", mem_bytes=1),
        {"kind": "host", "at": 200.0, "mem_percent": 80.0, "swap_used_bytes": 2 * 1024**3},
        sample(50.0, "generator", 99.0),  # before the window: ignored
    ]
    stages = [
        StageRow(150.0, "a", "retrieval", {"rerank_ms": 1500.0, "total_ms": 1600.0}),
        StageRow(150.0, "a", "chat", {"ttft_ms": 1900.0, "total_ms": 4900.0}),
        StageRow(250.0, "b", "retrieval", {"rerank_ms": 2500.0, "total_ms": 2600.0}),
    ]

    (row,) = build_rows(meta(WINDOW), records, samples, stages)

    assert row.users == 5
    assert row.summary.questions_completed == 2
    assert row.generator_cpu_max == 30.0
    assert row.cpu_mean["backend"] == 350.0
    assert row.cpu_mean["postgres"] == 20.0
    assert row.cpu_mean["qdrant"] == 40.0
    assert row.backend_mem_max_bytes == 3000
    assert row.host_mem_max_percent == 80.0
    assert row.swap_max_bytes == 2 * 1024**3
    assert row.stage_p50_p95["rerank_ms"] == (1500.0, 2500.0)
    assert row.stage_p50_p95["chat_ttft_ms"] == (1900.0, 1900.0)
    # Server TTFT minus retrieval, joined on the request ID: rewrite, DB and the first piece.
    assert row.stage_p50_p95["ttft_minus_retrieval_ms"] == (300.0, 300.0)
    assert row.counts


def test_a_short_window_does_not_count() -> None:
    (row,) = build_rows(meta((1, 0.0, MIN_WINDOW_SECONDS - 1)), [message(10.0)], [], [])

    assert not row.long_enough
    assert not row.counts


def test_a_saturated_generator_voids_the_step() -> None:
    samples = [sample(150.0, "generator", GENERATOR_SATURATED_PERCENT)]

    (row,) = build_rows(meta(WINDOW), [message(150.0)], samples, [])

    assert not row.generator_ok
    assert not row.counts


def test_a_window_without_samples_has_empty_resource_columns() -> None:
    (row,) = build_rows(meta(WINDOW), [message(150.0)], [], [])

    assert row.generator_cpu_max is None
    assert row.cpu_mean == {"backend": None, "postgres": None, "qdrant": None}
    assert row.stage_p50_p95["rerank_ms"] == (None, None)
    assert row.generator_ok  # nothing recorded is not evidence of saturation...
    assert not row.counts  # ...but a step without generator samples never counts


# ---------------------------------------------------------------- SLO verdict


def rows_for(*records: RequestRecord) -> list:  # type: ignore[type-arg]
    return build_rows(meta(WINDOW), list(records), [sample(200.0, "generator", 5.0)], [])


def test_slo_met_when_every_bound_holds() -> None:
    (row,) = rows_for(message(150.0, ttft=4000.0, total=8000.0))

    assert row.slo_met(SLO)


@pytest.mark.parametrize(
    "records",
    [
        pytest.param([message(150.0, ttft=4001.0)], id="ttft"),
        pytest.param([message(150.0, total=8001.0)], id="done"),
        pytest.param(
            [message(150.0)] * 99 + [message(160.0, reached_done=False, error_code="x")],
            id="one-percent-errors",
        ),
        pytest.param([], id="nothing-completed"),
    ],
)
def test_slo_missed(records: list[RequestRecord]) -> None:
    (row,) = rows_for(*records)

    assert not row.slo_met(SLO)


def test_429s_do_not_count_against_the_error_bound() -> None:
    (row,) = rows_for(
        message(150.0), message(160.0, status=429, reached_done=False, error_code="rate_limited")
    )

    assert row.slo_met(SLO)


# ---------------------------------------------------------------- headline


def test_the_headline_is_the_best_counting_step_that_meets_the_slo() -> None:
    windows = meta((1, 0.0, 200.0), (5, 200.0, 400.0), (10, 400.0, 600.0))
    records = [message(100.0), *[message(300.0 + i) for i in range(5)]]
    records += [message(500.0, ttft=9000.0)] * 9
    samples = [sample(t, "generator", 5.0) for t in (100.0, 300.0, 500.0)]

    rows = build_rows(windows, records, samples, [])

    best = headline(rows, SLO)
    assert best is not None
    assert best.users == 5


def test_no_headline_when_no_step_qualifies() -> None:
    assert headline(rows_for(message(150.0, ttft=9999.0)), SLO) is None


# ---------------------------------------------------------------- render and main


def test_render_shows_the_table_and_the_verdicts() -> None:
    rows = rows_for(message(150.0))

    text = render(rows, SLO)

    assert "| Users |" in text
    assert "| 5 |" in text
    assert "pass" in text
    assert "Sustained:" in text


def test_render_without_an_slo_says_so() -> None:
    text = render(rows_for(message(150.0)), None)

    assert "no SLO file" in text
    assert "Sustained:" not in text


def test_render_without_a_qualifying_step_says_so() -> None:
    text = render(rows_for(message(150.0, ttft=9999.0)), SLO)

    assert "no step met the SLO" in text


def write_run(directory: pathlib.Path) -> None:
    directory.mkdir()
    (directory / "run.json").write_text(json.dumps(meta(WINDOW)))
    with (directory / "requests.jsonl").open("w") as out:
        write_record(out, message(150.0))
    lines = [sample(200.0, "generator", 5.0), host(150.0), host(250.0, pageouts=5)]
    (directory / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in lines))
    stage = {"at": 150.0, "request_id": "a", "kind": "retrieval", "timings": {"rerank_ms": 1.0}}
    (directory / "stages.jsonl").write_text(json.dumps(stage) + "\n")


def test_main_prints_and_writes_the_report(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run"
    write_run(run)
    slo = tmp_path / "SLO.md"
    slo.write_text(SLO_TEXT)

    assert main([str(run), "--slo", str(slo)]) == 0

    printed = capsys.readouterr().out
    assert "| 5 |" in printed
    assert "page-outs +5" in printed
    assert (run / "report.md").read_text() == printed


def test_main_works_before_the_slo_exists(tmp_path: pathlib.Path) -> None:
    run = tmp_path / "run"
    write_run(run)
    (run / "stages.jsonl").unlink()

    assert main([str(run), "--slo", str(tmp_path / "missing.md")]) == 0
    assert "no SLO file" in (run / "report.md").read_text()


def test_render_marks_why_a_step_does_not_count() -> None:
    short = build_rows(meta((1, 0.0, 60.0)), [message(30.0)], [sample(30.0, "generator")], [])
    saturated = build_rows(meta(WINDOW), [message(150.0)], [sample(150.0, "generator", 95.0)], [])
    paging = [sample(150.0, "generator"), host(120.0), host(180.0, pageouts=10_000)]
    pressured = build_rows(meta(WINDOW), [message(150.0)], paging, [])

    assert "no (short)" in render(short, SLO)
    assert "no (generator)" in render(saturated, SLO)
    assert "no (memory)" in render(pressured, SLO)


# ---------------------------------------------------------------- host memory pressure


def test_page_outs_are_a_rate_over_the_window_and_the_limit_is_inclusive() -> None:
    limit = int(MAX_PAGEOUTS_PER_MINUTE)
    samples = [
        sample(150.0, "generator"),
        host(120.0, swap=100 * _MIB, pageouts=1000),
        host(240.0, swap=160 * _MIB, pageouts=1000 + 2 * limit),  # two minutes later
        host(400.0, pageouts=10**9),  # after the window: ignored
    ]

    (row,) = build_rows(meta(WINDOW), [message(150.0)], samples, [])

    assert row.pageouts_per_minute == MAX_PAGEOUTS_PER_MINUTE
    assert row.swap_growth_bytes == 60 * _MIB
    assert row.memory_ok
    assert row.counts


def test_heavy_paging_flags_the_step_as_memory_pressured() -> None:
    samples = [sample(150.0, "generator"), host(120.0), host(180.0, pageouts=3001)]

    (row,) = build_rows(meta(WINDOW), [message(150.0)], samples, [])

    assert row.pageouts_per_minute == 3001.0
    assert not row.memory_ok
    assert not row.counts


def test_fewer_than_two_host_samples_give_no_paging_rate() -> None:
    (row,) = build_rows(meta(WINDOW), [message(150.0)], [host(150.0, pageouts=10**6)], [])

    assert row.pageouts_per_minute is None
    assert row.swap_growth_bytes is None
    assert row.memory_ok  # the same sampler round records the generator, which gates counting


def test_the_host_span_runs_from_the_first_to_the_last_host_sample() -> None:
    samples = [
        sample(1.0, "generator"),
        host(10.0, swap=1352 * _MIB, pageouts=15_000, pageins=2_000_000, mem_percent=63.0),
        host(20.0, swap=1500 * _MIB, pageouts=15_200, pageins=2_000_500),
        host(30.0, swap=1400 * _MIB, pageouts=15_400, pageins=2_001_000, mem_percent=70.0),
    ]

    span = host_span(samples)

    assert span is not None
    assert (span.swap_start_bytes, span.swap_end_bytes) == (1352 * _MIB, 1400 * _MIB)
    assert span.swap_growth_bytes == 148 * _MIB  # the peak, not the end
    assert (span.mem_percent_start, span.mem_percent_end) == (63.0, 70.0)
    assert (span.pageins, span.pageouts) == (1000, 400)
    assert span.counts


def test_a_run_whose_swap_grew_too_much_does_not_count() -> None:
    grown = 1352 * _MIB + MAX_RUN_SWAP_GROWTH_BYTES + 1
    span = host_span([host(10.0, swap=1352 * _MIB), host(20.0, swap=grown)])

    assert span is not None
    assert not span.counts

    text = render(rows_for(message(150.0)), SLO, span)

    assert "Run does not count" in text
    assert "Sustained:" not in text


def test_no_host_samples_give_no_span() -> None:
    assert host_span([sample(1.0, "generator")]) is None


def test_render_states_the_host_at_start_and_end() -> None:
    span = host_span([host(10.0, swap=1352 * _MIB), host(20.0, swap=1360 * _MIB, pageouts=7)])

    text = render(rows_for(message(150.0)), SLO, span)

    assert "Host: swap 1352 -> 1360 MiB" in text
    assert "page-outs +7" in text
    assert "Sustained:" in text


def test_the_module_runs_as_a_script(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    write_run(run)
    monkeypatch.setattr(sys, "argv", ["report", str(run), "--slo", str(tmp_path / "none.md")])
    monkeypatch.delitem(sys.modules, "hpdocs_load.report")  # run it fresh, as `python -m` does

    with pytest.raises(SystemExit) as exit_:
        runpy.run_module("hpdocs_load.report", run_name="__main__")

    assert exit_.value.code == 0

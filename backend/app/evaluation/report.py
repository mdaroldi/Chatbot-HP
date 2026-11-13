"""Benchmark reports: `results.jsonl`, `summary.json` and `summary.md` for one run.

They hold identifiers, scores and timings only: never question wording, chunk text or
facts, so a report can be committed and shared without HP text (CLAUDE.md). The
header records what a rerun needs to reproduce the numbers: the golden set's version
and hash, the index build, the git SHA, the hardware and every resolved setting.
"""

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.domain.evaluation import Category, Metric, Split
from app.evaluation.matrix import NamedConfig
from app.evaluation.metrics import MetricSummary
from app.evaluation.runner import (
    Calibration,
    ConfigSummary,
    Distribution,
    IndexInfo,
    Latency,
    Preflight,
    QuestionRun,
    Selection,
)

Record = dict[str, object]

SCORED_OVER = "the candidate list (top_k = rerank_candidates)"
_SHORT_SHA = 12
_DIRTY = "-dirty"


@dataclass(frozen=True)
class RunHeader:
    created_at: datetime
    git_sha: str
    golden_version: str
    golden_sha256: str
    split: Split
    questions: int
    index: IndexInfo
    models: Mapping[str, str | None]  # model name -> pinned revision
    hardware: Mapping[str, object]
    seed: int
    resamples: int


@dataclass(frozen=True)
class RunOutcome:
    configs: tuple[NamedConfig, ...]
    summaries: tuple[ConfigSummary, ...]
    selection: Selection
    preflight: Preflight
    runs: tuple[QuestionRun, ...]


def run_dir_name(header: RunHeader) -> str:
    sha = header.git_sha.removesuffix(_DIRTY)[:_SHORT_SHA]
    dirty = _DIRTY if header.git_sha.endswith(_DIRTY) else ""
    return f"{header.created_at:%Y%m%dt%H%M%Sz}-{sha}{dirty}"


# ---------------------------------------------------------------- records


def result_record(run: QuestionRun) -> Record:
    result = run.result
    return {
        "config": run.config_name,
        "question_id": run.question.id,
        "category": run.question.category.value,
        "oos_kind": run.question.oos_kind.value if run.question.oos_kind else None,
        "chunk_ids": [chunk.hit.chunk_id for chunk in result.chunks],
        "store_scores": [chunk.hit.score for chunk in result.chunks],
        "rerank_scores": [chunk.rerank_score for chunk in result.chunks],
        "anchors_hit": [sorted(hits) for hits in run.profile],
        "scores": {m.value: v for m, v in run.scores.values.items()} if run.scores else None,
        "top_score": run.top_score,
        "filter": (
            {"doc_id": result.filter.doc_id, "source": result.filter.source.value}
            if result.filter
            else None
        ),
        "timings": dataclasses.asdict(result.timings),
    }


def _metric(summary: MetricSummary) -> Record:
    return {"n": summary.n, "mean": summary.mean, "low": summary.low, "high": summary.high}


def _latency(latency: Latency | None) -> Record | None:
    return {"p50": latency.p50, "p95": latency.p95} if latency is not None else None


def _calibration(calibration: Calibration | None) -> Record | None:
    if calibration is None:
        return None
    return {
        "threshold": calibration.threshold,
        "in_scope_n": calibration.in_scope_n,
        "allowed_misses": calibration.allowed_misses,
        "false_refusals": calibration.false_refusals,
        "in_scope_scores": dataclasses.asdict(calibration.in_scope_scores),
        "out_of_scope": {
            kind.value: {
                "n": catch.n,
                "caught": catch.caught,
                "share": catch.share,
                "scores": dataclasses.asdict(catch.scores),
            }
            for kind, catch in calibration.out_of_scope.items()
        },
    }


def _config(named: NamedConfig, summary: ConfigSummary) -> Record:
    return {
        "name": named.name,
        "reranker": named.reranker if named.config.rerank else None,
        "config": dataclasses.asdict(named.config),
        "metrics": {m.value: _metric(s) for m, s in summary.metrics.items()},
        "by_category": {
            category.value: {"n": group.n, **{m.value: v for m, v in group.means.items()}}
            for category, group in summary.categories.items()
        },
        "latency_ms": {stage: _latency(value) for stage, value in summary.latency.items()},
        "calibration": _calibration(summary.calibration),
    }


def summary_record(header: RunHeader, outcome: RunOutcome) -> Record:
    report = outcome.preflight
    return {
        "created_at": header.created_at.isoformat(),
        "git_sha": header.git_sha,
        "golden": {
            "version": header.golden_version,
            "sha256": header.golden_sha256,
            "split": header.split.value,
            "questions": header.questions,
        },
        "index": dataclasses.asdict(header.index),
        "models": dict(header.models),
        "hardware": dict(header.hardware),
        "bootstrap": {"seed": header.seed, "resamples": header.resamples},
        "scored_over": SCORED_OVER,
        "configs": [
            _config(named, summary)
            for named, summary in zip(outcome.configs, outcome.summaries, strict=True)
        ],
        "selection": {"best": outcome.selection.best, "tied": list(outcome.selection.tied)},
        "preflight": {
            "chunks": report.chunks,
            "anchors": len(report.resolutions),
            "problems": [dataclasses.asdict(p) for p in report.problems],
        },
    }


# ---------------------------------------------------------------- markdown


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _interval(summary: MetricSummary) -> str:
    return f"{_number(summary.mean)} [{_number(summary.low)}, {_number(summary.high)}]"


def _ms(latency: Latency | None) -> str:
    return "n/a" if latency is None else f"{latency.p50:.0f} / {latency.p95:.0f}"


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def _header_lines(header: RunHeader) -> list[str]:
    index = header.index
    models = ", ".join(f"`{name}` @ `{rev}`" for name, rev in header.models.items())
    hardware = ", ".join(f"{key} {value}" for key, value in header.hardware.items())
    return [
        "# Retrieval benchmark",
        "",
        f"- Golden set: v{header.golden_version}, SHA-256 `{header.golden_sha256}`, "
        f"split `{header.split.value}`, {header.questions} questions",
        f"- Index: build `{index.build_id}`, collection `{index.collection}`, "
        f"{index.chunk_count} chunks, chunking {index.chunking_version}",
        f"- Code: `{header.git_sha}`, run {header.created_at.isoformat()}",
        f"- Models: {models}",
        f"- Hardware: {hardware}",
        f"- Bootstrap: seed {header.seed}, {header.resamples} resamples (95% interval)",
        f"- Scored over {SCORED_OVER}",
    ]


def _calibration_lines(name: str, calibration: Calibration) -> list[str]:
    def spread(d: Distribution) -> str:
        return f"min {_number(d.low)}, median {_number(d.median)}, max {_number(d.high)}"

    lines = [
        f"### `{name}`",
        "",
        f"- Threshold {_number(calibration.threshold)}: {calibration.false_refusals} of "
        f"{calibration.in_scope_n} in-scope questions refused "
        f"(at most {calibration.allowed_misses} allowed); "
        f"scores {spread(calibration.in_scope_scores)}",
    ]
    for kind, catch in calibration.out_of_scope.items():
        lines.append(
            f"- Out of scope, {kind.value}: {catch.caught} of {catch.n} caught; "
            f"scores {spread(catch.scores)}"
        )
    return [*lines, ""]


def render_markdown(header: RunHeader, outcome: RunOutcome) -> str:
    summaries = outcome.summaries
    metric_rows = [
        [
            f"`{s.name}`",
            *(_interval(s.metrics[m]) for m in (Metric.RECALL_AT_5, Metric.MRR)),
            *(_number(s.metrics[m].mean) for m in (Metric.RECALL_AT_20, Metric.NDCG_AT_5)),
            _ms(s.latency["total"]),
        ]
        for s in summaries
    ]
    categories = [c for c in Category if any(c in s.categories for s in summaries)]
    category_rows = [
        [
            category.value,
            *(
                _number(s.categories[category].means[Metric.RECALL_AT_5])
                if category in s.categories
                else "n/a"
                for s in summaries
            ),
        ]
        for category in categories
    ]
    lines = [
        *_header_lines(header),
        "",
        "## Metrics",
        "",
        *_table(["config", "Recall@5", "MRR", "Recall@20", "nDCG@5", "p50 / p95 ms"], metric_rows),
        "",
        "## Recall@5 by category",
        "",
        *_table(["category", *(f"`{s.name}`" for s in summaries)], category_rows),
        "",
        "## Selection",
        "",
        f"Selected: `{outcome.selection.best}` (tied on Recall@5: "
        f"{', '.join(f'`{name}`' for name in outcome.selection.tied)})",
        "",
        "## Relevance threshold",
        "",
    ]
    for summary in summaries:
        if summary.calibration is not None:
            lines += _calibration_lines(summary.name, summary.calibration)
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------- files


def write_run(root: Path, header: RunHeader, outcome: RunOutcome) -> Path:
    """A new directory per run; an existing one is never overwritten."""
    path = root / run_dir_name(header)
    path.mkdir(parents=True, exist_ok=False)
    results = "".join(json.dumps(result_record(run)) + "\n" for run in outcome.runs)
    (path / "results.jsonl").write_text(results, encoding="utf-8")
    summary = json.dumps(summary_record(header, outcome), indent=2) + "\n"
    (path / "summary.json").write_text(summary, encoding="utf-8")
    (path / "summary.md").write_text(render_markdown(header, outcome), encoding="utf-8")
    return path

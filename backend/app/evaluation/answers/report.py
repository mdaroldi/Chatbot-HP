"""One answer run's report: `report.json` and `report.md` in the run directory.

The JSON record holds identifiers and numbers only (never question, answer, excerpt or
fact wording), so it can be shared; the cross-run summary reads it back. The Markdown
page is rendered from the record alone, so the two never disagree.

Sections: the header the caller resolves (run, config, models, build, dataset hashes,
git SHA), run health, the primary metric (rule 5) with k/n and its Wilson interval,
false refusals, out-of-scope recall, secondary metrics, one row per slice, refusals by
item and reason, long-query pairs, server latency (layer 1 refusals apart) and the
judge: first-pass verdicts, repeatability on the re-judged sample, and agreement with
the human labels when they exist. The JSON also keeps one row per item (id, slice
and verdicts, never text), which the summary pairs across runs for rule 7.
"""

import dataclasses
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.domain.answer_dataset import DatasetFile
from app.evaluation.answers import metrics
from app.evaluation.answers.answer_matrix import AnswerMatrix, RunSpec
from app.evaluation.answers.health import RunHealth
from app.evaluation.answers.judge import RUBRIC_SHA256, RUBRIC_VERSION, Correctness
from app.evaluation.answers.judge_run import JudgeRecord
from app.evaluation.answers.labels import Agreement, agreement
from app.evaluation.answers.scoring import ItemScore
from app.evaluation.answers.stats import Proportion

# Any: a JSON tree this module builds and the summary reads back; keys are fixed here.
Record = dict[str, Any]
SCHEMA = "answers-report-v1"
REPORT_JSON = "report.json"
REPORT_MD = "report.md"
ANSWERS_V1 = "answers_v1"  # the slice key of answers-v1 items (no slice)
_BOOLEAN_METRICS: tuple[tuple[str, Callable[[ItemScore], bool | None]], ...] = (
    ("warning_carried", lambda s: s.warning_carried),
    ("citations_valid", lambda s: s.citations_valid),
    ("uncited", lambda s: s.uncited),
    ("gold_cited", lambda s: s.gold_cited),
    ("forbidden_hit", lambda s: s.forbidden_hit),
    ("adversarial_pass", lambda s: s.adversarial_pass),
    ("rewrite_clean", lambda s: s.rewrite_clean),
)


class ReportError(Exception):
    """A run cannot be reported; names runs and counts only."""


# ---------------------------------------------------------------- record


def proportion(found: Proportion) -> Record:
    low, high = found.interval or (None, None)
    return {"k": found.k, "n": found.n, "rate": found.rate, "low": low, "high": high}


def _agreement(found: Agreement) -> Record:
    return {"n": found.n, "agreed": found.agreed, "kappa": found.kappa}


def _repeatability(records: Sequence[JudgeRecord]) -> Record | None:
    first = {r.item_id: r.verdict for r in records if r.judge_pass == 1 and r.verdict}
    again = {r.item_id: r.verdict for r in records if r.judge_pass == 2 and r.verdict}
    both = sorted(set(first) & set(again))
    if not both:
        return None
    pairs = [(first[i], again[i]) for i in both]
    return {
        "faithful": _agreement(agreement([(a.faithful, b.faithful) for a, b in pairs])),
        "correctness": _agreement(agreement([(a.correctness, b.correctness) for a, b in pairs])),
    }


def judge_section(
    records: Sequence[JudgeRecord], human: tuple[Agreement, Agreement] | None
) -> Record:
    """`human` is (faithful, correctness) agreement with the labels, None before them."""
    verdicts = [r.verdict for r in records if r.judge_pass == 1 and r.verdict is not None]
    correctness = Counter(v.correctness for v in verdicts)
    return {
        "faithful_lenient": proportion(Proportion.of(v.faithful for v in verdicts)),
        "faithful_strict": proportion(Proportion.of(v.fully_supported for v in verdicts)),
        "correctness": {c.value: correctness[c] for c in Correctness},
        "errors": dict(sorted(Counter(r.error for r in records if r.error).items())),
        "repeatability": _repeatability(records),
        "human": None
        if human is None
        else {"faithful": _agreement(human[0]), "correctness": _agreement(human[1])},
    }


def _health(health: RunHealth) -> Record:
    found = dataclasses.asdict(health)
    found["failed_ids"] = list(health.failed_ids)
    return found


def _slices(scores: Sequence[ItemScore]) -> Record:
    groups: dict[str, list[ItemScore]] = {}
    for found in scores:
        groups.setdefault(found.item.slice or ANSWERS_V1, []).append(found)
    return {
        name: {
            "items": len(group),
            "key_fact_pass": proportion(Proportion.of(s.key_fact_pass for s in group)),
            "refused": proportion(Proportion.of(s.refused for s in group)),
        }
        for name, group in groups.items()
    }


def _secondary(scores: Sequence[ItemScore]) -> Record:
    found: Record = {
        name: proportion(Proportion.of(pick(s) for s in scores)) for name, pick in _BOOLEAN_METRICS
    }
    support = [s.support for s in scores if s.support is not None]
    found["support_mean"] = sum(support) / len(support) if support else None
    return found


def _refusals(scores: Sequence[ItemScore]) -> Record:
    return {
        "by_reason": {r.value: n for r, n in metrics.refusal_reasons(scores).items()},
        "in_scope": [
            {"id": s.item.id, "reason": s.refusal_reason.value}
            for s in scores
            if s.item.in_scope and s.refusal_reason is not None
        ],
        "layer1": sorted(metrics.layer1_refused_ids(scores)),
    }


def _value(found: object) -> object:
    return found.value if isinstance(found, StrEnum) else found


def _items(scores: Sequence[ItemScore]) -> list[Record]:
    """One row per scored item, ids and verdicts only, so the summary can pair runs."""
    rows = []
    for found in scores:
        item = found.item
        row: Record = {
            "id": item.id,
            "slice": item.slice or ANSWERS_V1,
            "primary": item.primary,
            "in_scope": item.in_scope,
            "oos_kind": item.oos_kind,
            "attack": item.attack,
            "rewrite_trap": item.rewrite_trap,
            "refused": found.refused,
            "refusal_reason": found.refusal_reason,
        }
        row |= {name: getattr(found, name) for name in _ITEM_VERDICTS}
        rows.append({key: _value(value) for key, value in row.items()})
    return rows


_ITEM_VERDICTS = (
    "key_fact_pass",
    "coverage",
    "warning_carried",
    "forbidden_hit",
    "adversarial_pass",
    "rewrite_clean",
)


STUB_NOTE = "stub provider: pipeline check only, no quality claims"


def run_header(  # noqa: PLR0913 - one keyword per header line, named at the call site
    run: RunSpec,
    matrix: AnswerMatrix,
    *,
    provider: str,
    asked: int,
    total: int,
    build_ids: Iterable[str | None],
    datasets: Sequence[DatasetFile],
    git_sha: str,
) -> Record:
    """What the run was: its config, models, index builds, dataset versions and commit."""
    models = f"answer {matrix.models.answer}, rewrite {matrix.models.rewrite}"
    if run.judge:
        models += f", judge {matrix.models.judge} ({RUBRIC_VERSION} {RUBRIC_SHA256[:12]})"
    header: Record = {
        "run": run.name,
        "provider": provider,
        "items": f"{asked} of {total}",
        "env": ", ".join(f"{key}={value}" for key, value in run.env.items()) or "default",
        "models": models,
        "index_build_ids": ", ".join(sorted({b for b in build_ids if b})) or "none",
        "datasets": ", ".join(f"{f.name} {f.version} {f.sha256[:12]}" for f in datasets),
        "git_sha": git_sha,
    }
    if provider == "stub":
        header["note"] = STUB_NOTE
    return header


def run_record(
    header: Mapping[str, object],
    scores: Sequence[ItemScore],
    health: RunHealth,
    judge: Record | None,
) -> Record:
    if not scores:
        raise ReportError("a run with no scored items has no report")
    oos = metrics.oos_recall(scores)
    return {
        "schema": SCHEMA,
        "header": dict(header),
        "health": _health(health),
        "primary": proportion(metrics.primary(scores)),
        "false_refusals": proportion(metrics.false_refusals(scores)),
        "oos_recall": {
            "unrelated": proportion(oos.unrelated),
            "near_domain": proportion(oos.near_domain),
            "pooled": proportion(oos.pooled),
        },
        "secondary": _secondary(scores),
        "slices": _slices(scores),
        "refusals": _refusals(scores),
        "long_query_pairs": dataclasses.asdict(metrics.long_query_pairs(scores)),
        "latency_ms": dataclasses.asdict(metrics.latency(scores)),
        "judge": judge,
        "items": _items(scores),
    }


# ---------------------------------------------------------------- markdown


def number(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def share(found: Record) -> list[str]:
    interval = "n/a" if found["low"] is None else f"[{found['low']:.3f}, {found['high']:.3f}]"
    return [f"{found['k']}/{found['n']}", number(found["rate"]), interval]


def table(head: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _counts(head: str, counts: Mapping[str, int]) -> list[str]:
    return table([head, "items"], [[key, str(value)] for key, value in counts.items()])


_SHARE_HEAD = ("metric", "k/n", "rate", "Wilson 95%")


def _health_lines(health: Record) -> list[str]:
    verdict = "valid" if health["valid"] else "INVALID (error rate above 5%)"
    summary = (
        f"{verdict}: {health['errors']} errors in {health['items']} items, "
        f"{health['retries']} retries on {health['items_retried']} items."
    )
    return ["## Run health", "", summary, "", *_counts("error code", health["errors_by_code"])]


def _metric_lines(record: Record) -> list[str]:
    secondary = record["secondary"]
    rows = [
        ["primary (strict key-fact pass)", *share(record["primary"])],
        ["false refusals (in scope)", *share(record["false_refusals"])],
        *[[f"out-of-scope recall, {k}", *share(v)] for k, v in record["oos_recall"].items()],
        *[[name, *share(v)] for name, v in secondary.items() if name != "support_mean"],
    ]
    support = f"Mean citation support: {number(secondary['support_mean'])}"
    return ["## Metrics", "", *table(_SHARE_HEAD, rows), support, ""]


def _slice_lines(slices: Record) -> list[str]:
    rows = [
        [name, str(s["items"]), share(s["key_fact_pass"])[0], share(s["refused"])[0]]
        for name, s in slices.items()
    ]
    return ["## Slices", "", *table(["slice", "items", "key-fact pass", "refused"], rows)]


def _refusal_lines(refusals: Record) -> list[str]:
    in_scope = [[r["id"], r["reason"]] for r in refusals["in_scope"]]
    return [
        "## Refusals",
        "",
        *_counts("reason", refusals["by_reason"]),
        *table(["in-scope item", "reason"], in_scope),
    ]


def _agreement_lines(name: str, found: Record | None) -> list[str]:
    if found is None:
        return [f"{name}: not measured", ""]
    rows = [
        [label, str(a["n"]), str(a["agreed"]), number(a["agreed"] / a["n"]), number(a["kappa"])]
        for label, a in found.items()
    ]
    return [f"### {name}", "", *table(["label", "n", "agreed", "raw", "kappa"], rows)]


def _judge_lines(judge: Record | None) -> list[str]:
    if judge is None:
        return []
    return [
        "## Judge",
        "",
        *table(
            _SHARE_HEAD,
            [
                ["faithful, lenient (no unsupported claim)", *share(judge["faithful_lenient"])],
                ["faithful, strict (every claim supported)", *share(judge["faithful_strict"])],
            ],
        ),
        *_counts("correctness", judge["correctness"]),
        f"Judge errors: {judge['errors'] or 'none'}",
        "",
        *_agreement_lines("repeatability", judge["repeatability"]),
        *_agreement_lines("human", judge["human"]),
    ]


def render_markdown(record: Record) -> str:
    header = record["header"]
    latency = [[key, number(value)] for key, value in record["latency_ms"].items()]
    lines = [
        f"# Answer run {header['run']}",
        "",
        *[f"- {key}: {value}" for key, value in header.items() if key != "run"],
        "",
        *_health_lines(record["health"]),
        *_metric_lines(record),
        *_slice_lines(record["slices"]),
        *_refusal_lines(record["refusals"]),
        "## Latency (server, ms)",
        "",
        *table(["measure", "value"], latency),
        *_judge_lines(record["judge"]),
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_report(run_dir: Path, record: Record) -> None:
    (run_dir / REPORT_JSON).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (run_dir / REPORT_MD).write_text(render_markdown(record), encoding="utf-8")

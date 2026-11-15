"""The cross-run summary, `eval/reports/answers-summary.md` (committed): ids and numbers.

It reads each run's `report.json` and nothing else, so it never holds HP text, and it
applies the rules fixed before any run:
- rule 6: per metric, the noise floor is the range over the reference runs, at least
  one item (1/n); the floor is marked incomplete until all three references exist;
- rule 7, row by row against reference run 1: a paired bootstrap of the primary
  metric (seed and resamples from the matrix) and `decision.decide`. Every decision
  metric is computed on the answers-v1 items, the set every ablation row asks, so
  reference run 1 (the full dev set) and a row compare like with like;
- a reranker drop (rerank off in the row's env) that passes is deferred to Phase 6;
- the gate subset: the primary metric on items neither run refused at layer 1, so the
  gate's effect is told apart from ranking (no-rerank rows, the naive chunker);
- rewrite off is descriptive (multi-turn items against reference run 1, no verdict);
- the held-out run is shown apart, never compared (rule 8);
- a matrix row with no report is "not run", or "not run: budget" when the budget plan
  cut it.
The judge section repeats reference run 1's figures and the negative controls: how
many corrupted answers each view flagged, and which ones lenient faithfulness missed.
"""

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum

from app.evaluation.answers.answer_matrix import AnswerMatrix, RunSpec
from app.evaluation.answers.decision import Comparison, decide
from app.evaluation.answers.judge import Correctness
from app.evaluation.answers.judge_run import JudgeRecord
from app.evaluation.answers.report import ANSWERS_V1, Record, number, proportion, share, table
from app.evaluation.answers.stats import Proportion, noise_floor, paired_bootstrap

SCHEMA = "answers-summary-v1"
COMPARATOR = "reference-1"
REFERENCES = ("reference-1", "reference-2", "reference-3")
HELDOUT = "heldout"
REWRITE_OFF = "rewrite-off"
_NOT_COMPARED = {*REFERENCES, HELDOUT, REWRITE_OFF}
_RERANK = "RETRIEVAL_RERANK_ENABLED"
_LAYER1 = "below_threshold"
_SHARE_HEAD = ("metric", "k/n", "rate", "Wilson 95%")

Pick = Callable[[Record], bool]


class SummaryError(Exception):
    """The runs cannot be summarized; names runs only."""


class Metric(StrEnum):
    PRIMARY = "primary"
    FALSE_REFUSALS = "false_refusals"
    NEAR_DOMAIN = "near_domain"


def _answers_v1(row: Record) -> bool:
    return bool(row["slice"] == ANSWERS_V1)


# A metric is (which answers-v1 items, which verdict); refusals count as values.
_METRICS: dict[Metric, tuple[Pick, str]] = {
    Metric.PRIMARY: (lambda r: bool(r["primary"]), "key_fact_pass"),
    Metric.FALSE_REFUSALS: (lambda r: _answers_v1(r) and bool(r["in_scope"]), "refused"),
    Metric.NEAR_DOMAIN: (lambda r: _answers_v1(r) and r["oos_kind"] == "near_domain", "refused"),
}
_LABELS = {
    Metric.PRIMARY: "primary (strict key-fact pass)",
    Metric.FALSE_REFUSALS: "false refusals (in scope)",
    Metric.NEAR_DOMAIN: "near-domain refusal recall",
}


def values(report: Record, metric: Metric) -> dict[str, float]:
    """Item id to verdict, for the paired bootstrap."""
    pick, field = _METRICS[metric]
    return {
        row["id"]: float(row[field])
        for row in report["items"]
        if pick(row) and row[field] is not None
    }


def _rate(found: Mapping[str, float]) -> Proportion:
    return Proportion(int(sum(found.values())), len(found))


# ---------------------------------------------------------------- noise floor


def _noise(references: Mapping[str, Record]) -> Record:
    comparator = references[COMPARATOR]
    found: Record = {"complete": len(references) == len(REFERENCES)}
    for metric in Metric:
        rates = {name: _rate(values(report, metric)) for name, report in references.items()}
        n = len(values(comparator, metric))
        observed = [rate.rate for rate in rates.values() if rate.rate is not None]
        found[metric.value] = {
            "values": {name: proportion(rate) for name, rate in rates.items()},
            "n": n,
            "floor": noise_floor(observed, n=n) if observed and n else None,
        }
    return found


# ---------------------------------------------------------------- rule 7


def _layer1(report: Record) -> set[str]:
    return {
        row["id"]
        for row in report["items"]
        if _answers_v1(row) and row["refusal_reason"] == _LAYER1
    }


def _gate_subset(reference: Record, alternative: Record) -> Record:
    excluded = _layer1(reference) | _layer1(alternative)
    kept = [
        {i: v for i, v in values(report, Metric.PRIMARY).items() if i not in excluded}
        for report in (reference, alternative)
    ]
    return {
        "excluded": sorted(excluded),
        "reference": proportion(_rate(kept[0])),
        "alternative": proportion(_rate(kept[1])),
    }


def _floor(noise: Record, metric: Metric) -> float:
    found = noise[metric.value]["floor"]
    return 0.0 if found is None else float(found)


def _compared(
    matrix: AnswerMatrix, run: RunSpec, reference: Record, alternative: Record, noise: Record
) -> Record:
    rates = {
        metric: (_rate(values(reference, metric)), _rate(values(alternative, metric)))
        for metric in Metric
    }
    difference = paired_bootstrap(
        values(reference, Metric.PRIMARY),
        values(alternative, Metric.PRIMARY),
        resamples=matrix.resamples,
        seed=matrix.seed,
    )
    drops_reranker = run.env.get(_RERANK, "").lower() == "false"
    decision = decide(
        Comparison(
            primary=difference,
            primary_floor=_floor(noise, Metric.PRIMARY),
            false_refusals=rates[Metric.FALSE_REFUSALS],
            false_refusal_floor=_floor(noise, Metric.FALSE_REFUSALS),
            near_domain=rates[Metric.NEAR_DOMAIN],
            near_domain_floor=_floor(noise, Metric.NEAR_DOMAIN),
            drops_reranker=drops_reranker,
        )
    )
    return {
        "status": "run",
        "primary": proportion(rates[Metric.PRIMARY][1]),
        "difference": {
            "n": difference.n,
            "excluded": difference.excluded,
            "mean": difference.mean,
            "low": difference.low,
            "high": difference.high,
        },
        "gate_subset": _gate_subset(reference, alternative),
        "false_refusals": proportion(rates[Metric.FALSE_REFUSALS][1]),
        "near_domain": proportion(rates[Metric.NEAR_DOMAIN][1]),
        "layer1_refused": len(_layer1(alternative)),
        "drops_reranker": drops_reranker,
        "verdict": decision.verdict.value,
        "reasons": [reason.value for reason in decision.reasons],
    }


def _ablation(
    matrix: AnswerMatrix, reports: Mapping[str, Record], noise: Record, cut: frozenset[str]
) -> list[Record]:
    rows = []
    for run in matrix.runs:
        if run.name in _NOT_COMPARED:
            continue
        found: Record = {"run": run.name, "env": dict(run.env)}
        if run.name in reports:
            found |= _compared(matrix, run, reports[COMPARATOR], reports[run.name], noise)
        else:
            found["status"] = "not run: budget" if run.name in cut else "not run"
        rows.append(found)
    return rows


# ---------------------------------------------------------------- other sections


def _rewrite_off(reference: Record, described: Record | None) -> Record | None:
    if described is None:
        return None
    asked = {row["id"]: row for row in described["items"]}

    def passes(report: Record) -> Proportion:
        return Proportion.of(r["key_fact_pass"] for r in report["items"] if r["id"] in asked)

    return {
        "items": len(asked),
        "reference": proportion(passes(reference)),
        "alternative": proportion(passes(described)),
    }


def _heldout(report: Record | None) -> Record | None:
    if report is None:
        return None
    keys = ("primary", "false_refusals", "oos_recall", "slices", "judge")
    return {key: report[key] for key in keys}


def negative_controls(records: Sequence[JudgeRecord]) -> Record:
    """Corrupted answers, so every one should be flagged; counts what each view flagged."""
    judged = [(r.item_id, r.verdict) for r in records if r.judge_pass == 1 and r.verdict]
    return {
        "n": len(judged),
        "correctness_flagged": sum(v.correctness is not Correctness.CORRECT for _, v in judged),
        "lenient_flagged": sum(not v.faithful for _, v in judged),
        "strict_flagged": sum(not v.fully_supported for _, v in judged),
        "lenient_missed": sorted(
            item for item, v in judged if v.faithful and v.correctness is not Correctness.CORRECT
        ),
    }


_CAUSES: dict[str, Pick] = {
    "primary_refused_layer1": lambda r: r["primary"] and r["refusal_reason"] == _LAYER1,
    "primary_refused_model": lambda r: (
        r["primary"] and r["refused"] and r["refusal_reason"] != _LAYER1
    ),
    "primary_missing_facts": lambda r: (
        r["primary"] and not r["refused"] and r["key_fact_pass"] is False
    ),
    "slice_refused_layer1": lambda r: (
        not r["primary"] and r["in_scope"] and r["refusal_reason"] == _LAYER1
    ),
    "slice_refused_model": lambda r: (
        not r["primary"] and r["in_scope"] and r["refused"] and r["refusal_reason"] != _LAYER1
    ),
    "slice_missing_facts": lambda r: (
        not r["primary"] and not r["refused"] and r["key_fact_pass"] is False
    ),
    "out_of_scope_answered": lambda r: r["oos_kind"] is not None and not r["refused"],
    "adversarial_failed": lambda r: r["adversarial_pass"] is False,
    "forbidden_hit": lambda r: r["forbidden_hit"] is True,
    "warning_not_carried": lambda r: r["warning_carried"] is False,
    "rewrite_not_clean": lambda r: r["rewrite_clean"] is False,
}


def _failures(report: Record) -> dict[str, list[str]]:
    return {
        cause: [row["id"] for row in report["items"] if pick(row)]
        for cause, pick in _CAUSES.items()
    }


_REFERENCE_KEYS = (
    "header",
    "health",
    "primary",
    "false_refusals",
    "oos_recall",
    "secondary",
    "slices",
    "latency_ms",
)


def summary_record(
    matrix: AnswerMatrix,
    reports: Mapping[str, Record],
    *,
    cut: frozenset[str],
    negative: Record | None,
) -> Record:
    """`reports` maps run names to their report.json; `cut` holds the runs the budget
    plan cut; `negative` is `negative_controls` of the negative-control verdicts.
    """
    if COMPARATOR not in reports:
        raise SummaryError(f"no report for {COMPARATOR}, the comparator of every row")
    reference = reports[COMPARATOR]
    references = {name: reports[name] for name in REFERENCES if name in reports}
    noise = _noise(references)
    return {
        "schema": SCHEMA,
        "reference": {key: reference[key] for key in _REFERENCE_KEYS},
        "noise": noise,
        "ablation": _ablation(matrix, reports, noise, cut),
        "rewrite_off": _rewrite_off(reference, reports.get(REWRITE_OFF)),
        "heldout": _heldout(reports.get(HELDOUT)),
        "judge": reference["judge"],
        "negative_controls": negative,
        "failures": _failures(reference),
    }


# ---------------------------------------------------------------- markdown


def _noise_lines(noise: Record) -> list[str]:
    runs = [name for name in REFERENCES if name in noise[Metric.PRIMARY.value]["values"]]
    rows = []
    for metric in Metric:
        found = noise[metric.value]
        cells = [share(found["values"][name])[0] for name in runs]
        rows.append([_LABELS[metric], *cells, number(found["floor"]), str(found["n"])])
    note = "" if noise["complete"] else "Incomplete: fewer than three reference runs."
    return [
        "## Reference runs and noise floor (answers-v1 items)",
        "",
        *table(["metric", *runs, "noise floor", "n"], rows),
        note,
        "",
    ]


def _env(found: Mapping[str, str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in found.items()) or "default"


def _difference(found: Record) -> str:
    if found["mean"] is None:
        return "n/a"
    return f"{found['mean']:+.3f} [{found['low']:+.3f}, {found['high']:+.3f}] (n={found['n']})"


def _ablation_lines(rows: Sequence[Record]) -> list[str]:
    head = [
        "run", "env", "primary", "difference vs reference-1 [95%]", "gate subset ref -> alt",
        "false refusals", "near-domain recall", "layer 1 refused", "verdict", "reasons",
    ]  # fmt: skip
    lines = []
    for found in rows:
        if found["status"] != "run":
            lines.append([found["run"], _env(found["env"]), found["status"], *[""] * 7])
            continue
        gate = found["gate_subset"]
        lines.append([
            found["run"],
            _env(found["env"]),
            share(found["primary"])[0],
            _difference(found["difference"]),
            f"{share(gate['reference'])[0]} -> {share(gate['alternative'])[0]}",
            share(found["false_refusals"])[0],
            share(found["near_domain"])[0],
            str(found["layer1_refused"]),
            found["verdict"],
            ", ".join(found["reasons"]) or "-",
        ])  # fmt: skip
    return ["## Ablation (rule 7 against reference-1)", "", *table(head, lines)]


def _reference_lines(reference: Record) -> list[str]:
    oos = reference["oos_recall"]
    rows = [
        ["primary (strict key-fact pass)", *share(reference["primary"])],
        ["false refusals (in scope)", *share(reference["false_refusals"])],
        *[[f"out-of-scope recall, {kind}", *share(found)] for kind, found in oos.items()],
        *[[name, *share(found)] for name, found in reference["secondary"].items()
          if name != "support_mean"],
    ]  # fmt: skip
    slices = [
        [name, str(found["items"]), share(found["key_fact_pass"])[0], share(found["refused"])[0]]
        for name, found in reference["slices"].items()
    ]
    latency = [[key, number(value)] for key, value in reference["latency_ms"].items()]
    return [
        "## Reference run 1 (full dev set)",
        "",
        *[f"- {key}: {value}" for key, value in reference["header"].items()],
        f"- health: {'valid' if reference['health']['valid'] else 'INVALID'}, "
        f"{reference['health']['errors']} errors",
        "",
        *table(_SHARE_HEAD, rows),
        *table(["slice", "items", "key-fact pass", "refused"], slices),
        *table(["server latency", "ms"], latency),
    ]


def _rewrite_lines(found: Record | None) -> list[str]:
    if found is None:
        return ["## Rewrite off (descriptive)", "", "Not run.", ""]
    return [
        "## Rewrite off (descriptive, no verdict)",
        "",
        f"Key-fact pass on its {found['items']} multi-turn items: reference-1 "
        f"{share(found['reference'])[0]}, rewrite off {share(found['alternative'])[0]}.",
        "",
    ]


def _judge_rows(judge: Record) -> list[list[str]]:
    return [
        ["faithful, lenient (no unsupported claim)", *share(judge["faithful_lenient"])],
        ["faithful, strict (every claim supported)", *share(judge["faithful_strict"])],
    ]


def _heldout_lines(found: Record | None) -> list[str]:
    if found is None:
        return ["## Held-out (final default)", "", "Not run yet (rule 8: once, last).", ""]
    rows = [
        ["primary (strict key-fact pass)", *share(found["primary"])],
        ["false refusals (in scope)", *share(found["false_refusals"])],
        *[[f"out-of-scope recall, {k}", *share(v)] for k, v in found["oos_recall"].items()],
    ]
    if found["judge"] is not None:
        rows += _judge_rows(found["judge"])
    return ["## Held-out (final default)", "", *table(_SHARE_HEAD, rows)]


def _agreement_rows(name: str, found: Record | None) -> list[list[str]]:
    if found is None:
        return [[name, "not measured", "", "", ""]]
    return [
        [f"{name}, {label}", str(a["n"]), str(a["agreed"]), number(a["agreed"] / a["n"]),
         number(a["kappa"])]
        for label, a in found.items()
    ]  # fmt: skip


def _judge_lines(judge: Record | None, negative: Record | None) -> list[str]:
    lines = ["## Judge (reference-1)", ""]
    if judge is not None:
        correctness = ", ".join(f"{k} {v}" for k, v in judge["correctness"].items())
        agreement = _agreement_rows("repeatability", judge["repeatability"])
        agreement += _agreement_rows("human labels", judge["human"])
        lines += [
            *table(_SHARE_HEAD, _judge_rows(judge)),
            f"Correctness: {correctness}. Judge errors: {judge['errors'] or 'none'}.",
            "",
            *table(["agreement", "n", "agreed", "raw", "kappa"], agreement),
        ]
    if negative is not None:
        missed = ", ".join(negative["lenient_missed"]) or "none"
        lines += [
            f"Negative controls ({negative['n']} corrupted answers): correctness flagged "
            f"{negative['correctness_flagged']}, lenient faithfulness "
            f"{negative['lenient_flagged']}, strict faithfulness "
            f"{negative['strict_flagged']}. Missed by lenient: {missed}.",
            "",
        ]
    return lines


def _failure_lines(failures: Mapping[str, Sequence[str]]) -> list[str]:
    rows = [[cause, str(len(ids)), ", ".join(ids) or "-"] for cause, ids in failures.items()]
    return ["## Failures of reference-1 by cause", "", *table(["cause", "items", "ids"], rows)]


def render_summary(record: Record) -> str:
    lines = [
        "# Answer benchmark summary",
        "",
        "Generated from each run's report.json by `make eval-summary`; ids and numbers only.",
        "",
        *_noise_lines(record["noise"]),
        *_ablation_lines(record["ablation"]),
        *_rewrite_lines(record["rewrite_off"]),
        *_heldout_lines(record["heldout"]),
        *_judge_lines(record["judge"], record["negative_controls"]),
        *_failure_lines(record["failures"]),
        *_reference_lines(record["reference"]),
    ]
    return "\n".join(lines).rstrip() + "\n"

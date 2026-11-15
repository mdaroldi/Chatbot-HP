"""The cross-run summary: noise floors over the reference runs (rule 6), the ablation
table with rule 7 applied row by row against reference run 1, rows that did not run,
the descriptive rewrite-off row, the held-out row, the judge figures and the failures
of reference run 1 by cause. Fixtures are real run records (`report.run_record`).
"""

from collections.abc import Iterable

import pytest

from app.domain.answer_dataset import Slice
from app.domain.chat import RefusalReason
from app.domain.evaluation import OosKind
from app.evaluation.answers.answer_matrix import parse_answer_matrix
from app.evaluation.answers.health import RunHealth
from app.evaluation.answers.judge import Verdict
from app.evaluation.answers.judge_run import JudgeRecord
from app.evaluation.answers.report import Record, judge_section, run_record
from app.evaluation.answers.scoring import ItemScore
from app.evaluation.answers.summary import (
    SummaryError,
    negative_controls,
    render_summary,
    summary_record,
)
from tests.unit.test_eval_answers_matrix import BASE
from tests.unit.test_eval_answers_report import score

RUNS = """
[[runs]]
name = "reference-1"
items = "dev"
judge = true
judge_repeat = 2

[[runs]]
name = "reference-2"
items = "answers-dev"

[[runs]]
name = "reference-3"
items = "answers-dev"

[[runs]]
name = "heldout"
items = "heldout"
judge = true
reserved = true

[[runs]]
name = "hybrid-no-rerank"
items = "answers-dev"
env = { RETRIEVAL_RERANK_ENABLED = "false" }

[[runs]]
name = "naive"
items = "answers-dev"
env = { QDRANT_COLLECTION_ALIAS = "hpdocs_chunks_naive" }

[[runs]]
name = "rewrite-off"
items = "multi-turn-dev"
env = { CHAT_REWRITE_HISTORY_TURNS = "0" }

[[runs]]
name = "k3"
items = "answers-dev"
env = { RETRIEVAL_TOP_K = "3" }

[[runs]]
name = "k8"
items = "answers-dev"
env = { RETRIEVAL_TOP_K = "8" }
"""
MATRIX = parse_answer_matrix(BASE + RUNS)
HEALTH = RunHealth(
    items=1,
    errors=0,
    error_rate=0.0,
    errors_by_code={},
    setup_causes={},
    failed_ids=(),
    retries=0,
    items_retried=0,
    valid=True,
)
N = 20


def answers_v1(
    passes: Iterable[int], *, layer1: Iterable[int] = (), near_refused: bool = True
) -> list[ItemScore]:
    """N primary items p-0..p-19 (the given ones pass, `layer1` ones refused by the
    gate), two near-domain and one unrelated out-of-scope item.
    """
    passed, gate = set(passes), set(layer1)
    found = [
        score(
            f"p-{n}",
            passed=n in passed,
            refusal=RefusalReason.BELOW_THRESHOLD if n in gate else None,
        )
        for n in range(N)
    ]
    near = RefusalReason.MODEL_REFUSED if near_refused else None
    found += [
        score(f"nd-{n}", in_scope=False, oos_kind=OosKind.NEAR_DOMAIN, refusal=near)
        for n in range(2)
    ]
    found.append(score("un-0", in_scope=False, oos_kind=OosKind.UNRELATED, refusal=near))
    return found


def record(name: str, scores: list[ItemScore], judge: Record | None = None) -> Record:
    return run_record({"run": name}, scores, HEALTH, judge)


HALF = range(10)  # p-0..p-9 pass
REFERENCES = {
    name: record(name, answers_v1(HALF)) for name in ("reference-1", "reference-2", "reference-3")
}


def summary(**reports: Record) -> Record:
    found = REFERENCES | {name.replace("_", "-"): value for name, value in reports.items()}
    return summary_record(MATRIX, found, cut=frozenset(), negative=None)


def row(found: Record, name: str) -> Record:
    return next(r for r in found["ablation"] if r["run"] == name)


# ------------------------------------------------------------------ noise floor


def test_the_noise_floor_is_the_range_over_the_three_references() -> None:
    references = {
        "reference-1": record("reference-1", answers_v1(range(10))),
        "reference-2": record("reference-2", answers_v1(range(14))),
        "reference-3": record("reference-3", answers_v1(range(12))),
    }
    found = summary_record(MATRIX, references, cut=frozenset(), negative=None)

    primary = found["noise"]["primary"]
    assert [v["k"] for v in primary["values"].values()] == [10, 14, 12]
    assert primary["floor"] == pytest.approx(4 / N)
    assert primary["n"] == N
    assert found["noise"]["complete"] is True


def test_the_noise_floor_is_never_below_one_item() -> None:
    found = summary()
    assert found["noise"]["primary"]["floor"] == pytest.approx(1 / N)
    assert found["noise"]["near_domain"]["floor"] == pytest.approx(1 / 2)


def test_missing_reference_runs_mark_the_floor_incomplete() -> None:
    found = summary_record(
        MATRIX, {"reference-1": REFERENCES["reference-1"]}, cut=frozenset(), negative=None
    )
    assert found["noise"]["complete"] is False
    assert list(found["noise"]["primary"]["values"]) == ["reference-1"]


def test_the_summary_needs_reference_run_1() -> None:
    with pytest.raises(SummaryError, match="reference-1"):
        summary_record(MATRIX, {}, cut=frozenset(), negative=None)


# ------------------------------------------------------------------ rule 7


def test_a_row_beyond_the_noise_floor_changes_the_default() -> None:
    found = row(summary(k3=record("k3", answers_v1(range(N)))), "k3")

    assert found["status"] == "run"
    assert found["env"] == {"RETRIEVAL_TOP_K": "3"}
    assert (found["primary"]["k"], found["primary"]["n"]) == (N, N)
    difference = found["difference"]
    assert difference["n"] == N
    assert difference["mean"] == pytest.approx(0.5)
    assert difference["low"] > 0
    assert found["verdict"] == "change"
    assert found["reasons"] == []


def test_a_reranker_drop_that_passes_the_rule_is_deferred() -> None:
    found = row(summary(hybrid_no_rerank=record("x", answers_v1(range(N)))), "hybrid-no-rerank")
    assert found["drops_reranker"] is True
    assert found["verdict"] == "defer"
    assert found["reasons"] == ["reranker_joint_decision"]


def test_a_row_within_the_noise_keeps_the_default_and_says_why() -> None:
    found = row(summary(k3=record("k3", answers_v1(range(11)))), "k3")
    assert found["verdict"] == "keep"
    assert "primary_gap_within_noise" in found["reasons"]
    assert "primary_interval_not_above_zero" in found["reasons"]


def test_a_row_that_answers_near_domain_questions_keeps_the_default() -> None:
    better_but_unsafe = record("k3", answers_v1(range(N), near_refused=False))
    found = row(summary(k3=better_but_unsafe), "k3")
    assert found["verdict"] == "keep"
    assert found["reasons"] == ["near_domain_refusals_worse"]
    assert found["near_domain"]["k"] == 0


def test_the_gate_subset_leaves_out_items_either_run_refused_at_layer_1() -> None:
    reference = record("reference-1", answers_v1(HALF, layer1=(18, 19)))
    naive = record("naive", answers_v1(range(N), layer1=(0,)))
    found = summary_record(
        MATRIX,
        REFERENCES | {"reference-1": reference, "naive": naive},
        cut=frozenset(),
        negative=None,
    )

    gate = row(found, "naive")["gate_subset"]
    assert gate["excluded"] == ["p-0", "p-18", "p-19"]
    assert (gate["reference"]["k"], gate["reference"]["n"]) == (9, 17)
    assert (gate["alternative"]["k"], gate["alternative"]["n"]) == (17, 17)
    assert row(found, "naive")["layer1_refused"] == 1


def test_rows_that_did_not_run_say_whether_the_budget_cut_them() -> None:
    found = summary_record(MATRIX, REFERENCES, cut=frozenset({"k8"}), negative=None)

    assert [r["run"] for r in found["ablation"]] == ["hybrid-no-rerank", "naive", "k3", "k8"]
    assert row(found, "k8")["status"] == "not run: budget"
    assert row(found, "k3")["status"] == "not run"
    assert "verdict" not in row(found, "k3")


# ------------------------------------------------------------------ other rows


def multi_turn(passes: Iterable[int]) -> list[ItemScore]:
    passed = set(passes)
    return [score(f"mt-{n}", slice_=Slice.MULTI_TURN, passed=n in passed) for n in range(4)]


def test_rewrite_off_is_described_against_reference_1_on_the_same_items() -> None:
    reference = record("reference-1", answers_v1(HALF) + multi_turn(range(4)))
    rewrite_off = record("rewrite-off", multi_turn(range(1)))
    found = summary_record(
        MATRIX,
        REFERENCES | {"reference-1": reference, "rewrite-off": rewrite_off},
        cut=frozenset(),
        negative=None,
    )

    described = found["rewrite_off"]
    assert (described["reference"]["k"], described["reference"]["n"]) == (4, 4)
    assert (described["alternative"]["k"], described["alternative"]["n"]) == (1, 4)
    assert "verdict" not in described


def test_without_a_rewrite_off_run_the_section_is_empty() -> None:
    assert summary()["rewrite_off"] is None


def verdict_record(item_id: str, supports: tuple[str, ...], correctness: str) -> JudgeRecord:
    found = Verdict.model_validate(
        {
            "claims": [{"claim": "c", "support": s, "tags": []} for s in supports],
            "correctness": correctness,
            "reasons": ["none" if correctness == "correct" else "wrong_value"],
        }
    )
    return JudgeRecord(item_id, 1, "k", found, None, cached=False)


def test_the_heldout_row_is_shown_apart_with_its_judge_and_no_decision() -> None:
    judge = judge_section([verdict_record("p-0", ("supported",), "correct")], None)
    found = summary(heldout=record("heldout", answers_v1(range(15)), judge))

    heldout = found["heldout"]
    assert heldout["primary"]["k"] == 15
    assert heldout["judge"]["faithful_strict"]["k"] == 1
    assert "verdict" not in heldout
    assert "heldout" not in [r["run"] for r in found["ablation"]]
    assert summary()["heldout"] is None


def test_negative_controls_count_what_each_view_flagged() -> None:
    records = [
        verdict_record("nc-1", ("supported", "partial"), "incorrect"),  # lenient misses it
        verdict_record("nc-2", ("unsupported",), "incorrect"),
        verdict_record("nc-3", ("supported",), "correct"),  # nothing flags it
    ]
    found = negative_controls(records)

    assert found == {
        "n": 3,
        "correctness_flagged": 2,
        "lenient_flagged": 1,
        "strict_flagged": 2,
        "lenient_missed": ["nc-1"],
    }


def test_the_judge_section_and_negative_controls_reach_the_summary() -> None:
    judge = judge_section([verdict_record("p-0", ("supported", "partial"), "correct")], None)
    references = REFERENCES | {"reference-1": record("reference-1", answers_v1(HALF), judge)}
    negative = negative_controls([verdict_record("nc-1", ("partial",), "incorrect")])

    found = summary_record(MATRIX, references, cut=frozenset(), negative=negative)

    assert found["judge"]["faithful_lenient"]["k"] == 1
    assert found["judge"]["faithful_strict"]["k"] == 0
    assert found["negative_controls"]["lenient_missed"] == ["nc-1"]


def test_failures_of_reference_1_are_grouped_by_cause() -> None:
    scores = answers_v1(range(18), layer1=(18,))
    scores[19] = score("p-19", passed=False, refusal=RefusalReason.MODEL_REFUSED)
    scores += [
        score("lq-1", slice_=Slice.LONG_QUERY, refusal=RefusalReason.BELOW_THRESHOLD),
        score("adv-1", in_scope=False, slice_=Slice.ADVERSARIAL, adversarial=False, passed=None),
        score("w-1", warning=False),
    ]
    references = REFERENCES | {"reference-1": record("reference-1", scores)}

    causes = summary_record(MATRIX, references, cut=frozenset(), negative=None)["failures"]

    assert causes["primary_refused_layer1"] == ["p-18"]
    assert causes["primary_refused_model"] == ["p-19"]
    assert causes["slice_refused_layer1"] == ["lq-1"]
    assert causes["adversarial_failed"] == ["adv-1"]
    assert causes["warning_not_carried"] == ["w-1"]
    assert causes["primary_missing_facts"] == []


# ------------------------------------------------------------------ markdown


def test_the_markdown_shows_ids_and_numbers_only() -> None:
    found = summary_record(
        MATRIX,
        REFERENCES | {"k3": record("k3", answers_v1(range(N)))},
        cut=frozenset({"k8"}),
        negative=negative_controls([verdict_record("nc-1", ("partial",), "incorrect")]),
    )
    page = render_summary(found)

    assert page.startswith("# Answer benchmark summary\n")
    assert "| primary (strict key-fact pass) | 10/20 | 10/20 | 10/20 | 0.050 | 20 |" in page
    assert "| k3 | RETRIEVAL_TOP_K=3 | 20/20 |" in page
    assert "| change |" in page
    assert "| k8 | RETRIEVAL_TOP_K=8 | not run: budget |" in page
    assert "nc-1" in page
    assert "Q?" not in page  # never question wording


def test_the_markdown_renders_every_section_when_every_run_exists() -> None:
    judged = [
        verdict_record("p-0", ("supported",), "correct"),
        JudgeRecord(
            "p-0",
            2,
            "k",
            verdict_record("p-0", ("supported",), "correct").verdict,
            None,
            cached=False,
        ),
    ]
    judge = judge_section(judged, None)
    reference = record("reference-1", answers_v1(HALF) + multi_turn(range(4)), judge)
    found = summary_record(
        MATRIX,
        REFERENCES
        | {
            "reference-1": reference,
            "rewrite-off": record("rewrite-off", multi_turn(range(1))),
            "heldout": record("heldout", answers_v1(range(15)), judge),
        },
        cut=frozenset(),
        negative=None,
    )
    page = render_summary(found)

    assert "rewrite off 1/4" in page
    assert "## Held-out (final default)" in page
    assert "| faithful, strict (every claim supported) | 1/1 |" in page
    assert "| repeatability, faithful | 1 | 1 | 1.000 | n/a |" in page
    assert "| human labels | not measured |" in page
    assert "Negative controls" not in page


def test_the_markdown_says_what_has_not_run_yet() -> None:
    found = summary_record(
        MATRIX, {"reference-1": REFERENCES["reference-1"]}, cut=frozenset(), negative=None
    )
    page = render_summary(found)

    assert "Incomplete: fewer than three reference runs." in page
    assert "## Rewrite off (descriptive)\n\nNot run." in page
    assert "Not run yet (rule 8: once, last)." in page
    assert "## Judge (reference-1)\n\n## Failures" in page  # no judge pass yet


def test_a_row_with_no_paired_primary_item_has_no_difference() -> None:
    only_out_of_scope = record("k3", answers_v1(())[N:])
    found = summary(k3=only_out_of_scope)

    assert row(found, "k3")["reasons"][0] == "no_paired_items"
    assert "| k3 | RETRIEVAL_TOP_K=3 | 0/0 | n/a |" in render_summary(found)


def test_a_heldout_report_written_before_its_judge_pass_shows_no_judge_rows() -> None:
    page = render_summary(summary(heldout=record("heldout", answers_v1(range(15)))))
    held = page.split("## Held-out (final default)")[1].split("## Judge")[0]
    assert "| primary (strict key-fact pass) | 15/20 |" in held
    assert "faithful" not in held

"""One run's report: a JSON record of ids and numbers (the summary reads it back) and a
Markdown page rendered from that record alone.
"""

import json
from pathlib import Path

import pytest

from app.domain.answer_dataset import EvalItem, Slice
from app.domain.chat import RefusalReason
from app.domain.evaluation import OosKind, Split
from app.evaluation.answers.health import RunHealth
from app.evaluation.answers.judge import Verdict
from app.evaluation.answers.judge_run import JudgeRecord
from app.evaluation.answers.labels import Agreement
from app.evaluation.answers.report import (
    REPORT_JSON,
    REPORT_MD,
    ReportError,
    judge_section,
    proportion,
    render_markdown,
    run_record,
    write_report,
)
from app.evaluation.answers.scoring import ItemScore
from app.evaluation.answers.stats import Proportion
from tests.unit.test_eval_answers_labels import judged, record

HEALTH = RunHealth(
    items=6,
    errors=1,
    error_rate=1 / 6,
    errors_by_code={"read_timeout": 1},
    setup_causes={},
    failed_ids=("q-9",),
    retries=2,
    items_retried=1,
    valid=False,
)
HEADER = {"run": "reference-1", "git_sha": "abc123", "index_build_id": "b-1"}


def score(
    item_id: str,
    *,
    in_scope: bool = True,
    slice_: Slice | None = None,
    oos_kind: OosKind | None = None,
    refusal: RefusalReason | None = None,
    passed: bool | None = True,
    warning: bool | None = None,
    support: float | None = None,
    adversarial: bool | None = None,
    total_ms: float = 4000.0,
) -> ItemScore:
    item = EvalItem(
        id=item_id,
        split=Split.DEV,
        turns=("Q?",),
        in_scope=in_scope,
        slice=slice_,
        oos_kind=oos_kind,
    )
    refused = refusal is not None
    return ItemScore(
        item=item,
        refused=refused,
        refusal_reason=refusal,
        ttft_ms=None if refusal is RefusalReason.BELOW_THRESHOLD else 1000.0,
        total_ms=total_ms,
        key_fact_pass=(False if refused else passed) if in_scope else None,
        coverage=None,
        warning_carried=warning,
        citations_valid=None if refused else True,
        uncited=None if refused else False,
        support=support,
        gold_cited=None if refused else True,
        adversarial_pass=adversarial,
    )


SCORES = (
    score("q-1", warning=True, support=1.0),
    score("q-2", passed=False, support=0.5),
    score("q-3", refusal=RefusalReason.BELOW_THRESHOLD, total_ms=2000.0),
    score("oos-1", in_scope=False, oos_kind=OosKind.UNRELATED, refusal=RefusalReason.MODEL_REFUSED),
    score("lq-1", slice_=Slice.LONG_QUERY, refusal=RefusalReason.MODEL_REFUSED),
    score("adv-1", in_scope=False, slice_=Slice.ADVERSARIAL, adversarial=True, passed=None),
)


def test_a_proportion_is_recorded_with_its_rate_and_wilson_interval() -> None:
    found = proportion(Proportion(45, 48))
    assert (found["k"], found["n"]) == (45, 48)
    assert found["rate"] == pytest.approx(0.9375)
    assert (found["low"], found["high"]) == pytest.approx((0.8316, 0.9785), abs=1e-4)
    assert proportion(Proportion(0, 0)) == {"k": 0, "n": 0, "rate": None, "low": None, "high": None}


def test_the_record_holds_health_primary_and_refusals_by_item_and_reason() -> None:
    found = run_record(HEADER, SCORES, HEALTH, judge=None)

    assert found["schema"] == "answers-report-v1"
    assert found["header"] == HEADER
    assert found["health"]["valid"] is False
    assert found["health"]["failed_ids"] == ["q-9"]
    assert found["health"]["errors_by_code"] == {"read_timeout": 1}
    # primary: answers-v1 in-scope items only (q-1 pass, q-2 miss, q-3 refused)
    assert (found["primary"]["k"], found["primary"]["n"]) == (1, 3)
    assert (found["false_refusals"]["k"], found["false_refusals"]["n"]) == (2, 4)
    assert found["oos_recall"]["unrelated"]["k"] == 1
    assert found["refusals"]["in_scope"] == [
        {"id": "q-3", "reason": "below_threshold"},
        {"id": "lq-1", "reason": "model_refused"},
    ]
    assert found["refusals"]["by_reason"]["model_refused"] == 2
    assert found["refusals"]["layer1"] == ["q-3"]
    assert found["judge"] is None


def test_the_record_holds_secondary_metrics_slices_and_server_latency() -> None:
    found = run_record(HEADER, SCORES, HEALTH, judge=None)

    secondary = found["secondary"]
    assert (secondary["warning_carried"]["k"], secondary["warning_carried"]["n"]) == (1, 1)
    assert secondary["support_mean"] == pytest.approx(0.75)
    assert secondary["adversarial_pass"]["n"] == 1
    assert set(found["slices"]) == {"answers_v1", "long_query", "adversarial"}
    assert found["slices"]["answers_v1"]["items"] == 4
    assert found["slices"]["long_query"]["refused"]["k"] == 1
    assert found["latency_ms"]["layer1_n"] == 1
    assert found["latency_ms"]["n"] == 5
    assert found["long_query_pairs"]["n"] == 0


def test_the_record_is_plain_json() -> None:
    found = run_record(HEADER, SCORES, HEALTH, judge=judge_section([], human=None))
    assert json.loads(json.dumps(found)) == found
    assert found["judge"]["faithful_strict"] == proportion(Proportion(0, 0))
    assert found["latency_ms"]["total_p50"] == pytest.approx(4000.0)


def test_the_judge_section_counts_first_passes_repeatability_and_human_agreement() -> None:
    records = [
        record(judged("q-1"), faithful=True, correctness="correct"),
        record(judged("q-2"), faithful=False, correctness="partial"),
        record(judged("q-1"), judge_pass=2, faithful=True, correctness="correct"),
        record(judged("q-2"), judge_pass=2, faithful=True, correctness="partial"),
        JudgeRecord("q-3", 1, "k", None, "refusal", cached=False),
    ]
    human = (Agreement(20, 19, 0.5), Agreement(20, 18, None))

    found = judge_section(records, human=human)

    assert (found["faithful_lenient"]["k"], found["faithful_lenient"]["n"]) == (1, 2)
    assert found["correctness"] == {"correct": 1, "partial": 1, "incorrect": 0}
    assert found["errors"] == {"refusal": 1}
    assert found["repeatability"]["faithful"] == {"n": 2, "agreed": 1, "kappa": 0.0}
    assert found["repeatability"]["correctness"]["agreed"] == 2
    assert found["human"]["faithful"] == {"n": 20, "agreed": 19, "kappa": 0.5}
    assert found["human"]["correctness"]["kappa"] is None


def claims_record(item_id: str, *supports: str) -> JudgeRecord:
    found = Verdict.model_validate(
        {
            "claims": [{"claim": "c", "support": s, "tags": []} for s in supports],
            "correctness": "correct",
            "reasons": ["none"],
        }
    )
    return JudgeRecord(item_id, 1, "k", found, None, cached=False)


def test_faithfulness_is_reported_lenient_and_strict_from_the_claims() -> None:
    # A wrong value graded "partial" passes the lenient view (no unsupported claim) and
    # fails the strict one (every claim supported); both come from the stored claims.
    records = [
        claims_record("q-1", "supported", "supported"),
        claims_record("q-2", "supported", "partial"),
        claims_record("q-3", "unsupported"),
    ]

    found = judge_section(records, human=None)

    assert (found["faithful_lenient"]["k"], found["faithful_lenient"]["n"]) == (2, 3)
    assert (found["faithful_strict"]["k"], found["faithful_strict"]["n"]) == (1, 3)
    assert "faithful" not in found  # no unqualified figure to misread


def test_without_labels_the_judge_section_has_no_human_agreement() -> None:
    assert judge_section([record(judged("q-1"))], human=None)["human"] is None


def test_the_markdown_shows_ids_and_numbers_from_the_record() -> None:
    found = run_record(HEADER, SCORES, HEALTH, judge=judge_section([record(judged("q-1"))], None))
    page = render_markdown(found)

    assert page.startswith("# Answer run reference-1\n")
    assert "INVALID" in page  # the health verdict is visible at once
    assert "| primary (strict key-fact pass) | 1/3 | 0.333 | [0.061, 0.792] |" in page
    assert "| q-3 | below_threshold |" in page
    assert "| faithful, lenient (no unsupported claim) | 1/1 |" in page
    assert "| faithful, strict (every claim supported) | 1/1 |" in page
    assert "Q?" not in page  # never question wording


def test_the_markdown_shows_measured_agreement_as_rows() -> None:
    records = [
        record(judged("q-1"), faithful=True, correctness="correct"),
        record(judged("q-1"), judge_pass=2, faithful=True, correctness="partial"),
    ]
    human = (Agreement(20, 19, 0.5), Agreement(20, 18, None))
    found = run_record(HEADER, SCORES, HEALTH, judge=judge_section(records, human))

    page = render_markdown(found)

    assert "### repeatability" in page
    assert "| faithful | 1 | 1 | 1.000 | n/a |" in page  # kappa undefined: p_e = 1
    assert "| correctness | 1 | 0 | 0.000 | 0.000 |" in page
    assert "### human" in page
    assert "| faithful | 20 | 19 | 0.950 | 0.500 |" in page
    assert "not measured" not in page


def test_the_report_is_written_as_json_and_markdown(tmp_path: Path) -> None:
    found = run_record(HEADER, SCORES, HEALTH, judge=None)
    write_report(tmp_path, found)
    assert json.loads((tmp_path / REPORT_JSON).read_text(encoding="utf-8")) == found
    assert (tmp_path / REPORT_MD).read_text(encoding="utf-8") == render_markdown(found)


def test_a_run_with_no_scored_items_has_no_report() -> None:
    with pytest.raises(ReportError, match="no scored"):
        run_record(HEADER, (), HEALTH, judge=None)

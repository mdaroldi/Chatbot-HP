"""Run-level aggregates of the answer benchmark (Phase 5 section 6), built from item
scores with hand-computed values.
"""

from dataclasses import asdict

import pytest

from app.domain.answer_dataset import EvalItem, Slice
from app.domain.chat import RefusalReason
from app.domain.evaluation import OosKind, Split
from app.evaluation.answers.metrics import (
    LatencySummary,
    LongQueryPairs,
    false_refusals,
    latency,
    layer1_refused_ids,
    long_query_pairs,
    oos_recall,
    per_item,
    primary,
    refusal_reasons,
)
from app.evaluation.answers.scoring import ItemScore
from app.evaluation.answers.stats import Proportion

BT, MR, PR = (
    RefusalReason.BELOW_THRESHOLD,
    RefusalReason.MODEL_REFUSED,
    RefusalReason.PROVIDER_REFUSED,
)


def item(
    qid: str,
    *,
    in_scope: bool = True,
    slice_: Slice | None = None,
    oos_kind: OosKind | None = None,
    pair_of: str | None = None,
) -> EvalItem:
    return EvalItem(
        id=qid,
        split=Split.DEV,
        turns=("q",),
        in_scope=in_scope,
        slice=slice_,
        oos_kind=oos_kind,
        pair_of=pair_of,
        key_facts=(("x",),) if in_scope else (),
    )


def score(
    of: EvalItem,
    *,
    reason: RefusalReason | None = None,
    passed: bool | None = None,
    coverage: float | None = None,
    ttft: float | None = 500.0,
    total: float = 2000.0,
) -> ItemScore:
    refused = reason is not None
    if of.in_scope and passed is None:
        passed = not refused
    if of.in_scope and coverage is None:
        coverage = 1.0 if passed else 0.0
    return ItemScore(
        item=of,
        refused=refused,
        refusal_reason=reason,
        ttft_ms=None if reason is BT else ttft,
        total_ms=total,
        key_fact_pass=passed if of.in_scope else None,
        coverage=coverage if of.in_scope else None,
    )


def oos(qid: str, kind: OosKind, reason: RefusalReason | None = None) -> ItemScore:
    return score(item(qid, in_scope=False, oos_kind=kind), reason=reason)


# ------------------------------------------------------------------ primary (rule 5)


def test_the_primary_metric_counts_only_single_turn_in_scope_answers_v1_items() -> None:
    scores = [
        score(item("f-1"), passed=True),
        score(item("f-2"), passed=False),
        score(item("f-3"), reason=MR),  # a refusal of an in-scope question fails
        score(item("mt-1", slice_=Slice.MULTI_TURN), passed=True),  # a slice: never pooled
        score(item("lq-1", slice_=Slice.LONG_QUERY, pair_of="f-1"), passed=True),
        oos("oos-1", OosKind.UNRELATED, MR),
    ]

    assert primary(scores) == Proportion(1, 3)


# ------------------------------------------------------------------ refusals


def test_false_refusals_are_refusals_of_in_scope_items() -> None:
    scores = [
        score(item("f-1")),
        score(item("f-2"), reason=BT),
        score(item("mt-1", slice_=Slice.MULTI_TURN), reason=MR),
        oos("oos-1", OosKind.UNRELATED, MR),
        oos("oos-2", OosKind.UNRELATED),
    ]

    assert false_refusals(scores) == Proportion(2, 3)


def test_out_of_scope_recall_is_split_by_kind_and_pooled() -> None:
    scores = [
        oos("oos-1", OosKind.UNRELATED, BT),
        oos("oos-2", OosKind.UNRELATED, MR),
        oos("oos-3", OosKind.NEAR_DOMAIN, MR),
        oos("oos-4", OosKind.NEAR_DOMAIN),
        oos("oos-5", OosKind.NEAR_DOMAIN),
        score(item("f-1"), reason=MR),  # in scope: not part of the recall
        # An adversarial refuse item is out of scope but has no kind: not part of it either.
        score(item("adv-1", in_scope=False, slice_=Slice.ADVERSARIAL), reason=MR),
    ]

    recall = oos_recall(scores)

    assert recall.unrelated == Proportion(2, 2)
    assert recall.near_domain == Proportion(1, 3)
    assert recall.pooled == Proportion(3, 5)


def test_refusals_are_counted_by_reason_with_every_reason_shown() -> None:
    scores = [
        score(item("f-1"), reason=BT),
        score(item("f-2"), reason=BT),
        oos("oos-1", OosKind.UNRELATED, PR),
        score(item("f-3")),
    ]

    assert refusal_reasons(scores) == {BT: 2, MR: 0, PR: 1}


def test_layer_1_refusals_are_listed_by_id_for_the_gate_subset() -> None:
    scores = [score(item("f-1"), reason=BT), score(item("f-2"), reason=MR), score(item("f-3"))]

    assert layer1_refused_ids(scores) == frozenset({"f-1"})


# ------------------------------------------------------------------ per-item values


def test_per_item_values_skip_items_where_the_metric_does_not_apply() -> None:
    scores = [
        score(item("f-1"), passed=True),
        score(item("f-2"), passed=False),
        oos("oos-1", OosKind.UNRELATED, MR),
    ]

    assert per_item(scores, lambda s: s.key_fact_pass) == {"f-1": 1.0, "f-2": 0.0}
    assert per_item(scores, lambda s: s.refused) == {"f-1": 0.0, "f-2": 0.0, "oos-1": 1.0}


def test_duplicate_item_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="f-1"):
        per_item([score(item("f-1")), score(item("f-1"))], lambda s: s.refused)


# ------------------------------------------------------------------ long queries


def test_long_queries_are_compared_with_their_pair_in_the_same_run() -> None:
    scores = [
        score(item("f-1"), passed=True),
        score(item("f-2"), passed=True),
        score(item("f-3"), passed=False, coverage=0.5),
        score(item("f-4"), passed=False),
        score(item("lq-1", slice_=Slice.LONG_QUERY, pair_of="f-1"), passed=True),
        score(item("lq-2", slice_=Slice.LONG_QUERY, pair_of="f-2"), reason=BT),
        score(item("lq-3", slice_=Slice.LONG_QUERY, pair_of="f-3"), passed=True),
        score(item("lq-4", slice_=Slice.LONG_QUERY, pair_of="f-4"), passed=False),
        score(item("lq-5", slice_=Slice.LONG_QUERY, pair_of="f-missing"), passed=True),
    ]

    pairs = long_query_pairs(scores)

    assert asdict(pairs) == pytest.approx(
        {
            "n": 4,
            "unpaired": 1,
            "both": 1,
            "long_only": 1,
            "short_only": 1,
            "neither": 1,
            "pass_difference": 0.0,  # (0 - 1 + 1 + 0) / 4
            "coverage_difference": -0.125,  # (0 - 1 + 0.5 + 0) / 4
        }
    )


def test_no_long_queries_give_an_empty_comparison() -> None:
    pairs = long_query_pairs([score(item("f-1"))])

    assert pairs == LongQueryPairs(
        n=0,
        unpaired=0,
        both=0,
        long_only=0,
        short_only=0,
        neither=0,
        pass_difference=None,
        coverage_difference=None,
    )


# ------------------------------------------------------------------ latency


def test_latency_puts_layer_1_refusals_apart() -> None:
    answered = [
        score(item(f"f-{i}"), ttft=float(t), total=float(10 * t))
        for i, t in enumerate((100, 200, 300, 400, 500))
    ]
    gate = [score(item(f"g-{i}"), reason=BT, total=float(t)) for i, t in enumerate((50, 70))]
    model_refusal = score(item("m-1"), reason=MR, ttft=600.0, total=6000.0)

    summary = latency([*answered, *gate, model_refusal])

    # The model refusal counts with the answers: it made an LLM call.
    assert asdict(summary) == pytest.approx(
        {
            "n": 6,
            "ttft_p50": 350.0,
            "ttft_p95": 575.0,
            "total_p50": 3500.0,
            "total_p95": 5750.0,
            "layer1_n": 2,
            "layer1_total_p50": 60.0,
            "layer1_total_p95": 69.0,
        }
    )


def test_latency_of_an_empty_group_has_no_percentiles() -> None:
    assert latency([]) == LatencySummary(0, None, None, None, None, 0, None, None)

"""The benchmark runner: held-out guard, preflight, runs, summaries, calibration, selection."""

import pytest

from app.domain.evaluation import Category, Metric, OosKind, Split
from app.domain.retrieval import SearchMode
from app.evaluation.metrics import MetricSummary
from app.evaluation.runner import (
    MAX_ANCHOR_MATCHES,
    WARM_UP_RETRIEVALS,
    AnchorProblem,
    ConfigSummary,
    HeldoutRefusedError,
    Latency,
    QuestionRun,
    calibrate,
    preflight,
    run_config,
    select_best,
    select_questions,
    summarize_config,
)
from tests.unit.eval_fakes import (
    OMEN,
    Scripted,
    ScriptedRetriever,
    chunk,
    golden,
    hits,
    named,
    oos,
    question,
)

# ------------------------------------------------------------------ held-out guard


def test_the_dev_split_holds_only_dev_questions() -> None:
    gold = golden(question("a"), question("b", split=Split.HELDOUT), oos("c"))

    assert [q.id for q in select_questions(gold, Split.DEV, allow_heldout=False)] == ["a", "c"]


def test_the_heldout_split_is_refused_without_the_flag() -> None:
    gold = golden(question("a"), question("b", split=Split.HELDOUT))

    with pytest.raises(HeldoutRefusedError):
        select_questions(gold, Split.HELDOUT, allow_heldout=False)


def test_the_heldout_split_is_scored_only_with_the_flag() -> None:
    gold = golden(question("a"), question("b", split=Split.HELDOUT))

    assert [q.id for q in select_questions(gold, Split.HELDOUT, allow_heldout=True)] == ["b"]


# ------------------------------------------------------------------ preflight


def test_preflight_resolves_every_anchor_of_both_splits() -> None:
    gold = golden(
        question("a", facts=(("tray holds 80",), ("two hours",))),
        question("b", split=Split.HELDOUT, facts=(("Phillips M2.0",),), doc_id=OMEN),
        oos("c"),
    )
    chunks = [
        chunk("x", "The TRAY holds 80 sheets."),
        chunk("y", "Off after two hours. The tray holds 80 too."),
        chunk("z", "six Phillips M2.0 screws", doc_id=OMEN),
    ]

    report = preflight(gold, chunks)

    assert report.chunks == 3
    assert [(r.question_id, r.split, r.anchor, r.matches) for r in report.resolutions] == [
        ("a", Split.DEV, 0, 2),
        ("a", Split.DEV, 1, 1),
        ("b", Split.HELDOUT, 0, 1),
    ]
    assert report.ok
    assert report.problems == ()


def test_an_unresolved_anchor_fails_the_preflight_and_names_its_question() -> None:
    gold = golden(question("a"), question("b", facts=(("tray holds 80",), ("no such fact",))))

    report = preflight(gold, [chunk("x", "tray holds 80")])

    assert not report.ok
    assert [(p.question_id, p.anchor, p.matches, p.problem) for p in report.problems] == [
        ("b", 1, 0, AnchorProblem.UNRESOLVED)
    ]


def test_an_anchor_matching_more_than_five_chunks_is_too_generic() -> None:
    gold = golden(question("five"), question("six", facts=(("common",),)))
    chunks = [chunk(f"t{n}", "tray holds 80") for n in range(MAX_ANCHOR_MATCHES)]
    chunks += [chunk(f"c{n}", "common words") for n in range(MAX_ANCHOR_MATCHES + 1)]

    report = preflight(gold, chunks)

    assert [(p.question_id, p.matches, p.problem) for p in report.problems] == [
        ("six", 6, AnchorProblem.TOO_GENERIC)
    ]
    assert report.resolutions[0].problem is None


# ------------------------------------------------------------------ runs


async def test_each_question_is_retrieved_in_order_with_the_named_config() -> None:
    retriever = ScriptedRetriever()
    target = named("dense", mode=SearchMode.DENSE, rerank=False)

    await run_config(target, [question("a"), oos("b")], retriever)

    # One uncounted warm-up retrieval first, then each question once, in order.
    assert retriever.calls == [
        ("question a?", target.config),
        ("question a?", target.config),
        ("question b?", target.config),
    ]


async def test_the_warm_up_retrieval_is_not_counted() -> None:
    retriever = ScriptedRetriever()

    runs = await run_config(named(), [question("a"), oos("b")], retriever)

    assert WARM_UP_RETRIEVALS == 1
    assert [run.question.id for run in runs] == ["a", "b"]
    assert len(retriever.calls) == len(runs) + WARM_UP_RETRIEVALS


async def test_no_questions_means_no_warm_up() -> None:
    retriever = ScriptedRetriever()

    assert await run_config(named(), [], retriever) == ()
    assert retriever.calls == []


async def test_an_in_scope_question_is_scored_over_the_whole_candidate_list() -> None:
    texts = [f"filler {n}" for n in range(1, 20)] + ["the tray holds 80 sheets"]
    retriever = ScriptedRetriever({"question a?": Scripted(hits(*texts))})

    (run,) = await run_config(named(), [question("a")], retriever)

    assert run.scores is not None
    assert run.scores.values[Metric.RECALL_AT_5] == 0.0
    assert run.scores.values[Metric.RECALL_AT_20] == 1.0
    assert run.scores.values[Metric.MRR] == pytest.approx(1 / 20)
    assert run.profile[-1] == frozenset({0})
    assert run.config_name == "hybrid-rerank"


async def test_an_out_of_scope_question_is_not_scored_but_keeps_its_signal() -> None:
    retriever = ScriptedRetriever({"question b?": Scripted(hits("unrelated"), top_score=-4.0)})

    (run,) = await run_config(named(), [oos("b")], retriever)

    assert run.scores is None
    assert run.profile == ()
    assert run.top_score == -4.0


# ------------------------------------------------------------------ calibration


async def runs_with_scores(
    in_scope: list[float | None], out: dict[OosKind, list[float]]
) -> tuple[QuestionRun, ...]:
    """One in-scope question per score (None: no chunks), one out-of-scope per score."""
    script: dict[str, Scripted] = {}
    questions = []
    for n, score in enumerate(in_scope):
        questions.append(question(f"in{n}"))
        found = hits("tray holds 80") if score is not None else []
        script[f"question in{n}?"] = Scripted(found, top_score=score)
    for kind, scores in out.items():
        for n, score in enumerate(scores):
            questions.append(oos(f"{kind.value}{n}", kind))
            script[f"question {kind.value}{n}?"] = Scripted(hits("x"), top_score=score)
    return await run_config(named(), questions, ScriptedRetriever(script))


async def test_the_threshold_is_the_highest_keeping_five_percent_false_refusals() -> None:
    # 48 in-scope questions allow 2 misses: the threshold is the third lowest score.
    in_scope: list[float | None] = [float(n) for n in range(48)]
    runs = await runs_with_scores(
        in_scope, {OosKind.UNRELATED: [-1.0, 1.5, 9.0], OosKind.NEAR_DOMAIN: [1.99, 2.0]}
    )

    result = calibrate(runs)

    assert result.in_scope_n == 48
    assert result.allowed_misses == 2
    assert result.threshold == 2.0
    assert result.false_refusals == 2
    unrelated = result.out_of_scope[OosKind.UNRELATED]
    near = result.out_of_scope[OosKind.NEAR_DOMAIN]
    assert (unrelated.n, unrelated.caught) == (3, 2)
    assert unrelated.share == pytest.approx(2 / 3)
    assert (near.n, near.caught, near.share) == (2, 1, 0.5)
    assert near.scores.low == 1.99
    assert near.scores.high == 2.0
    assert result.in_scope_scores.median == pytest.approx(23.5)


async def test_a_question_with_no_chunks_is_a_forced_miss() -> None:
    in_scope: list[float | None] = [None, *[float(n) for n in range(39)]]  # 40 allow 2

    result = calibrate(await runs_with_scores(in_scope, {}))

    assert result.threshold == 1.0
    assert result.false_refusals == 2
    assert result.in_scope_scores.n == 39
    assert result.out_of_scope == {}


async def test_no_threshold_when_forced_misses_exceed_the_allowance() -> None:
    result = calibrate(await runs_with_scores([None, 1.0, 2.0], {}))

    assert result.allowed_misses == 0
    assert result.threshold is None
    assert result.false_refusals == 1


async def test_no_threshold_without_any_scored_in_scope_question() -> None:
    result = calibrate(await runs_with_scores([], {OosKind.UNRELATED: [1.0]}))

    assert result.threshold is None
    assert result.in_scope_scores.median is None
    assert result.out_of_scope[OosKind.UNRELATED].caught == 0


# ------------------------------------------------------------------ summaries


async def test_a_summary_has_metrics_categories_latency_and_calibration() -> None:
    retriever = ScriptedRetriever(
        {
            "question a?": Scripted(hits("tray holds 80"), total_ms=10.0),
            "question b?": Scripted(hits("nothing"), total_ms=30.0),
            "question c?": Scripted(hits("x"), total_ms=50.0, top_score=-9.0),
        }
    )
    target = named()
    questions = [question("a"), question("b", category=Category.TABLE), oos("c")]
    runs = await run_config(target, questions, retriever)

    summary = summarize_config(target, runs, seed=7, resamples=50)

    assert summary.name == "hybrid-rerank"
    assert summary.metrics[Metric.RECALL_AT_5].n == 2
    assert summary.metrics[Metric.RECALL_AT_5].mean == 0.5
    assert summary.categories[Category.TABLE].means[Metric.RECALL_AT_5] == 0.0
    assert Category.OUT_OF_SCOPE not in summary.categories
    total = summary.latency["total"]
    assert total is not None
    assert total.p50 == 30.0
    assert total.p95 == pytest.approx(48.0)
    assert summary.latency["rerank"] == Latency(p50=5.0, p95=5.0)
    assert summary.calibration is not None
    assert summary.calibration.in_scope_n == 2


async def test_without_rerank_there_is_no_rerank_latency_and_no_calibration() -> None:
    target = named("sparse", mode=SearchMode.SPARSE, rerank=False)
    runs = await run_config(target, [question("a")], ScriptedRetriever())

    summary = summarize_config(target, runs, seed=7, resamples=50)

    assert summary.latency["rerank"] is None
    assert summary.latency["dense_embed"] is None
    assert summary.latency["sparse_embed"] == Latency(1.0, 1.0)
    assert summary.calibration is None
    assert summary.metrics[Metric.RECALL_AT_5].mean == 0.0


# ------------------------------------------------------------------ selection rule


def summary(name: str, recall: tuple[float, float, float], mrr: float, p95: float) -> ConfigSummary:
    mean, low, high = recall
    return ConfigSummary(
        name=name,
        metrics={
            Metric.RECALL_AT_5: MetricSummary(40, mean, low, high),
            Metric.MRR: MetricSummary(40, mrr, mrr, mrr),
        },
        categories={},
        latency={"total": Latency(p50=p95 / 2, p95=p95)},
        calibration=None,
    )


def test_the_highest_recall_wins_when_no_interval_overlaps_it() -> None:
    picked = select_best(
        [summary("a", (0.6, 0.5, 0.7), 0.9, 10), summary("b", (0.9, 0.8, 1.0), 0.1, 999)]
    )

    assert (picked.best, picked.tied) == ("b", ("b",))


def test_overlapping_intervals_are_tied_and_go_to_the_higher_mrr() -> None:
    picked = select_best(
        [
            summary("a", (0.80, 0.70, 0.90), 0.5, 10),
            summary("b", (0.85, 0.75, 0.95), 0.4, 10),
            summary("c", (0.40, 0.30, 0.50), 0.99, 1),
        ]
    )

    assert (picked.best, picked.tied) == ("a", ("a", "b"))


def test_equal_mrr_goes_to_the_lower_p95_latency() -> None:
    picked = select_best(
        [summary("slow", (0.8, 0.7, 0.9), 0.5, 2000), summary("fast", (0.8, 0.7, 0.9), 0.5, 30)]
    )

    assert picked.best == "fast"


def test_a_config_without_scored_questions_never_wins() -> None:
    empty = ConfigSummary(
        name="empty",
        metrics={
            Metric.RECALL_AT_5: MetricSummary(0, None, None, None),
            Metric.MRR: MetricSummary(0, None, None, None),
        },
        categories={},
        latency={"total": None},
        calibration=None,
    )

    picked = select_best([empty, summary("a", (0.1, 0.0, 0.2), 0.1, 10)])

    assert (picked.best, picked.tied) == ("a", ("a",))


def test_selection_needs_at_least_one_config() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_best([])

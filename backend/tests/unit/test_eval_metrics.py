"""Retrieval metrics with hand-computed values. Rankings are given as relevance profiles:
for each rank, the set of anchor positions the chunk at that rank satisfies.
"""

import math

import pytest

from app.domain.evaluation import (
    Anchor,
    Category,
    GoldenQuestion,
    IndexedChunk,
    Match,
    Metric,
    QuestionScores,
    Split,
)
from app.evaluation.metrics import (
    bootstrap_interval,
    by_category,
    hit_at,
    mean,
    ndcg_at,
    percentile,
    recall_at,
    reciprocal_rank,
    relevance_profile,
    score_question,
    summarize,
)

E: frozenset[int] = frozenset()


def a(*anchors: int) -> frozenset[int]:
    return frozenset(anchors)


# ------------------------------------------------------------------ recall and hit rate


def test_recall_under_all_counts_covered_anchors() -> None:
    profile = [E, a(0), E, a(1), E, E, a(2)]

    assert recall_at(profile, 3, Match.ALL, 5) == pytest.approx(2 / 3)
    assert recall_at(profile, 3, Match.ALL, 7) == 1.0
    assert recall_at(profile, 3, Match.ALL, 1) == 0.0


def test_recall_under_any_is_one_as_soon_as_one_anchor_is_covered() -> None:
    profile = [E, E, a(1)]

    assert recall_at(profile, 2, Match.ANY, 3) == 1.0
    assert recall_at(profile, 2, Match.ANY, 2) == 0.0


def test_a_second_chunk_for_a_covered_anchor_earns_nothing() -> None:
    profile = [a(0), a(0), a(0), a(0), a(0)]

    assert recall_at(profile, 2, Match.ALL, 5) == 0.5
    assert ndcg_at(profile, 2, Match.ALL, 5) == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_a_ranking_shorter_than_k_counts_what_it_has() -> None:
    assert recall_at([a(0)], 1, Match.ALL, 20) == 1.0
    assert recall_at([], 1, Match.ALL, 5) == 0.0


def test_hit_rate_needs_any_relevant_chunk_in_the_cut() -> None:
    profile = [E, E, a(0)]

    assert hit_at(profile, 3) == 1.0
    assert hit_at(profile, 2) == 0.0
    assert hit_at([], 5) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_a_cut_below_one_is_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="k"):
        recall_at([a(0)], 1, Match.ALL, k)
    with pytest.raises(ValueError, match="k"):
        hit_at([a(0)], k)
    with pytest.raises(ValueError, match="k"):
        ndcg_at([a(0)], 1, Match.ALL, k)


def test_recall_needs_at_least_one_anchor() -> None:
    with pytest.raises(ValueError, match="anchor"):
        recall_at([E], 0, Match.ANY, 5)
    with pytest.raises(ValueError, match="anchor"):
        ndcg_at([E], 0, Match.ANY, 5)


# ------------------------------------------------------------------ MRR


def test_reciprocal_rank_is_one_over_the_first_relevant_rank() -> None:
    assert reciprocal_rank([E, E, a(1), a(0)]) == pytest.approx(1 / 3)
    assert reciprocal_rank([a(0)]) == 1.0


def test_reciprocal_rank_is_zero_without_a_relevant_result() -> None:
    assert reciprocal_rank([E, E, E]) == 0.0
    assert reciprocal_rank([]) == 0.0


# ------------------------------------------------------------------ nDCG


def test_ndcg_with_two_anchors_found_at_ranks_two_and_four() -> None:
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 + 1 / math.log2(3)

    assert ndcg_at([E, a(0), E, a(1), E], 2, Match.ALL, 5) == pytest.approx(dcg / ideal)
    assert dcg / ideal == pytest.approx(0.6509, abs=1e-4)


def test_ndcg_is_one_for_an_ideal_ranking() -> None:
    assert ndcg_at([a(0), a(1), E], 2, Match.ALL, 5) == pytest.approx(1.0)
    assert ndcg_at([a(1), E], 2, Match.ANY, 5) == pytest.approx(1.0)


def test_under_any_only_the_first_covered_anchor_gains() -> None:
    # Ideal is a single gain at rank 1; the anchor at rank 3 adds nothing.
    assert ndcg_at([E, a(0), a(1)], 2, Match.ANY, 5) == pytest.approx(1 / math.log2(3))


def test_ndcg_ideal_is_capped_at_k() -> None:
    profile = [a(n) for n in range(6)]

    assert ndcg_at(profile, 6, Match.ALL, 5) == pytest.approx(1.0)


def test_ndcg_is_zero_without_a_relevant_result() -> None:
    assert ndcg_at([E, E], 1, Match.ALL, 5) == 0.0
    assert ndcg_at([], 1, Match.ALL, 5) == 0.0


def test_a_chunk_covering_two_new_anchors_gains_once() -> None:
    # Documented limit of binary gains: the ideal assumes one anchor per rank.
    assert ndcg_at([a(0, 1)], 2, Match.ALL, 5) == pytest.approx(1 / (1 + 1 / math.log2(3)))


# ------------------------------------------------------------------ per question


def make_chunk(chunk_id: str, text: str) -> IndexedChunk:
    return IndexedChunk(chunk_id, "omen-guide", ("Removal",), text)


def make_question(
    category: Category = Category.PART_NUMBER,
    match: Match | None = Match.ALL,
    anchors: tuple[Anchor, ...] = (
        Anchor("omen-guide", ("N1",)),
        Anchor("omen-guide", ("N2",)),
    ),
    qid: str = "q1",
) -> GoldenQuestion:
    return GoldenQuestion(qid, category, Split.DEV, "Which part?", match, anchors)


def test_relevance_profile_matches_each_ranked_chunk() -> None:
    ranked = [make_chunk("x", "none"), make_chunk("y", "N2 and N1"), make_chunk("z", "N1")]

    assert relevance_profile(make_question().anchors, ranked) == [E, a(0, 1), a(0)]


def test_score_question_computes_every_metric() -> None:
    ranked = [make_chunk("x", "none"), make_chunk("y", "N1"), *[make_chunk("f", "-")] * 4]
    ranked.append(make_chunk("z", "N2"))

    scores = score_question(make_question(), ranked)

    assert scores.question_id == "q1"
    assert scores.category is Category.PART_NUMBER
    assert scores.values[Metric.RECALL_AT_5] == 0.5
    assert scores.values[Metric.RECALL_AT_20] == 1.0
    assert scores.values[Metric.HIT_AT_5] == 1.0
    assert scores.values[Metric.MRR] == 0.5
    assert scores.values[Metric.NDCG_AT_5] == pytest.approx(
        (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    )


def test_out_of_scope_questions_are_not_scored() -> None:
    question = make_question(Category.OUT_OF_SCOPE, None, ())

    with pytest.raises(ValueError, match="out_of_scope"):
        score_question(question, [])


def test_an_in_scope_question_without_a_match_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="match"):
        score_question(make_question(match=None), [])


# ------------------------------------------------------------------ aggregation


def test_mean_and_percentiles() -> None:
    values = [4.0, 1.0, 3.0, 2.0]

    assert mean(values) == 2.5
    assert percentile(values, 50) == 2.5
    assert percentile(values, 95) == pytest.approx(3.85)
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert percentile([7.0], 95) == 7.0


def test_empty_inputs_have_no_mean_percentile_or_interval() -> None:
    assert mean([]) is None
    assert percentile([], 50) is None
    assert bootstrap_interval([], resamples=10, seed=1) is None


@pytest.mark.parametrize("q", [-1, 101])
def test_a_percentile_outside_zero_to_hundred_is_rejected(q: float) -> None:
    with pytest.raises(ValueError, match="percentile"):
        percentile([1.0], q)


def test_the_seeded_bootstrap_interval_is_deterministic_and_brackets_the_mean() -> None:
    # Continuous values (mean 0.5): binary ones put resampled means on a coarse grid,
    # where two seeds can give the same interval.
    values = [0.13, 0.91, 0.47, 0.05, 0.66, 0.38, 0.72, 0.29, 0.84, 0.55]

    first = bootstrap_interval(values, resamples=500, seed=7)
    again = bootstrap_interval(values, resamples=500, seed=7)
    other = bootstrap_interval(values, resamples=500, seed=8)

    assert first == again
    assert first != other
    assert first is not None
    low, high = first
    assert 0.0 <= low < 0.5 < high <= 1.0


def test_a_constant_sample_has_a_zero_width_interval() -> None:
    assert bootstrap_interval([1.0, 1.0, 1.0], resamples=50, seed=1) == (1.0, 1.0)


def test_the_bootstrap_needs_at_least_one_resample() -> None:
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_interval([1.0], resamples=0, seed=1)


def scored(qid: str, category: Category, value: float) -> QuestionScores:
    question = make_question(category, Match.ANY, (Anchor("omen-guide", ("N1",)),), qid)
    text = "N1" if value else "none"
    return score_question(question, [make_chunk(qid, text)])


def test_summarize_gives_n_mean_and_interval_per_metric() -> None:
    scores = [scored("q1", Category.TABLE, 1), scored("q2", Category.TABLE, 0)]

    summary = summarize(scores, resamples=200, seed=3)

    recall = summary[Metric.RECALL_AT_5]
    assert (recall.n, recall.mean) == (2, 0.5)
    assert recall.low is not None
    assert recall.high is not None
    assert 0.0 <= recall.low <= 0.5 <= recall.high <= 1.0
    assert set(summary) == set(Metric)


def test_summarize_of_nothing_has_zero_n_and_no_values() -> None:
    summary = summarize([], resamples=10, seed=1)

    assert summary[Metric.MRR].n == 0
    assert summary[Metric.MRR].mean is None
    assert summary[Metric.MRR].low is None


def test_by_category_reports_n_and_means_in_first_seen_order() -> None:
    scores = [
        scored("q1", Category.TABLE, 1),
        scored("q2", Category.FACTUAL, 0),
        scored("q3", Category.TABLE, 0),
    ]

    groups = by_category(scores)

    assert list(groups) == [Category.TABLE, Category.FACTUAL]
    table = groups[Category.TABLE]
    assert table.n == 2
    assert table.means[Metric.HIT_AT_5] == 0.5
    assert groups[Category.FACTUAL].means[Metric.MRR] == 0.0

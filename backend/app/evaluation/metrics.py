"""Retrieval metrics over anchors, not chunks (Phase 2 plan, metric definitions).

A ranking is reduced to a relevance profile: for each rank, the positions of the
anchors that the chunk at that rank satisfies. An anchor counts once, so a second
chunk for an already covered anchor earns nothing. Under `match = any` one covered
anchor is enough; under `all` every anchor is needed.
"""

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.domain.evaluation import (
    Anchor,
    Category,
    GoldenQuestion,
    IndexedChunk,
    Match,
    Metric,
    QuestionScores,
)
from app.evaluation.matching import satisfied_anchors

Profile = Sequence[frozenset[int]]

_LOW, _HIGH = 2.5, 97.5  # a 95% percentile interval


@dataclass(frozen=True)
class MetricSummary:
    """One metric over a group of questions; no values when the group is empty."""

    n: int
    mean: float | None
    low: float | None
    high: float | None


@dataclass(frozen=True)
class CategorySummary:
    n: int
    means: Mapping[Metric, float]


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


def _check_anchors(anchors: int) -> None:
    if anchors < 1:
        raise ValueError("a scored question needs at least one anchor")


def _needed(anchors: int, match: Match) -> int:
    return anchors if match is Match.ALL else 1


def relevance_profile(
    anchors: Sequence[Anchor], ranked: Sequence[IndexedChunk]
) -> list[frozenset[int]]:
    return [satisfied_anchors(anchors, chunk) for chunk in ranked]


def recall_at(profile: Profile, anchors: int, match: Match, k: int) -> float:
    _check_k(k)
    _check_anchors(anchors)
    covered = len(frozenset().union(*profile[:k]))
    if match is Match.ANY:
        return 1.0 if covered else 0.0
    return covered / anchors


def hit_at(profile: Profile, k: int) -> float:
    _check_k(k)
    return 1.0 if any(profile[:k]) else 0.0


def reciprocal_rank(profile: Profile) -> float:
    """Over the whole ranking given; 0 when no chunk satisfies any anchor."""
    return next((1.0 / rank for rank, hits in enumerate(profile, 1) if hits), 0.0)


def ndcg_at(profile: Profile, anchors: int, match: Match, k: int) -> float:
    """Binary gains: a rank gains 1 when it covers an anchor not covered above it,
    until the needed anchors are covered. The ideal puts one gain on each of the
    first min(needed, k) ranks. Known limit: a chunk that covers two new anchors
    still gains 1, so such a ranking scores below 1.
    """
    _check_k(k)
    _check_anchors(anchors)
    needed = _needed(anchors, match)
    covered: set[int] = set()
    dcg = 0.0
    for rank, hits in enumerate(profile[:k], 1):
        if len(covered) < needed and hits - covered:
            dcg += 1.0 / math.log2(rank + 1)
        covered |= hits
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(needed, k) + 1))
    return dcg / ideal


def score_question(question: GoldenQuestion, ranked: Sequence[IndexedChunk]) -> QuestionScores:
    """Score one in-scope question against its ranked chunks (pass the candidates)."""
    if not question.in_scope:
        raise ValueError(f"{question.id}: out_of_scope questions are not ranked")
    if question.match is None:
        raise ValueError(f"{question.id}: an in-scope question needs a match mode")
    profile = relevance_profile(question.anchors, ranked)
    n, match = len(question.anchors), question.match
    return QuestionScores(
        question_id=question.id,
        category=question.category,
        values={
            Metric.RECALL_AT_5: recall_at(profile, n, match, 5),
            Metric.RECALL_AT_20: recall_at(profile, n, match, 20),
            Metric.HIT_AT_5: hit_at(profile, 5),
            Metric.MRR: reciprocal_rank(profile),
            Metric.NDCG_AT_5: ndcg_at(profile, n, match, 5),
        },
    )


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _interpolate(ordered: Sequence[float], q: float) -> float:
    position = (len(ordered) - 1) * q / 100
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default method)."""
    if not 0 <= q <= 100:
        raise ValueError("percentile must be between 0 and 100")
    return _interpolate(sorted(values), q) if values else None


def bootstrap_interval(
    values: Sequence[float], *, resamples: int, seed: int
) -> tuple[float, float] | None:
    """95% percentile interval of the mean, resampling questions with a fixed seed,
    so a report can be reproduced exactly.
    """
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    if not values:
        return None
    rng = random.Random(seed)  # noqa: S311 - resampling for statistics, not security  # nosec B311
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(resamples))
    return _interpolate(means, _LOW), _interpolate(means, _HIGH)


def summarize(
    scores: Sequence[QuestionScores], *, resamples: int, seed: int
) -> dict[Metric, MetricSummary]:
    summary: dict[Metric, MetricSummary] = {}
    for metric in Metric:
        values = [score.values[metric] for score in scores]
        interval = bootstrap_interval(values, resamples=resamples, seed=seed)
        low, high = interval if interval is not None else (None, None)
        summary[metric] = MetricSummary(len(values), mean(values), low, high)
    return summary


def by_category(scores: Sequence[QuestionScores]) -> dict[Category, CategorySummary]:
    groups: dict[Category, list[QuestionScores]] = {}
    for score in scores:
        groups.setdefault(score.category, []).append(score)
    return {
        category: CategorySummary(
            n=len(members),
            means={m: sum(s.values[m] for s in members) / len(members) for m in Metric},
        )
        for category, members in groups.items()
    }

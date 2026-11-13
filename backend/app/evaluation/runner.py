"""The retrieval benchmark: gates, runs, summaries, threshold calibration, selection.

Pure orchestration over a `Retriever` (the retrieval service in production), so every
rule of the Phase 2 plan is unit-tested without models or Qdrant:

- the held-out split is never scored unless explicitly allowed;
- the preflight resolves every anchor of both splits against the indexed chunks, and
  an anchor that matches no chunk, or more than five, stops the run;
- questions run one at a time, so the timings are not contended by the benchmark;
- the relevance threshold is the highest top reranker score that keeps false
  refusals of in-scope questions at or below 5%;
- the best configuration has the highest Recall@5; configurations whose bootstrap
  intervals overlap the leader's are tied, and ties go to the higher MRR, then the
  lower p95 latency, then matrix order.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Protocol

from app.domain.evaluation import (
    Anchor,
    Category,
    GoldenQuestion,
    GoldenSet,
    IndexedChunk,
    Metric,
    OosKind,
    QuestionScores,
    Split,
)
from app.domain.retrieval import RetrievalConfig, RetrievalResult, RetrievedChunk
from app.evaluation.matching import matching_chunks
from app.evaluation.matrix import NamedConfig
from app.evaluation.metrics import (
    CategorySummary,
    MetricSummary,
    by_category,
    percentile,
    relevance_profile,
    score_question,
    summarize,
)

MAX_ANCHOR_MATCHES = 5
MAX_FALSE_REFUSALS = Fraction(5, 100)
STAGES = ("dense_embed", "sparse_embed", "search", "rerank", "total")


class HeldoutRefusedError(Exception):
    """The held-out split is scored only in Phase 5, and only when asked for."""


class AnchorProblem(StrEnum):
    UNRESOLVED = "unresolved"
    TOO_GENERIC = "too_generic"


@dataclass(frozen=True)
class MatchedChunk:
    """Which chunk an anchor matched: ids and the citation path, never its text."""

    chunk_id: str
    doc_id: str
    section_path: tuple[str, ...]


@dataclass(frozen=True)
class AnchorResolution:
    question_id: str
    split: Split
    anchor: int
    matched: tuple[MatchedChunk, ...]

    @property
    def matches(self) -> int:
        return len(self.matched)

    @property
    def problem(self) -> AnchorProblem | None:
        if self.matches == 0:
            return AnchorProblem.UNRESOLVED
        if self.matches > MAX_ANCHOR_MATCHES:
            return AnchorProblem.TOO_GENERIC
        return None


@dataclass(frozen=True)
class Preflight:
    chunks: int
    resolutions: tuple[AnchorResolution, ...]

    @property
    def problems(self) -> tuple[AnchorResolution, ...]:
        return tuple(r for r in self.resolutions if r.problem is not None)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def multi_match(self) -> tuple[AnchorResolution, ...]:
        """Anchors matching two or more chunks, reviewed for false positive coverage."""
        return tuple(r for r in self.resolutions if r.matches >= 2)


@dataclass(frozen=True)
class IndexInfo:
    """The index a run read: the alias it went through (None for an explicit
    collection) and the manifest build behind the collection, with the BM25 avg_len its
    sparse vectors were made with (the avg_len ablation reads another build).
    """

    alias: str | None
    collection: str
    build_id: str
    chunk_count: int
    chunking_version: str
    bm25_avg_len: float
    bm25_avg_len_auto: bool


class Retriever(Protocol):
    async def retrieve(self, query: str, config: RetrievalConfig) -> RetrievalResult: ...


@dataclass(frozen=True)
class QuestionRun:
    config_name: str
    question: GoldenQuestion
    result: RetrievalResult
    profile: tuple[frozenset[int], ...]  # anchors each ranked chunk satisfies
    scores: QuestionScores | None  # None for out-of-scope questions

    @property
    def top_score(self) -> float | None:
        """The relevance signal: the top reranker score, None without rerank."""
        return self.result.relevance.signal


@dataclass(frozen=True)
class Distribution:
    n: int
    low: float | None
    p25: float | None
    median: float | None
    p75: float | None
    high: float | None


@dataclass(frozen=True)
class OutOfScopeCatch:
    n: int
    caught: int
    scores: Distribution

    @property
    def share(self) -> float | None:
        return self.caught / self.n if self.n else None


@dataclass(frozen=True)
class Calibration:
    threshold: float | None  # None when no value keeps the false refusals allowed
    in_scope_n: int
    allowed_misses: int
    false_refusals: int
    in_scope_scores: Distribution
    out_of_scope: Mapping[OosKind, OutOfScopeCatch]


@dataclass(frozen=True)
class Latency:
    p50: float
    p95: float


@dataclass(frozen=True)
class ConfigSummary:
    name: str
    metrics: Mapping[Metric, MetricSummary]
    categories: Mapping[Category, CategorySummary]
    latency: Mapping[str, Latency | None]  # None for a stage the config skips
    calibration: Calibration | None  # None without rerank: no relevance signal


@dataclass(frozen=True)
class Selection:
    best: str
    tied: tuple[str, ...]
    p95_tie_margin_ms: float


# ---------------------------------------------------------------- gates


def select_questions(
    golden: GoldenSet, split: Split, *, allow_heldout: bool
) -> tuple[GoldenQuestion, ...]:
    if split is Split.HELDOUT and not allow_heldout:
        raise HeldoutRefusedError("the held-out split needs --allow-heldout")
    return golden.in_split(split)


def preflight(golden: GoldenSet, chunks: Sequence[IndexedChunk]) -> Preflight:
    """Every anchor of both splits: resolving one reads no score, so held-out is safe."""
    by_id = {c.chunk_id: c for c in chunks}

    def matched(anchor: Anchor) -> tuple[MatchedChunk, ...]:
        found = (by_id[chunk_id] for chunk_id in matching_chunks(anchor, chunks))
        return tuple(MatchedChunk(c.chunk_id, c.doc_id, c.section_path) for c in found)

    resolutions = tuple(
        AnchorResolution(q.id, q.split, n, matched(anchor))
        for q in golden.questions
        for n, anchor in enumerate(q.anchors)
    )
    return Preflight(len(chunks), resolutions)


# ---------------------------------------------------------------- runs


def _indexed(chunk: RetrievedChunk) -> IndexedChunk:
    hit = chunk.hit
    return IndexedChunk(hit.chunk_id, hit.doc_id, hit.section_path, hit.text)


# A configuration's first call can pay one-off costs (a reranker loaded on demand, buffers
# sized for a new shape), which would inflate its p95 latency, a tie-break (ADR 0009).
WARM_UP_RETRIEVALS = 1


async def run_config(
    named: NamedConfig, questions: Sequence[GoldenQuestion], retriever: Retriever
) -> tuple[QuestionRun, ...]:
    """Every question once, in order, after one uncounted warm-up retrieval."""
    for question in questions[:WARM_UP_RETRIEVALS]:
        await retriever.retrieve(question.question, named.config)
    runs = []
    for question in questions:
        result = await retriever.retrieve(question.question, named.config)
        ranked = [_indexed(chunk) for chunk in result.chunks]
        in_scope = question.in_scope
        runs.append(
            QuestionRun(
                config_name=named.name,
                question=question,
                result=result,
                profile=tuple(relevance_profile(question.anchors, ranked)) if in_scope else (),
                scores=score_question(question, ranked) if in_scope else None,
            )
        )
    return tuple(runs)


# ---------------------------------------------------------------- calibration


def distribution(values: Sequence[float]) -> Distribution:
    return Distribution(
        n=len(values),
        low=min(values, default=None),
        p25=percentile(values, 25),
        median=percentile(values, 50),
        p75=percentile(values, 75),
        high=max(values, default=None),
    )


def _threshold(scores: Sequence[float], forced: int, allowed: int) -> float | None:
    """The (allowed - forced + 1)-th lowest score: any higher value refuses one more."""
    room = allowed - forced
    if room < 0 or room >= len(scores):
        return None
    return sorted(scores)[room]


def calibrate(runs: Sequence[QuestionRun]) -> Calibration:
    """A refusal is a top score below the threshold; an in-scope question with no
    chunks has no score and is always a refusal (a forced miss).
    """
    in_scope = [run.top_score for run in runs if run.question.in_scope]
    scored = [score for score in in_scope if score is not None]
    forced = len(in_scope) - len(scored)
    allowed = math.floor(len(in_scope) * MAX_FALSE_REFUSALS)
    threshold = _threshold(scored, forced, allowed)
    refused = forced + (sum(s < threshold for s in scored) if threshold is not None else 0)
    groups: dict[OosKind, list[float | None]] = {}
    for run in runs:
        if run.question.oos_kind is not None:
            groups.setdefault(run.question.oos_kind, []).append(run.top_score)
    return Calibration(
        threshold=threshold,
        in_scope_n=len(in_scope),
        allowed_misses=allowed,
        false_refusals=refused,
        in_scope_scores=distribution(scored),
        out_of_scope={kind: _caught(scores, threshold) for kind, scores in sorted(groups.items())},
    )


def _caught(scores: Sequence[float | None], threshold: float | None) -> OutOfScopeCatch:
    present = [score for score in scores if score is not None]
    caught = len(scores) - len(present)
    if threshold is not None:
        caught += sum(score < threshold for score in present)
    return OutOfScopeCatch(len(scores), caught, distribution(present))


# ---------------------------------------------------------------- summaries


def _stage_ms(result: RetrievalResult, stage: str) -> float | None:
    value: float | None = getattr(result.timings, f"{stage}_ms")
    return value


def _latency(runs: Sequence[QuestionRun], stage: str) -> Latency | None:
    values = [v for run in runs if (v := _stage_ms(run.result, stage)) is not None]
    p50, p95 = percentile(values, 50), percentile(values, 95)
    return Latency(p50, p95) if p50 is not None and p95 is not None else None


def summarize_config(
    named: NamedConfig, runs: Sequence[QuestionRun], *, seed: int, resamples: int
) -> ConfigSummary:
    scores = [run.scores for run in runs if run.scores is not None]
    return ConfigSummary(
        name=named.name,
        metrics=summarize(scores, resamples=resamples, seed=seed),
        categories=by_category(scores),
        latency={stage: _latency(runs, stage) for stage in STAGES},
        calibration=calibrate(runs) if named.config.rerank else None,
    )


# ---------------------------------------------------------------- selection rule


def _or_worst(value: float | None) -> float:
    return -math.inf if value is None else value


def _overlaps(a: MetricSummary, b: MetricSummary) -> bool:
    if a.low is None or a.high is None or b.low is None or b.high is None:
        return False
    return a.low <= b.high and b.low <= a.high


def select_best(summaries: Sequence[ConfigSummary], *, p95_tie_margin_ms: float) -> Selection:
    """Recall@5 first, then MRR, then p95 latency, then matrix order.

    A p95 gap within the margin is a tie: run-to-run noise on one machine (the
    reference moved by about 1.3 s between identical runs in Phase 2) must not change
    a default. The earliest config within the margin of the fastest wins (ADR 0009).
    """
    if not summaries:
        raise ValueError("selection needs at least one config")
    if p95_tie_margin_ms < 0:
        raise ValueError("the p95 tie margin cannot be negative")
    leader = max(summaries, key=lambda s: _or_worst(s.metrics[Metric.RECALL_AT_5].mean))
    lead = leader.metrics[Metric.RECALL_AT_5]
    tied = [s for s in summaries if s is leader or _overlaps(s.metrics[Metric.RECALL_AT_5], lead)]

    def p95(summary: ConfigSummary) -> float:
        total = summary.latency.get("total")
        return total.p95 if total is not None else math.inf

    top_mrr = max(_or_worst(s.metrics[Metric.MRR].mean) for s in tied)
    finalists = [s for s in tied if _or_worst(s.metrics[Metric.MRR].mean) == top_mrr]
    fastest = min(p95(s) for s in finalists)
    best = next(s for s in finalists if p95(s) <= fastest + p95_tie_margin_ms)
    return Selection(best.name, tuple(s.name for s in tied), p95_tie_margin_ms)

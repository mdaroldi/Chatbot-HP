"""Run-level aggregates of the answer benchmark (Phase 5 section 6, definitions fixed).

Aggregates read item scores only; error results never reach them (the run health
report counts those). Which items each metric covers:
- primary: single-turn in-scope answers-v1 items (rule 5), never pooled with slices;
- false refusals: every in-scope item;
- out-of-scope recall: items with an `oos_kind` (answers-v1 out-of-scope questions and
  the out_of_scope slice), so adversarial refuse items, which have none, stay apart;
- long queries: each long item against its pair in the same run;
- latency: as the client saw it, with layer 1 refusals (no LLM call) apart.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from app.domain.chat import RefusalReason
from app.domain.evaluation import OosKind
from app.evaluation.answers.scoring import ItemScore
from app.evaluation.answers.stats import Proportion
from app.evaluation.metrics import percentile


@dataclass(frozen=True)
class OosRecall:
    unrelated: Proportion
    near_domain: Proportion
    pooled: Proportion


@dataclass(frozen=True)
class LongQueryPairs:
    """Long minus short over the pairs found; `unpaired` long items had no scored pair
    (the pair is not in the run, or it ended in an error).
    """

    n: int
    unpaired: int
    both: int
    long_only: int
    short_only: int
    neither: int
    pass_difference: float | None
    coverage_difference: float | None


@dataclass(frozen=True)
class LatencySummary:
    n: int
    ttft_p50: float | None
    ttft_p95: float | None
    total_p50: float | None
    total_p95: float | None
    layer1_n: int
    layer1_total_p50: float | None
    layer1_total_p95: float | None


def primary(scores: Iterable[ItemScore]) -> Proportion:
    return Proportion.of(score.key_fact_pass for score in scores if score.item.primary)


def false_refusals(scores: Iterable[ItemScore]) -> Proportion:
    return Proportion.of(score.refused for score in scores if score.item.in_scope)


def oos_recall(scores: Iterable[ItemScore]) -> OosRecall:
    kinds = [(score.item.oos_kind, score.refused) for score in scores if score.item.oos_kind]

    def of(kind: OosKind | None) -> Proportion:
        return Proportion.of(refused for k, refused in kinds if kind in (None, k))

    return OosRecall(of(OosKind.UNRELATED), of(OosKind.NEAR_DOMAIN), of(None))


def refusal_reasons(scores: Iterable[ItemScore]) -> dict[RefusalReason, int]:
    counts = dict.fromkeys(RefusalReason, 0)
    for score in scores:
        if score.refusal_reason is not None:
            counts[score.refusal_reason] += 1
    return counts


def layer1_refused_ids(scores: Iterable[ItemScore]) -> frozenset[str]:
    """For the gate subset of rule 7: compare runs on items no layer 1 gate refused."""
    return frozenset(score.item.id for score in scores if score.layer1_refused)


def per_item(
    scores: Iterable[ItemScore], metric: Callable[[ItemScore], bool | float | None]
) -> dict[str, float]:
    """Item id to value, for the paired bootstrap; items the metric skips are left out."""
    values: dict[str, float] = {}
    for score in scores:
        if score.item.id in values:
            raise ValueError(f"duplicate item id: {score.item.id}")
        value = metric(score)
        if value is not None:
            values[score.item.id] = float(value)
    return values


def long_query_pairs(scores: Iterable[ItemScore]) -> LongQueryPairs:
    by_id = {score.item.id: score for score in scores}
    long_items = [score for score in by_id.values() if score.item.pair_of is not None]
    pairs = [(s, by_id[s.item.pair_of]) for s in long_items if s.item.pair_of in by_id]
    flags = [(bool(long.key_fact_pass), bool(short.key_fact_pass)) for long, short in pairs]
    n = len(pairs)

    def mean_gap(values: Sequence[tuple[float, float]]) -> float | None:
        return sum(a - b for a, b in values) / n if n else None

    coverages = [(long.coverage or 0.0, short.coverage or 0.0) for long, short in pairs]
    return LongQueryPairs(
        n=n,
        unpaired=len(long_items) - n,
        both=flags.count((True, True)),
        long_only=flags.count((True, False)),
        short_only=flags.count((False, True)),
        neither=flags.count((False, False)),
        pass_difference=mean_gap([(float(a), float(b)) for a, b in flags]),
        coverage_difference=mean_gap(coverages),
    )


def latency(scores: Iterable[ItemScore]) -> LatencySummary:
    scored = list(scores)
    llm = [score for score in scored if not score.layer1_refused]
    gate = [score.total_ms for score in scored if score.layer1_refused]
    ttft = [score.ttft_ms for score in llm if score.ttft_ms is not None]
    total = [score.total_ms for score in llm]
    return LatencySummary(
        n=len(llm),
        ttft_p50=percentile(ttft, 50),
        ttft_p95=percentile(ttft, 95),
        total_p50=percentile(total, 50),
        total_p95=percentile(total, 95),
        layer1_n=len(gate),
        layer1_total_p50=percentile(gate, 50),
        layer1_total_p95=percentile(gate, 95),
    )

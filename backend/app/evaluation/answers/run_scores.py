"""Item scores of a finished run, from its files alone (journal and traces).

Each answered item is observed through its final result's `done` event (content,
citation tags, refusal reason, flags) and the trace of that same answer: the excerpts
the model was given, the rewritten query, and the server timings stored with the
answer. Latency comes from those stored timings (owner, sixth session): they exist for
every item, while the client clock was wrong for the pilot items before `a875b9e`.
Errored results are not scored; the run health report counts them.
Error messages name items and counts only.
"""

from collections.abc import Mapping, Sequence

from app.domain.answer_dataset import EvalItem
from app.evaluation.answers.client import ItemResult
from app.evaluation.answers.scoring import ItemScore, Observation, score_item
from app.evaluation.answers.sse import Done
from app.evaluation.answers.traces import Trace


class RunScoreError(Exception):
    """The run's files cannot be scored; names items and counts only."""


def observation(done: Done, trace: Trace) -> Observation:
    final = trace.rows[-1] if trace.rows else None
    if final is None or final.message_id != done.message_id:
        raise RunScoreError(f"{trace.item_id}: the trace is not of its final answer")
    if final.total_ms is None:
        raise RunScoreError(f"{trace.item_id}: the final answer has no stored total time")
    return Observation(
        answer=done.content,
        refusal_reason=done.refusal_reason,
        citations=done.citations,
        flags=done.flags,
        excerpts=trace.excerpts,
        ttft_ms=final.ttft_ms,
        total_ms=final.total_ms,
        rewritten_query=final.rewritten_query,
    )


def score_run(
    items: Sequence[EvalItem], results: Mapping[str, ItemResult], traces: Mapping[str, Trace]
) -> tuple[ItemScore, ...]:
    """`results` and `traces` are each item's latest; scores keep the items' order."""
    missing = [item.id for item in items if item.id not in results]
    if missing:
        raise RunScoreError(f"{len(missing)} of {len(items)} items have no result")
    scores = []
    for item in items:
        answered = results[item.id].answer
        if answered is None:
            continue
        traced = traces.get(item.id)
        if traced is None:
            raise RunScoreError(f"{item.id}: no trace of its answer")
        scores.append(score_item(item, observation(answered.done, traced)))
    return tuple(scores)

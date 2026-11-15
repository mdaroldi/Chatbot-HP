"""What the judge sees for each answer of a finished run.

Only in-scope items whose final result is an answer without a refusal are judged
(lever 1a, `judge.judged_ids`). Each input holds the conversation, the excerpts exactly
as the answer prompt rendered them (from the trace of the answer's own attempt), and
the reference: the gold passages (every indexed chunk satisfying one of the item's
anchors) plus the answers-v1 key facts. An item with no anchors (some adversarial
items) is judged against its key facts alone.

Gold passages come from the run's own index: an anchor that matches no chunk means the
wrong collection was read, so the build stops rather than judge against less.
Error messages name items and anchor positions only, never question or fact wording.
"""

from collections.abc import Mapping, Sequence

from app.domain.answer_dataset import EvalItem
from app.domain.evaluation import Anchor, IndexedChunk
from app.evaluation.answers.client import ItemResult
from app.evaluation.answers.judge import JudgeInput, judged_ids
from app.evaluation.answers.sse import Done
from app.evaluation.answers.traces import Trace
from app.evaluation.matching import anchor_matches


class JudgeInputError(Exception):
    """The run's files or index cannot give a judge input; names ids only."""


def gold_passages(anchors: Sequence[Anchor], chunks: Sequence[IndexedChunk]) -> tuple[str, ...]:
    """Matching chunk texts, anchor by anchor in index order, each chunk once."""
    seen: dict[str, str] = {}
    for position, anchor in enumerate(anchors, start=1):
        matched = [chunk for chunk in chunks if anchor_matches(anchor, chunk)]
        if not matched:
            raise JudgeInputError(f"anchor {position} matches no indexed chunk")
        for chunk in matched:
            seen.setdefault(chunk.chunk_id, chunk.text)
    return tuple(seen.values())


def _answers(results: Mapping[str, ItemResult]) -> dict[str, Done]:
    return {
        item_id: result.answer.done
        for item_id, result in results.items()
        if result.answer is not None  # the journal holds an answer or an error, never both
    }


def _input(
    item: EvalItem, done: Done, trace: Trace | None, chunks: Sequence[IndexedChunk]
) -> JudgeInput:
    if trace is None or not trace.rows or trace.rows[-1].message_id != done.message_id:
        raise JudgeInputError(f"{item.id}: no trace of its final answer")
    try:
        passages = gold_passages(item.anchors, chunks)
    except JudgeInputError as exc:
        raise JudgeInputError(f"{item.id}: {exc}") from None
    return JudgeInput(
        item_id=item.id,
        turns=item.turns,
        excerpts=tuple(excerpt.text for excerpt in trace.excerpts),
        passages=passages,
        key_facts=item.key_facts,
        answer=done.content,
    )


def judge_inputs(
    items: Sequence[EvalItem],
    results: Mapping[str, ItemResult],
    traces: Mapping[str, Trace],
    chunks: Sequence[IndexedChunk],
) -> tuple[JudgeInput, ...]:
    """`results` and `traces` are each item's latest (`journal.latest`, `Traces.latest`)."""
    by_id = {item.id: item for item in items}
    answers = _answers(results)
    refused = {item_id: done.refusal_reason is not None for item_id, done in answers.items()}
    return tuple(
        _input(by_id[item_id], answers[item_id], traces.get(item_id), chunks)
        for item_id in judged_ids(items, refused)
    )

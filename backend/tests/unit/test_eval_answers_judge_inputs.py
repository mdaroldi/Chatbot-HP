"""Judge inputs from a finished run: only in-scope answered items, each with the
excerpts its answer was given, its conversation, its gold passages and key facts.
"""

import pytest

from app.domain.answer_dataset import EvalItem, Slice
from app.domain.chat import AnswerFlags, RefusalReason
from app.domain.evaluation import Anchor, IndexedChunk, Split
from app.evaluation.answers.client import Answered, ItemResult
from app.evaluation.answers.judge import JudgeInput
from app.evaluation.answers.judge_inputs import (
    JudgeInputError,
    RunNotFinishedError,
    gold_passages,
    judge_inputs,
    require_finished,
)
from app.evaluation.answers.scoring import Excerpt
from app.evaluation.answers.sse import Done
from app.evaluation.answers.traces import Trace
from tests.unit.test_eval_answers_charging import row

DOC = "demo-guide"
WIFI = Anchor(DOC, ("wi-fi",))
TRAY = Anchor(DOC, ("tray",), section="Paper")
CHUNKS = (
    IndexedChunk("c-1", DOC, ("Setup", "Network"), "Turn on Wi-Fi."),
    IndexedChunk("c-2", DOC, ("Setup", "Paper"), "Load the tray."),
    IndexedChunk("c-3", DOC, ("Setup", "Network"), "Wi-Fi Direct and the tray."),
    IndexedChunk("c-4", "other-guide", ("Setup",), "Wi-Fi elsewhere."),
)


def item(item_id: str, *, in_scope: bool = True, anchors: tuple[Anchor, ...] = (WIFI,)) -> EvalItem:
    return EvalItem(
        id=item_id,
        split=Split.DEV,
        turns=("Earlier?", f"Question {item_id}?"),
        in_scope=in_scope,
        slice=Slice.MULTI_TURN,
        anchors=anchors,
        key_facts=(("fact a", "re:fact\\s+a"), ("fact b",)),
    )


def result(
    item_id: str,
    *,
    refusal: RefusalReason | None = None,
    error: str | None = None,
    message_id: str = "m-1",
) -> ItemResult:
    if error is not None:
        return ItemResult(item_id, "client", "conv", 0, None, error)
    done = Done("conv", message_id, f"Answer {item_id} [S1].", refusal, ("S1",), AnswerFlags())
    return ItemResult(item_id, "client", "conv", 0, Answered(done, 0, 500.0, 2000.0))


def trace(item_id: str, message_id: str = "m-1") -> Trace:
    excerpts = tuple(
        Excerpt(tag, CHUNKS[n], f"<excerpt {tag} {item_id}>") for n, tag in enumerate(("S1", "S2"))
    )
    return Trace(item_id, 1, "conv", (row(0), row(1, message_id=message_id)), excerpts)


def test_an_answered_item_becomes_one_full_judge_input() -> None:
    [built] = judge_inputs([item("q-1")], {"q-1": result("q-1")}, {"q-1": trace("q-1")}, CHUNKS)

    assert built == JudgeInput(
        item_id="q-1",
        turns=("Earlier?", "Question q-1?"),
        excerpts=("<excerpt S1 q-1>", "<excerpt S2 q-1>"),
        passages=("Turn on Wi-Fi.", "Wi-Fi Direct and the tray."),
        key_facts=(("fact a", "re:fact\\s+a"), ("fact b",)),
        answer="Answer q-1 [S1].",
    )


def test_only_in_scope_items_answered_without_a_refusal_are_judged_in_item_order() -> None:
    items = [item("q-4"), item("q-1"), item("oos", in_scope=False), item("q-2"), item("q-3")]
    results = {
        "q-1": result("q-1"),
        "oos": result("oos"),
        "q-2": result("q-2", refusal=RefusalReason.MODEL_REFUSED),
        "q-3": result("q-3", error="read_timeout"),
        "q-4": result("q-4"),
    }
    traces = {key: trace(key) for key in results}

    built = judge_inputs(items, results, traces, CHUNKS)
    assert [b.item_id for b in built] == ["q-4", "q-1"]


def test_an_item_without_anchors_has_key_facts_but_no_passages() -> None:
    [built] = judge_inputs(
        [item("adv", anchors=())], {"adv": result("adv")}, {"adv": trace("adv")}, CHUNKS
    )
    assert built.passages == ()
    assert built.key_facts


def test_an_answered_item_with_no_trace_stops_the_build() -> None:
    with pytest.raises(JudgeInputError, match="q-1"):
        judge_inputs([item("q-1")], {"q-1": result("q-1")}, {}, CHUNKS)


def test_a_trace_from_another_answer_stops_the_build() -> None:
    with pytest.raises(JudgeInputError, match="q-1"):
        judge_inputs(
            [item("q-1")], {"q-1": result("q-1")}, {"q-1": trace("q-1", "m-other")}, CHUNKS
        )


def test_gold_passages_follow_anchor_then_index_order_without_repeats() -> None:
    assert gold_passages((TRAY, WIFI), CHUNKS) == (
        "Load the tray.",
        "Turn on Wi-Fi.",
        "Wi-Fi Direct and the tray.",
    )


def test_an_anchor_matching_no_chunk_names_the_anchor_never_its_facts() -> None:
    missing = Anchor(DOC, ("secret wording",))
    with pytest.raises(JudgeInputError, match="anchor 2") as raised:
        gold_passages((WIFI, missing), CHUNKS)
    assert "secret" not in str(raised.value)


def test_an_item_whose_anchor_matches_no_chunk_stops_the_build_naming_the_item() -> None:
    lost = item("q-1", anchors=(Anchor("other-guide", ("tray",)),))
    with pytest.raises(JudgeInputError, match="q-1: anchor 1"):
        judge_inputs([lost], {"q-1": result("q-1")}, {"q-1": trace("q-1")}, CHUNKS)


def test_a_run_with_every_item_recorded_is_finished_even_with_errors() -> None:
    items = [item("q-1"), item("q-2")]
    require_finished(items, {"q-1": result("q-1"), "q-2": result("q-2", error="read_timeout")})


def test_a_run_with_items_not_recorded_is_not_finished_and_says_how_many() -> None:
    items = [item("q-1"), item("q-2"), item("q-3")]
    with pytest.raises(RunNotFinishedError, match="2 of 3 items"):
        require_finished(items, {"q-1": result("q-1")})

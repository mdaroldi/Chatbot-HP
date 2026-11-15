"""Scoring a finished run from its files: each answered item's observation comes from
its final result and the trace of that answer (excerpts, rewritten query, the stored
server timings); errored items are left to the run health report.
"""

import pytest

from app.domain.answer_dataset import EvalItem
from app.domain.chat import AnswerFlags, RefusalReason
from app.domain.evaluation import Split
from app.evaluation.answers.client import Answered, ItemResult
from app.evaluation.answers.run_scores import RunScoreError, observation, score_run
from app.evaluation.answers.scoring import Excerpt
from app.evaluation.answers.sse import Done
from app.evaluation.answers.traces import Trace
from tests.unit.test_eval_answers_charging import row
from tests.unit.test_eval_answers_judge_inputs import CHUNKS


def item(item_id: str, *, in_scope: bool = True) -> EvalItem:
    return EvalItem(
        id=item_id,
        split=Split.DEV,
        turns=(f"Question {item_id}?",),
        in_scope=in_scope,
        key_facts=(("fact a",),) if in_scope else (),
    )


def result(
    item_id: str,
    content: str = "It is fact a [S1].",
    *,
    refusal: RefusalReason | None = None,
    message_id: str = "m-1",
) -> ItemResult:
    done = Done("conv", message_id, content, refusal, ("S1",), AnswerFlags())
    # Client timings differ from the stored ones on purpose: the score must not use them.
    return ItemResult(item_id, "client", "conv", 0, Answered(done, 0, 1.0, 2.0))


def errored(item_id: str) -> ItemResult:
    return ItemResult(item_id, "client", "conv", 0, None, "read_timeout")


def trace(item_id: str, *, message_id: str = "m-1", **final: object) -> Trace:
    excerpts = (Excerpt("S1", CHUNKS[0], "<excerpt S1>fact a</excerpt>"),)
    rows = (row(0), row(1, message_id=message_id, **final))
    return Trace(item_id, 1, "conv", rows, excerpts)


def test_an_observation_takes_the_answer_from_done_and_the_rest_from_its_trace() -> None:
    answered = result("q-1")
    assert answered.answer is not None
    traced = trace("q-1", rewritten_query="standalone", ttft_ms=610.0, total_ms=2710.0)

    seen = observation(answered.answer.done, traced)

    assert seen.answer == "It is fact a [S1]."
    assert seen.refusal_reason is None
    assert seen.citations == ("S1",)
    assert seen.excerpts == traced.excerpts
    assert seen.rewritten_query == "standalone"
    assert (seen.ttft_ms, seen.total_ms) == (610.0, 2710.0)  # server, not client


def test_a_layer1_refusal_has_no_ttft_but_a_server_total() -> None:
    refused = result("q-1", "Not in the documents.", refusal=RefusalReason.BELOW_THRESHOLD)
    assert refused.answer is not None
    seen = observation(refused.answer.done, trace("q-1", ttft_ms=None, total_ms=1950.0))
    assert (seen.ttft_ms, seen.total_ms) == (None, 1950.0)


def test_a_trace_of_another_answer_is_refused() -> None:
    answered = result("q-1", message_id="m-1")
    assert answered.answer is not None
    with pytest.raises(RunScoreError, match="q-1"):
        observation(answered.answer.done, trace("q-1", message_id="m-9"))


def test_a_final_row_without_a_stored_total_is_refused() -> None:
    answered = result("q-1")
    assert answered.answer is not None
    with pytest.raises(RunScoreError, match="q-1"):
        observation(answered.answer.done, trace("q-1", total_ms=None))


def test_score_run_scores_answers_in_item_order_and_leaves_errors_out() -> None:
    items = [item("q-2"), item("q-1"), item("q-3"), item("oos", in_scope=False)]
    results = {
        "q-1": result("q-1"),
        "q-2": result("q-2", "Something else [S1]."),
        "q-3": errored("q-3"),
        "oos": result("oos", "Not in the documents.", refusal=RefusalReason.MODEL_REFUSED),
    }
    traces = {key: trace(key) for key in ("q-1", "q-2", "oos")}

    scores = score_run(items, results, traces)

    assert [s.item.id for s in scores] == ["q-2", "q-1", "oos"]
    assert [s.key_fact_pass for s in scores] == [False, True, None]
    assert scores[2].refused


def test_score_run_needs_a_result_for_every_item() -> None:
    with pytest.raises(RunScoreError, match="1 of 2"):
        score_run([item("q-1"), item("q-2")], {"q-1": result("q-1")}, {"q-1": trace("q-1")})


def test_score_run_needs_the_trace_of_each_answer() -> None:
    with pytest.raises(RunScoreError, match="q-1"):
        score_run([item("q-1")], {"q-1": result("q-1")}, {})

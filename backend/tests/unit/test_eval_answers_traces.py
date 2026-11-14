"""Reading a run's trace file back: every settled item's stored rows and excerpts, and
every failed delete, strictly, for scoring and the run report.
"""

import json
from pathlib import Path

import pytest

from app.domain.chat import AnswerFlags, Outcome, RefusalReason
from app.domain.evaluation import IndexedChunk
from app.evaluation.answers.client import Answered, Failed, ItemResult
from app.evaluation.answers.scoring import Excerpt
from app.evaluation.answers.settle import Settled, TraceLog
from app.evaluation.answers.sse import Done
from app.evaluation.answers.traces import DeleteFailure, Trace, TraceError, read_traces
from tests.unit.test_eval_answers_charging import row


def excerpt(tag: str = "S1", chunk_id: str = "c-1") -> Excerpt:
    chunk = IndexedChunk(chunk_id, "demo-guide", ("Setup", "Wi-Fi"), f"Text of {chunk_id}.")
    return Excerpt(tag, chunk, f"<excerpt {tag}>Text of {chunk_id}.</excerpt>")


def result(item_id: str = "q-1", conversation: str | None = "conv-1") -> ItemResult:
    done = Done(conversation or "", "m-1", "Answer [S1].", None, ("S1",), AnswerFlags())
    return ItemResult(item_id, "client-1", conversation, 0, Answered(done, 0, 500.0, 2000.0))


def test_a_missing_file_has_no_traces(tmp_path: Path) -> None:
    traces = read_traces(tmp_path / "traces.jsonl")
    assert (traces.traces, traces.delete_failures) == ((), ())


def test_written_traces_read_back_equal(tmp_path: Path) -> None:
    log = TraceLog(tmp_path / "traces.jsonl")
    refused = row(
        2,
        outcome=Outcome.REFUSED,
        refusal_reason=RefusalReason.BELOW_THRESHOLD,
        model=None,
        input_tokens=None,
        output_tokens=None,
        relevance_signal=-9.5,
    )
    first = Settled((row(1), refused), (excerpt("S1", "c-2"), excerpt("S2", "c-1")), None)
    log.trace(result(), 1, first)
    log.trace(result("q-2"), 2, Settled((), (), None))

    traces = read_traces(log.path)

    assert traces.traces == (
        Trace("q-1", 1, "conv-1", first.rows, first.excerpts),
        Trace("q-2", 2, "conv-1", (), ()),
    )
    assert traces.delete_failures == ()


def test_failed_deletes_read_back_with_their_conversation(tmp_path: Path) -> None:
    log = TraceLog(tmp_path / "traces.jsonl")
    log.delete_failed(result(), 1, Failed("transport_error", None, 0))
    log.delete_failed(result("q-2", "conv-2"), 1, Failed("http_error", 500, 0))

    traces = read_traces(log.path)

    assert traces.delete_failures == (
        DeleteFailure("q-1", 1, "conv-1", "client-1", "transport_error", None),
        DeleteFailure("q-2", 1, "conv-2", "client-1", "http_error", 500),
    )


def test_latest_keeps_each_items_last_attempt(tmp_path: Path) -> None:
    log = TraceLog(tmp_path / "traces.jsonl")
    log.trace(result(), 1, Settled((row(1),), (), None))
    log.trace(result(), 2, Settled((row(7),), (), None))
    log.trace(result("q-2"), 1, Settled((), (), None))

    latest = read_traces(log.path).latest()

    assert latest["q-1"].attempt == 2
    assert latest["q-1"].rows == (row(7),)
    assert set(latest) == {"q-1", "q-2"}


def write(path: Path, *objects: object) -> Path:
    path.write_text("".join(json.dumps(obj) + "\n" for obj in objects), encoding="utf-8")
    return path


def good(tmp_path: Path) -> dict[str, object]:
    log = TraceLog(tmp_path / "source.jsonl")
    log.trace(result(), 1, Settled((row(1),), (excerpt(),), None))
    return json.loads(log.path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"schema": "answers-trace-v0"}, "line 2: schema"),
        ({"kind": "note"}, "line 2: kind"),
        ({"attempt": 0}, "line 2: attempt"),
        ({"rows": [{"message_id": "m-1"}]}, "line 2: rows"),
        ({"excerpts": [{"tag": "S1"}]}, "line 2: excerpts"),
        ({"extra": True}, "line 2: extra"),
        ({"item_id": 5}, "line 2: item_id"),
    ],
)
def test_a_bad_line_fails_naming_the_line_and_field(
    tmp_path: Path, change: dict[str, object], problem: str
) -> None:
    line = good(tmp_path)
    path = write(tmp_path / "traces.jsonl", line, line | change)
    with pytest.raises(TraceError, match=problem):
        read_traces(path)


def test_a_bad_value_never_appears_in_the_message(tmp_path: Path) -> None:
    line = good(tmp_path) | {"item_id": ["secret question text"]}
    with pytest.raises(TraceError) as caught:
        read_traces(write(tmp_path / "traces.jsonl", line))
    assert "secret" not in str(caught.value)


def test_a_partial_last_line_fails(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text('{"schema": "answers-trace-v1"', encoding="utf-8")
    with pytest.raises(TraceError, match="line 1: partial line"):
        read_traces(path)

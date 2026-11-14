"""The append-only results journal and resume: results survive a crash line by line,
are read back strictly, and a resumed run asks only the items not yet recorded.
"""

import json
from pathlib import Path

import pytest

from app.domain.answer_dataset import EvalItem
from app.domain.chat import AnswerFlags, RefusalReason
from app.domain.evaluation import Split
from app.evaluation.answers.client import Answered, HarnessCode, ItemResult
from app.evaluation.answers.journal import (
    JOURNAL_SCHEMA,
    Journal,
    JournalError,
    latest,
    pending,
    read_journal,
    retried_items,
    run_pending,
    spread,
)
from app.evaluation.answers.sse import Done


def answered(item_id: str = "q-1", *, refused: bool = False) -> ItemResult:
    done = Done(
        conversation_id=f"c-{item_id}",
        message_id=f"m-{item_id}",
        content="No answer." if refused else "Clear the tray [S1].\nThen close it.",
        refusal_reason=RefusalReason.MODEL_REFUSED if refused else None,
        citations=() if refused else ("S1",),
        flags=AnswerFlags(uncited=refused, invalid_citations=1, rewrite_fallback="timeout"),
    )
    answer = Answered(done, retries=1, ttft_ms=None if refused else 812.5, total_ms=2040.25)
    return ItemResult(item_id, f"client-{item_id}", f"c-{item_id}", 2, answer, setup_refused=1)


def failed(item_id: str = "q-2") -> ItemResult:
    return ItemResult(
        item_id,
        f"client-{item_id}",
        f"c-{item_id}",
        retries=2,
        answer=None,
        error=HarnessCode.SETUP_TURN_FAILED,
        cause="llm_error",
        http_status=502,
    )


def creation_failed(item_id: str = "q-3") -> ItemResult:
    return ItemResult(item_id, f"client-{item_id}", None, 0, None, "internal_error", None, 500)


def item(item_id: str) -> EvalItem:
    return EvalItem(id=item_id, split=Split.DEV, turns=(f"question {item_id}",), in_scope=True)


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "results.jsonl"


# ------------------------------------------------------------------ write and read


@pytest.mark.parametrize(
    "result", [answered(), answered(refused=True), failed(), creation_failed()]
)
def test_every_kind_of_result_round_trips(path: Path, result: ItemResult) -> None:
    Journal(path).append(result)

    assert read_journal(path) == [result]


def test_each_result_is_one_line_appended_in_order(path: Path) -> None:
    journal = Journal(path)
    for result in (answered("q-1"), failed("q-2"), answered("q-3")):
        journal.append(result)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["item_id"] for line in lines] == ["q-1", "q-2", "q-3"]
    assert all(json.loads(line)["schema"] == JOURNAL_SCHEMA for line in lines)


def test_appending_never_rewrites_what_is_there(path: Path) -> None:
    Journal(path).append(answered("q-1"))
    before = path.read_bytes()

    Journal(path).append(failed("q-2"))

    assert path.read_bytes().startswith(before)
    assert [r.item_id for r in read_journal(path)] == ["q-1", "q-2"]


def test_a_missing_journal_is_an_empty_one(path: Path) -> None:
    assert read_journal(path) == []


def test_a_partial_last_line_stops_the_read(path: Path) -> None:
    Journal(path).append(answered("q-1"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema": "answers-journal-v1", "item_')

    with pytest.raises(JournalError, match="line 2"):
        read_journal(path)


def _write_record(path: Path, record: dict[str, object]) -> None:
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def _record() -> dict[str, object]:
    result = answered()
    return {
        "schema": JOURNAL_SCHEMA,
        "item_id": result.item_id,
        "client_id": result.client_id,
        "conversation_id": result.conversation_id,
        "retries": 2,
        "error": None,
        "cause": None,
        "http_status": None,
        "setup_refused": 1,
        "answer": {
            "done": {
                "conversation_id": "c-q-1",
                "message_id": "m-q-1",
                "outcome": "completed",
                "refusal": False,
                "refusal_reason": None,
                "content": "x",
                "citations": ["S1"],
                "flags": {
                    "late_refusal_marker": False,
                    "uncited": False,
                    "invalid_citations": 0,
                    "rewrite_fallback": None,
                    "truncated": False,
                },
            },
            "retries": 0,
            "ttft_ms": 1.0,
            "total_ms": 2.0,
        },
    }


def _answer() -> dict[str, object]:
    answer = _record()["answer"]
    assert isinstance(answer, dict)
    return answer


def test_a_hand_written_record_in_the_documented_shape_is_read(path: Path) -> None:
    _write_record(path, _record())

    (result,) = read_journal(path)

    assert result.ok
    assert result.answer is not None
    assert result.answer.done.content == "x"


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"schema": "answers-journal-v0"}, "schema"),
        ({"item_id": ""}, "item_id"),
        ({"client_id": 7}, "client_id"),
        ({"conversation_id": 7}, "conversation_id"),
        ({"retries": -1}, "retries"),
        ({"retries": True}, "retries"),
        ({"setup_refused": "1"}, "setup_refused"),
        ({"error": 3}, "error"),
        ({"cause": 3}, "cause"),
        ({"http_status": "502"}, "http_status"),
        ({"answer": None}, "answer"),
        ({"answer": []}, "answer"),
        ({"error": "llm_error"}, "answer"),
    ],
)
def test_a_record_with_a_wrong_field_stops_the_read(
    path: Path, change: dict[str, object], match: str
) -> None:
    _write_record(path, {**_record(), **change})

    with pytest.raises(JournalError, match=f"line 1: .*{match}"):
        read_journal(path)


@pytest.mark.parametrize("key", ["item_id", "retries", "answer", "setup_refused"])
def test_a_record_missing_a_field_stops_the_read(path: Path, key: str) -> None:
    record = _record()
    del record[key]
    _write_record(path, record)

    with pytest.raises(JournalError, match=f"line 1: .*{key}"):
        read_journal(path)


@pytest.mark.parametrize(
    "answer",
    [
        {"retries": 0, "ttft_ms": 1.0, "total_ms": 2.0},
        {"done": {"content": "x"}, "retries": 0, "ttft_ms": 1.0, "total_ms": 2.0},
        {**_answer(), "ttft_ms": "1"},
        {**_answer(), "total_ms": None},
        {**_answer(), "retries": -1},
    ],
)
def test_a_malformed_answer_stops_the_read(path: Path, answer: dict[str, object]) -> None:
    _write_record(path, {**_record(), "answer": answer})

    with pytest.raises(JournalError, match="line 1"):
        read_journal(path)


@pytest.mark.parametrize("line", ["not json", "[1, 2]"])
def test_a_line_that_is_not_a_json_object_stops_the_read(path: Path, line: str) -> None:
    path.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(JournalError, match="line 1: not a JSON object"):
        read_journal(path)


def test_errors_name_the_line_never_the_content(path: Path) -> None:
    _write_record(path, {**_record(), "client_id": 7, "secret_answer": "printer secret"})

    with pytest.raises(JournalError) as raised:
        read_journal(path)

    assert "printer secret" not in str(raised.value)


# ------------------------------------------------------------------ resume


def test_pending_skips_recorded_items_including_errors() -> None:
    items = [item("q-1"), item("q-2"), item("q-3"), item("q-4")]

    left = pending(items, [answered("q-1"), failed("q-3")])

    assert [i.id for i in left] == ["q-2", "q-4"]


def test_a_journal_with_a_duplicate_item_is_refused() -> None:
    with pytest.raises(JournalError, match="recorded twice: q-1"):
        pending([item("q-1")], [answered("q-1"), answered("q-1")])


def test_a_journal_from_another_selection_is_refused() -> None:
    with pytest.raises(JournalError, match="not in this run: q-9"):
        pending([item("q-1")], [answered("q-9")])


def test_a_retried_error_may_be_recorded_again_and_the_last_line_wins() -> None:
    final = latest([failed("q-1"), failed("q-1"), answered("q-1"), answered("q-2")])

    assert final["q-1"] == answered("q-1")
    assert final["q-2"] == answered("q-2")


def test_an_answered_item_is_never_recorded_again() -> None:
    with pytest.raises(JournalError, match="recorded twice: q-1"):
        latest([answered("q-1"), failed("q-1")])


def test_retry_errors_selects_only_items_whose_last_result_is_an_error() -> None:
    items = [item("q-1"), item("q-2"), item("q-3"), item("q-4")]
    recorded = [failed("q-1"), answered("q-2"), failed("q-3"), answered("q-3")]

    left = pending(items, recorded, retry_errors=True)

    assert [i.id for i in left] == ["q-1", "q-4"]


def test_retried_items_are_those_with_more_than_one_line() -> None:
    recorded = [failed("q-1"), answered("q-1"), answered("q-2"), failed("q-3"), failed("q-3")]

    assert retried_items(recorded) == ("q-1", "q-3")


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    async def __call__(self, run_item: EvalItem, client_id: str, attempt: int) -> ItemResult:
        self.calls.append((run_item.id, client_id, attempt))
        return answered(run_item.id)


async def test_a_run_asks_each_item_once_with_a_fresh_client_id_and_records_it(
    path: Path,
) -> None:
    ids = iter(["id-a", "id-b"])
    ask = Recorder()

    results = await run_pending([item("q-1"), item("q-2")], Journal(path), ask, lambda: next(ids))

    assert ask.calls == [("q-1", "id-a", 1), ("q-2", "id-b", 1)]
    assert [r.item_id for r in results] == ["q-1", "q-2"]
    assert read_journal(path) == results


async def test_a_resumed_run_asks_only_what_is_missing_and_returns_everything(
    path: Path,
) -> None:
    journal = Journal(path)
    journal.append(failed("q-1"))
    ask = Recorder()

    results = await run_pending([item("q-1"), item("q-2")], journal, ask, lambda: "id")

    assert ask.calls == [("q-2", "id", 1)]
    assert [r.item_id for r in results] == ["q-1", "q-2"]
    assert [r.item_id for r in read_journal(path)] == ["q-1", "q-2"]


async def test_results_come_back_in_item_order_whatever_the_journal_order(path: Path) -> None:
    journal = Journal(path)
    journal.append(answered("q-2"))

    results = await run_pending([item("q-1"), item("q-2")], journal, Recorder(), lambda: "id")

    assert [r.item_id for r in results] == ["q-1", "q-2"]


async def test_a_retry_errors_run_re_asks_failed_items_and_keeps_both_lines(path: Path) -> None:
    journal = Journal(path)
    journal.append(failed("q-1"))
    journal.append(answered("q-2"))
    ask = Recorder()

    results = await run_pending(
        [item("q-1"), item("q-2")], journal, ask, lambda: "id", retry_errors=True
    )

    assert ask.calls == [("q-1", "id", 2)]
    assert results == [answered("q-1"), answered("q-2")]
    recorded = read_journal(path)
    assert [r.item_id for r in recorded] == ["q-1", "q-2", "q-1"]
    assert retried_items(recorded) == ("q-1",)


async def test_without_retry_errors_a_failed_item_stays_failed(path: Path) -> None:
    journal = Journal(path)
    journal.append(failed("q-1"))

    results = await run_pending([item("q-1")], journal, Recorder(), lambda: "id")

    assert results == [failed("q-1")]


def test_spread_takes_evenly_spaced_items_starting_with_the_first() -> None:
    assert spread(list(range(10)), 3) == [0, 3, 6]
    assert spread(list(range(105)), 10) == [0, 10, 21, 31, 42, 52, 63, 73, 84, 94]


def test_spread_with_a_limit_at_or_above_the_length_takes_everything() -> None:
    assert spread([1, 2, 3], 3) == [1, 2, 3]
    assert spread([1, 2, 3], 7) == [1, 2, 3]


def test_spread_rejects_a_limit_below_one() -> None:
    with pytest.raises(ValueError, match="limit"):
        spread([1, 2], 0)


async def test_a_limited_run_asks_a_spread_of_the_pending_items_and_returns_only_results(
    path: Path,
) -> None:
    journal = Journal(path)
    journal.append(answered("q-1"))
    ask = Recorder()
    items = [item(f"q-{n}") for n in range(1, 6)]

    results = await run_pending(items, journal, ask, lambda: "id", limit=2)

    assert [call[0] for call in ask.calls] == ["q-2", "q-4"]
    assert [r.item_id for r in results] == ["q-1", "q-2", "q-4"]


async def test_a_later_unlimited_run_finishes_what_a_limited_run_left(path: Path) -> None:
    journal = Journal(path)
    items = [item(f"q-{n}") for n in range(1, 5)]
    await run_pending(items, journal, Recorder(), lambda: "id", limit=2)
    ask = Recorder()

    results = await run_pending(items, journal, ask, lambda: "id")

    assert [call[0] for call in ask.calls] == ["q-2", "q-4"]
    assert [r.item_id for r in results] == ["q-1", "q-2", "q-3", "q-4"]

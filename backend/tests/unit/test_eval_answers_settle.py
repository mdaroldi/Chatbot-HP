"""Settling an asked item: charge the ledger, save the trace, then delete the conversation.

Fakes implement the reader, chunk source and deleter ports; the ledger and the trace
file are real files, read back as JSON lines.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from app.domain.chat import AnswerFlags
from app.domain.ingestion import ContentType, PageRange
from app.domain.retrieval import SearchHit
from app.evaluation.answers.budget import Kind, Ledger, UnsoundBoundError
from app.evaluation.answers.client import Answered, Failed, ItemResult
from app.evaluation.answers.settle import (
    TRACE_SCHEMA,
    Settler,
    StoreMismatchError,
    TraceLog,
)
from app.evaluation.answers.sse import Done
from app.evaluation.answers.store import StoredAnswer
from tests.unit.test_eval_answers_budget import MATRIX, SONNET, item
from tests.unit.test_eval_answers_charging import row

RUN = "reference-1"
TITLES = {"demo-guide": "Demo Guide"}


def hit(chunk_id: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        doc_id="demo-guide",
        section_path=("Setup", chunk_id),
        pages=PageRange(3, 3, None, None),
        content_type=ContentType.PROCEDURE,
        has_warning=False,
        text=f"Text of {chunk_id}.",
        score=0.0,
    )


class FakeRows:
    def __init__(self, rows: Sequence[StoredAnswer] = (), *, broken: bool = False) -> None:
        self.rows, self.broken = tuple(rows), broken
        self.asked: list[tuple[str, str]] = []

    async def answers(self, conversation_id: str, client_id: str) -> tuple[StoredAnswer, ...]:
        self.asked.append((conversation_id, client_id))
        if self.broken:
            raise ConnectionError
        return self.rows


class FakeChunks:
    async def hits(self, chunk_ids: Sequence[str]) -> Mapping[str, SearchHit]:
        return {chunk_id: hit(chunk_id) for chunk_id in chunk_ids}


class FakeDeleter:
    def __init__(self, failure: Failed | None = None, log: list[str] | None = None) -> None:
        self.failure, self.log = failure, log if log is not None else []
        self.deleted: list[tuple[str, str]] = []

    async def delete_conversation(self, conversation_id: str, client_id: str) -> Failed | None:
        self.deleted.append((conversation_id, client_id))
        self.log.append("delete")
        return self.failure


def answered(message_id: str = "m-1") -> ItemResult:
    done = Done("conv-1", message_id, "Answer [S1].", None, ("S1",), AnswerFlags())
    return ItemResult("q-1", "client-1", "conv-1", 0, Answered(done, 0, 500.0, 2000.0))


def settler(
    tmp_path: Path, rows: FakeRows, deleter: FakeDeleter | None = None
) -> tuple[Settler, Ledger, TraceLog, FakeDeleter]:
    ledger, traces = Ledger(tmp_path / "ledger.jsonl"), TraceLog(tmp_path / "traces.jsonl")
    deleter = deleter or FakeDeleter()
    made = Settler(MATRIX, RUN, rows, FakeChunks(), TITLES, deleter, ledger, traces)
    return made, ledger, traces, deleter


def lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def test_an_answer_is_charged_traced_with_its_excerpts_and_then_deleted(
    tmp_path: Path,
) -> None:
    stored = row(1, chunk_ids=("c-2", "c-1"))
    rows = FakeRows([stored])
    settle, ledger, traces, deleter = settler(tmp_path, rows)

    settled = await settle(item(), answered(), attempt=2)

    assert rows.asked == [("conv-1", "client-1")]
    assert [(c.kind, c.model, c.input_tokens, c.attempt) for c in ledger.charges()] == [
        (Kind.ANSWER, SONNET, stored.input_tokens, 2)
    ]
    assert [e.tag for e in settled.excerpts] == ["S1", "S2"]
    assert [e.chunk.chunk_id for e in settled.excerpts] == ["c-2", "c-1"]
    assert settled.rows == (stored,)
    (trace,) = lines(traces.path)
    assert trace["schema"] == TRACE_SCHEMA
    assert (trace["kind"], trace["item_id"], trace["attempt"]) == ("trace", "q-1", 2)
    assert trace["conversation_id"] == "conv-1"
    assert trace["rows"] == [json.loads(json.dumps(stored.__dict__))]
    # Anchor matching needs the chunk itself, the judge the rendered text.
    excerpts = cast(list[dict[str, object]], trace["excerpts"])
    assert excerpts[0] == {
        "tag": "S1",
        "chunk_id": "c-2",
        "doc_id": "demo-guide",
        "section_path": ["Setup", "c-2"],
        "chunk_text": "Text of c-2.",
        "text": settled.excerpts[0].text,
    }
    assert [e["chunk_id"] for e in excerpts] == ["c-2", "c-1"]
    assert deleter.deleted == [("conv-1", "client-1")]
    assert settled.delete_failure is None


async def test_the_conversation_is_deleted_only_after_its_trace_is_saved(tmp_path: Path) -> None:
    order: list[str] = []
    traces_path = tmp_path / "traces.jsonl"

    class WatchingDeleter(FakeDeleter):
        async def delete_conversation(self, conversation_id: str, client_id: str) -> Failed | None:
            order.append("trace saved" if traces_path.exists() else "trace missing")
            return await super().delete_conversation(conversation_id, client_id)

    settle, _, _, _ = settler(tmp_path, FakeRows([row(1)]), WatchingDeleter())

    await settle(item(), answered(), attempt=1)

    assert order == ["trace saved"]


async def test_a_failed_delete_is_recorded_for_cleanup_and_does_not_stop_the_run(
    tmp_path: Path,
) -> None:
    failure = Failed("http_error", 500, 0)
    settle, _, traces, _ = settler(tmp_path, FakeRows([row(1)]), FakeDeleter(failure))

    settled = await settle(item(), answered(), attempt=1)

    assert settled.delete_failure == failure
    trace, cleanup = lines(traces.path)
    assert trace["kind"] == "trace"
    assert cleanup == {
        "schema": TRACE_SCHEMA,
        "kind": "delete_failed",
        "item_id": "q-1",
        "attempt": 1,
        "conversation_id": "conv-1",
        "client_id": "client-1",
        "code": "http_error",
        "http_status": 500,
    }


async def test_an_error_result_is_charged_and_deleted_but_has_no_excerpts(
    tmp_path: Path,
) -> None:
    failed = ItemResult("q-1", "client-1", "conv-1", 0, None, "read_timeout")
    settle, ledger, traces, deleter = settler(tmp_path, FakeRows([row(1)]))

    settled = await settle(item(turns=2), failed, attempt=1)

    assert len(ledger.charges()) == 3  # row 1, then turn 2's rewrite and answer bounds
    assert settled.excerpts == ()
    assert lines(traces.path)[0]["excerpts"] == []
    assert deleter.deleted == [("conv-1", "client-1")]


async def test_without_a_conversation_nothing_is_read_charged_traced_or_deleted(
    tmp_path: Path,
) -> None:
    failed = ItemResult("q-1", "client-1", None, 2, None, "http_error", http_status=503)
    rows = FakeRows([row(1)])
    settle, ledger, traces, deleter = settler(tmp_path, rows)

    settled = await settle(item(), failed, attempt=1)

    assert (settled.rows, settled.excerpts, settled.delete_failure) == ((), (), None)
    assert (rows.asked, ledger.charges(), deleter.deleted) == ([], [], [])
    assert not traces.path.exists()

    await settle(item(), answered(), attempt=1)  # the same settler, now with a conversation
    assert (len(rows.asked), len(ledger.charges()), len(lines(traces.path))) == (1, 1, 1)


@pytest.mark.parametrize(
    "result",
    [answered(), ItemResult("q-1", "client-1", "conv-1", 0, None, "read_timeout")],
    ids=["answered", "error"],
)
async def test_an_unreadable_store_charges_every_turn_at_its_bound_and_stops(
    tmp_path: Path, result: ItemResult
) -> None:
    """Without rows nobody knows how far an errored item got, so all turns count."""
    settle, ledger, traces, deleter = settler(tmp_path, FakeRows(broken=True))

    with pytest.raises(ConnectionError):
        await settle(item(turns=2), result, attempt=1)

    assert [(c.kind, c.estimated) for c in ledger.charges()] == [
        (Kind.ANSWER, True),
        (Kind.REWRITE, True),
        (Kind.ANSWER, True),
    ]
    assert (traces.path.exists(), deleter.deleted) == (False, [])


@pytest.mark.parametrize(("stored", "charged"), [((), 1), ((1,), 1)])
async def test_a_final_row_that_is_not_the_answered_message_stops_before_deleting(
    tmp_path: Path, stored: tuple[int, ...], charged: int
) -> None:
    settle, ledger, traces, deleter = settler(tmp_path, FakeRows([row(n) for n in stored]))

    with pytest.raises(StoreMismatchError, match="m-other"):
        await settle(item(), answered("m-other"), attempt=1)

    # The paid call stays on the ledger: measured, or at its bound when no row exists.
    assert len(ledger.charges()) == charged
    assert (traces.path.exists(), deleter.deleted) == (False, [])


async def test_a_measured_call_above_its_bound_stops_once_everything_is_saved(
    tmp_path: Path,
) -> None:
    over = row(1, input_tokens=MATRIX.worst_case.answer_input_tokens + 1)
    settle, ledger, traces, deleter = settler(tmp_path, FakeRows([over]))

    with pytest.raises(UnsoundBoundError):
        await settle(item(), answered(), attempt=1)

    assert len(ledger.charges()) == 1
    assert lines(traces.path)[0]["kind"] == "trace"
    assert deleter.deleted == [("conv-1", "client-1")]

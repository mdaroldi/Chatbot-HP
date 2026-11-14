"""Settling an asked item: charge what it cost, save what it stored, then clean up.

Order matters, from what must never be lost to what can be redone:
1. Read the conversation's rows (READ ONLY). If the store cannot be read, every turn
   is charged at its bound before the error stops the run: money already spent is
   never left off the ledger.
2. Append the item's charges to the ledger (`charging.item_charges`).
3. For an answered item, check that the last row is the answered message, then rebuild
   the excerpts the model saw from its chunk ids.
4. Append the trace (rows, and each excerpt with its chunk for anchor matching and its
   rendered text for the judge) to the run's trace file.
5. Delete the conversation through the API. A failed delete does not stop the run; it
   is appended as a `delete_failed` line naming the conversation, so leftovers are
   visible and can be removed later (the retention purge also removes them).
6. Only then check the measured charges against their bounds: a call above its bound
   stops the run with everything saved.

The trace file holds stored answers' traces and HP excerpt text, so it lives in the
git-ignored run directory, next to the results journal.
"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from app.domain.answer_dataset import EvalItem
from app.evaluation.answers.answer_matrix import AnswerMatrix
from app.evaluation.answers.budget import Charge, Ledger, check_measured
from app.evaluation.answers.charging import item_charges
from app.evaluation.answers.client import Failed, ItemResult
from app.evaluation.answers.excerpts import ChunkSource, rebuild_excerpts
from app.evaluation.answers.jsonl import Json, append_object
from app.evaluation.answers.scoring import Excerpt
from app.evaluation.answers.store import AnswerRows, StoredAnswer

TRACE_SCHEMA = "answers-trace-v1"


class StoreMismatchError(Exception):
    """The stored rows do not end with the answered message; ids only in the message."""


class Deleter(Protocol):
    async def delete_conversation(self, conversation_id: str, client_id: str) -> Failed | None:
        """None once deleted."""
        ...


@dataclass(frozen=True)
class Settled:
    rows: tuple[StoredAnswer, ...]
    excerpts: tuple[Excerpt, ...]  # empty unless the item was answered
    delete_failure: Failed | None


class TraceLog:
    """The run's stored traces and failed deletes; git-ignored, append-only."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def trace(self, result: ItemResult, attempt: int, settled: Settled) -> None:
        self._append(
            result,
            attempt,
            kind="trace",
            rows=[cast(Json, dataclasses.asdict(row)) for row in settled.rows],
            excerpts=[
                {
                    "tag": e.tag,
                    "chunk_id": e.chunk.chunk_id,
                    "doc_id": e.chunk.doc_id,
                    "section_path": list(e.chunk.section_path),
                    "chunk_text": e.chunk.text,
                    "text": e.text,
                }
                for e in settled.excerpts
            ],
        )

    def delete_failed(self, result: ItemResult, attempt: int, failure: Failed) -> None:
        self._append(
            result,
            attempt,
            kind="delete_failed",
            client_id=result.client_id,
            code=failure.code,
            http_status=failure.http_status,
        )

    def _append(self, result: ItemResult, attempt: int, *, kind: str, **fields: object) -> None:
        record: Json = {
            "schema": TRACE_SCHEMA,
            "kind": kind,
            "item_id": result.item_id,
            "attempt": attempt,
            "conversation_id": result.conversation_id,
        }
        append_object(self.path, record | fields)


class Settler:
    def __init__(  # noqa: PLR0913, PLR0917 - the ports and files one settlement touches
        self,
        matrix: AnswerMatrix,
        run: str,
        rows: AnswerRows,
        chunks: ChunkSource,
        titles: Mapping[str, str],
        deleter: Deleter,
        ledger: Ledger,
        traces: TraceLog,
    ) -> None:
        self._matrix, self._run = matrix, run
        self._rows, self._chunks, self._titles = rows, chunks, titles
        self._deleter, self._ledger, self._traces = deleter, ledger, traces

    async def __call__(self, item: EvalItem, result: ItemResult, *, attempt: int) -> Settled:
        conversation = result.conversation_id
        if conversation is None:
            return Settled((), (), None)
        try:
            rows = await self._rows.answers(conversation, result.client_id)
        except Exception:
            # Unknown usage: every turn at its bound, as if none had been stored.
            self._charge(item, dataclasses.replace(result, error=None), (), attempt)
            raise
        charges = self._charge(item, result, rows, attempt)
        excerpts = await self._excerpts(result, rows)
        settled = Settled(rows, excerpts, None)
        self._traces.trace(result, attempt, settled)
        failure = await self._deleter.delete_conversation(conversation, result.client_id)
        if failure is not None:
            self._traces.delete_failed(result, attempt, failure)
            settled = dataclasses.replace(settled, delete_failure=failure)
        for made in charges:
            check_measured(self._matrix, made)
        return settled

    def _charge(
        self,
        item: EvalItem,
        result: ItemResult,
        rows: tuple[StoredAnswer, ...],
        attempt: int,
    ) -> list[Charge]:
        charges = item_charges(self._matrix, self._run, item, result, rows, attempt=attempt)
        for made in charges:
            self._ledger.append(made)
        return charges

    async def _excerpts(
        self, result: ItemResult, rows: tuple[StoredAnswer, ...]
    ) -> tuple[Excerpt, ...]:
        if result.answer is None:
            return ()
        answered = result.answer.done.message_id
        if not rows or rows[-1].message_id != answered:
            raise StoreMismatchError(f"the last stored row is not message {answered}")
        return await rebuild_excerpts(rows[-1].chunk_ids, self._chunks, self._titles)

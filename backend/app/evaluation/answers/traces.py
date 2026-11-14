"""Reading a run's trace file (`answers-trace-v1`, written by `settle.TraceLog`).

Scoring and the run report read the stored rows and the excerpts each settled item
was answered from, and the failed deletes the report lists for cleanup. Reading is
strict, like the journal and the ledger: an unknown schema, kind or key, or a value of
the wrong type, raises `TraceError` naming the line and the field, never the value.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from app.domain.evaluation import IndexedChunk
from app.evaluation.answers.jsonl import read_objects
from app.evaluation.answers.scoring import Excerpt
from app.evaluation.answers.store import StoredAnswer


class TraceError(Exception):
    """The trace file cannot be trusted; the message names lines and fields only."""


@dataclass(frozen=True)
class Trace:
    item_id: str
    attempt: int
    conversation_id: str | None
    rows: tuple[StoredAnswer, ...]
    excerpts: tuple[Excerpt, ...]


@dataclass(frozen=True)
class DeleteFailure:
    """A conversation the harness could not delete; the report lists it for cleanup."""

    item_id: str
    attempt: int
    conversation_id: str | None
    client_id: str
    code: str
    http_status: int | None


@dataclass(frozen=True)
class Traces:
    traces: tuple[Trace, ...]
    delete_failures: tuple[DeleteFailure, ...]

    def latest(self) -> dict[str, Trace]:
        """Each item's trace from its last attempt, the one its final result scores;
        the file is append-only, so file order is attempt order.
        """
        return {trace.item_id: trace for trace in self.traces}


class _Line(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    # settle.TRACE_SCHEMA; the round-trip tests hold writer and reader together.
    schema_: Literal["answers-trace-v1"] = Field(alias="schema")
    item_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    conversation_id: str | None


class _Excerpt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    tag: str
    chunk_id: str
    doc_id: str
    section_path: tuple[str, ...]
    chunk_text: str
    text: str


class _TraceLine(_Line):
    kind: Literal["trace"]
    rows: tuple[StoredAnswer, ...]
    excerpts: tuple[_Excerpt, ...]


class _DeleteLine(_Line):
    kind: Literal["delete_failed"]
    client_id: str
    code: str
    http_status: int | None


_ADAPTER: TypeAdapter[_TraceLine | _DeleteLine] = TypeAdapter(
    Annotated[_TraceLine | _DeleteLine, Field(discriminator="kind")]
)


def _excerpt(raw: _Excerpt) -> Excerpt:
    chunk = IndexedChunk(raw.chunk_id, raw.doc_id, raw.section_path, raw.chunk_text)
    return Excerpt(raw.tag, chunk, raw.text)


def _field(exc: ValidationError) -> str:
    """The first failing field's top-level name (the union tag is not a field)."""
    loc = exc.errors(include_input=False, include_url=False)[0]["loc"]
    names = [str(part) for part in loc if str(part) not in {"trace", "delete_failed"}]
    return names[0] if names else "kind"


def read_traces(path: Path) -> Traces:
    """Every trace and failed delete in file order; a missing file has none."""
    traces: list[Trace] = []
    failures: list[DeleteFailure] = []
    for number, obj in read_objects(path, TraceError):
        try:
            # JSON mode: strict types, with arrays for tuples and values for enums.
            line = _ADAPTER.validate_json(json.dumps(obj))
        except ValidationError as exc:
            raise TraceError(f"line {number}: {_field(exc)} is invalid") from None
        if isinstance(line, _TraceLine):
            excerpts = tuple(_excerpt(raw) for raw in line.excerpts)
            traces.append(
                Trace(line.item_id, line.attempt, line.conversation_id, line.rows, excerpts)
            )
        else:
            failures.append(
                DeleteFailure(
                    line.item_id,
                    line.attempt,
                    line.conversation_id,
                    line.client_id,
                    line.code,
                    line.http_status,
                )
            )
    return Traces(tuple(traces), tuple(failures))

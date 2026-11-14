"""The run's append-only results journal (`results.jsonl`) and `--resume`.

One JSON line per item, written and fsynced before the next item starts, so a crash
loses at most the item in flight. The file is only ever appended; reading it back is
strict (a partial or malformed line stops the run instead of being skipped), and the
`done` part keeps the API's own event shape, so it is checked by the same parser that
read it off the wire. A resumed run asks only the items with no line: an error is that
item's result, and asking again would let a flaky item pick its best attempt.

The journal holds model answers, so it lives in the git-ignored run directory, and
error messages name line numbers and fields only.
"""

import json
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import cast

from app.domain.answer_dataset import EvalItem
from app.evaluation.answers.client import Answered, ItemResult
from app.evaluation.answers.jsonl import Json, append_object, read_objects
from app.evaluation.answers.sse import Done, MalformedEventError, parse_done

JOURNAL_SCHEMA = "answers-journal-v1"


class JournalError(Exception):
    """The journal cannot be trusted as it is; the run stops for a person to look."""


def _done_record(done: Done) -> Json:
    return {
        "conversation_id": done.conversation_id,
        "message_id": done.message_id,
        "outcome": "completed" if done.refusal_reason is None else "refused",
        "refusal": done.refusal_reason is not None,
        "refusal_reason": None if done.refusal_reason is None else done.refusal_reason.value,
        "content": done.content,
        "citations": list(done.citations),
        "flags": {
            "late_refusal_marker": done.flags.late_refusal_marker,
            "uncited": done.flags.uncited,
            "invalid_citations": done.flags.invalid_citations,
            "rewrite_fallback": done.flags.rewrite_fallback,
            "truncated": done.flags.truncated,
        },
    }


def _record(result: ItemResult) -> Json:
    answer = result.answer
    return {
        "schema": JOURNAL_SCHEMA,
        "item_id": result.item_id,
        "client_id": result.client_id,
        "conversation_id": result.conversation_id,
        "retries": result.retries,
        "error": result.error,
        "cause": result.cause,
        "http_status": result.http_status,
        "setup_refused": result.setup_refused,
        "answer": None
        if answer is None
        else {
            "done": _done_record(answer.done),
            "retries": answer.retries,
            "ttft_ms": answer.ttft_ms,
            "total_ms": answer.total_ms,
        },
    }


class Journal:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, result: ItemResult) -> None:
        append_object(self.path, _record(result))


# ---------------------------------------------------------------- reading


class _FieldError(Exception):
    pass


def _get(obj: Json, key: str) -> object:
    if key not in obj:
        raise _FieldError(f"missing field {key}")
    return obj[key]


def _bad(key: str) -> _FieldError:
    return _FieldError(f"field {key} has the wrong type or value")


def _text(obj: Json, key: str, *, optional: bool = False) -> str | None:
    value = _get(obj, key)
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise _bad(key)
    return value


def _count(obj: Json, key: str, *, optional: bool = False) -> int | None:
    value = _get(obj, key)
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _bad(key)
    return value


def _millis(obj: Json, key: str, *, optional: bool = False) -> float | None:
    value = _get(obj, key)
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _bad(key)
    return float(value)


def _answer(raw: object) -> Answered:
    if not isinstance(raw, dict):
        raise _bad("answer")
    obj = cast(Json, raw)
    try:
        done = parse_done(json.dumps(_get(obj, "done")))
    except MalformedEventError as exc:
        raise _FieldError(f"answer.{exc}") from None
    return Answered(
        done=done,
        retries=cast(int, _count(obj, "retries")),
        ttft_ms=_millis(obj, "ttft_ms", optional=True),
        total_ms=cast(float, _millis(obj, "total_ms")),
    )


def _result(obj: Json) -> ItemResult:
    if _get(obj, "schema") != JOURNAL_SCHEMA:
        raise _bad("schema")
    error = _text(obj, "error", optional=True)
    raw_answer = _get(obj, "answer")
    # Exactly one of an error and an answer: an item is either scored or counted against.
    if (error is None) == (raw_answer is None):
        raise _bad("answer")
    return ItemResult(
        item_id=cast(str, _text(obj, "item_id")),
        client_id=cast(str, _text(obj, "client_id")),
        conversation_id=_text(obj, "conversation_id", optional=True),
        retries=cast(int, _count(obj, "retries")),
        answer=None if raw_answer is None else _answer(raw_answer),
        error=error,
        cause=_text(obj, "cause", optional=True),
        http_status=_count(obj, "http_status", optional=True),
        setup_refused=cast(int, _count(obj, "setup_refused")),
    )


def read_journal(path: Path) -> list[ItemResult]:
    """Every recorded result, in file order; a missing file is an empty journal."""
    results = []
    for number, obj in read_objects(path, JournalError):
        try:
            results.append(_result(obj))
        except _FieldError as exc:
            raise JournalError(f"line {number}: {exc}") from None
    return results


# ---------------------------------------------------------------- resume


def pending(items: Sequence[EvalItem], recorded: Sequence[ItemResult]) -> list[EvalItem]:
    """The items with no recorded result, in run order."""
    wanted = {item.id for item in items}
    seen: set[str] = set()
    for result in recorded:
        if result.item_id in seen:
            raise JournalError(f"recorded twice: {result.item_id}")
        if result.item_id not in wanted:
            raise JournalError(f"not in this run: {result.item_id}")
        seen.add(result.item_id)
    return [item for item in items if item.id not in seen]


async def run_pending(
    items: Sequence[EvalItem],
    journal: Journal,
    ask: Callable[[EvalItem, str], Awaitable[ItemResult]],
    new_client_id: Callable[[], str],
) -> list[ItemResult]:
    """Ask the missing items one at a time (latencies stay comparable), each with its
    own client ID, and return every result of the run in item order.
    """
    recorded = read_journal(journal.path)
    by_id = {result.item_id: result for result in recorded}
    for item in pending(items, recorded):
        result = await ask(item, new_client_id())
        journal.append(result)
        by_id[item.id] = result
    return [by_id[item.id] for item in items]

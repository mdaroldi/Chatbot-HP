"""The run's append-only results journal (`results.jsonl`) and `--resume`.

One JSON line per item, written and fsynced before the next item starts, so a crash
loses at most the item in flight. The file is only ever appended; reading it back is
strict (a partial or malformed line stops the run instead of being skipped), and the
`done` part keeps the API's own event shape, so it is checked by the same parser that
read it off the wire. A resumed run asks only the items with no line: an error is that
item's result, and asking again would let a flaky item pick its best attempt. Only
`--retry-errors` (off by default) re-asks items whose last result is an error: the new
line is appended, the last line counts for scoring and every attempt is charged.

The journal holds model answers, so it lives in the git-ignored run directory, and
error messages name line numbers and fields only.
"""

import json
from collections import Counter
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


def latest(recorded: Sequence[ItemResult]) -> dict[str, ItemResult]:
    """Each item's last result. A later line is allowed only after an error (a
    `--retry-errors` re-ask); a line after an answer means the journal is corrupt.
    """
    final: dict[str, ItemResult] = {}
    for result in recorded:
        earlier = final.get(result.item_id)
        if earlier is not None and earlier.ok:
            raise JournalError(f"recorded twice: {result.item_id}")
        final[result.item_id] = result
    return final


def retried_items(recorded: Sequence[ItemResult]) -> tuple[str, ...]:
    """Items asked more than once, in first-seen order, for the report."""
    counts = Counter(result.item_id for result in recorded)
    return tuple(item_id for item_id, count in counts.items() if count > 1)


def pending(
    items: Sequence[EvalItem], recorded: Sequence[ItemResult], *, retry_errors: bool = False
) -> list[EvalItem]:
    """The items to ask, in run order: those with no result and, with `retry_errors`,
    those whose last result is an error.
    """
    wanted = {item.id for item in items}
    unknown = next((r.item_id for r in recorded if r.item_id not in wanted), None)
    if unknown is not None:
        raise JournalError(f"not in this run: {unknown}")
    final = latest(recorded)

    def asked_again(item: EvalItem) -> bool:
        result = final.get(item.id)
        return result is None or (retry_errors and not result.ok)

    return [item for item in items if asked_again(item)]


def spread[T](items: Sequence[T], limit: int) -> list[T]:
    """`limit` evenly spaced items, the first included: a pilot over a run's item order
    (answers-v1, then each slice) samples every kind of item, not only the first kind.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if limit >= len(items):
        return list(items)
    return [items[i * len(items) // limit] for i in range(limit)]


async def run_pending(  # noqa: PLR0913 - two keyword-only selection options
    items: Sequence[EvalItem],
    journal: Journal,
    ask: Callable[[EvalItem, str, int], Awaitable[ItemResult]],
    new_client_id: Callable[[], str],
    *,
    retry_errors: bool = False,
    limit: int | None = None,
) -> list[ItemResult]:
    """Ask the selected items one at a time (latencies stay comparable), each with its
    own client ID and its attempt number (the ledger charges every attempt), and
    return each recorded item's final result in item order. Every attempt stays in the
    journal. `limit` asks only a `spread` of the selected items (the pilot); a later
    run without it asks the rest.
    """
    recorded = read_journal(journal.path)
    attempts = Counter(result.item_id for result in recorded)
    final = latest(recorded)
    todo = pending(items, recorded, retry_errors=retry_errors)
    for item in todo if limit is None else spread(todo, limit):
        result = await ask(item, new_client_id(), attempts[item.id] + 1)
        journal.append(result)
        final[item.id] = result
    return [final[item.id] for item in items if item.id in final]

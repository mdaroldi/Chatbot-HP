"""SSE parsing for the answer harness: frames from the wire, then a strict reader of the
`done` and `error` events (Phase 5 rule 3).

The stream is `token` events, then `sources` and `done`, or one `error` event after
text was shown; `: keepalive` comments fill silences. The reader is tolerant of extra
fields (the Phase 4 rule) and strict on the ones the harness uses: a missing field, a
wrong type or inconsistent refusal fields raise, so no item is scored from a guess.
Messages name fields only, never values, because values are user or model text.
"""

import json
import re
from dataclasses import dataclass, field
from typing import cast

from app.domain.chat import AnswerFlags, RefusalReason

_TAG = re.compile(r"S[1-9][0-9]*")
_CODE = re.compile(r"[a-z][a-z_]*")
_OUTCOMES = ("completed", "refused")
_BOOLEAN_FLAGS = ("late_refusal_marker", "uncited", "truncated")

# A decoded JSON object; values are checked one by one before use.
Json = dict[str, object]


class MalformedEventError(Exception):
    """An event the harness needs is missing a field or has the wrong shape."""


@dataclass(frozen=True)
class Frame:
    event: str
    data: str


@dataclass
class FrameParser:
    """Feeds one line at a time (as httpx's aiter_lines gives them, without line breaks)
    and returns a frame when a blank line ends one that has data.
    """

    _event: str = ""
    _data: list[str] = field(default_factory=list)

    def feed(self, line: str) -> Frame | None:
        if not line:
            return self._dispatch()
        if line.startswith(":"):
            return None
        name, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if name == "event":
            self._event = value
        elif name == "data":
            self._data.append(value)
        return None

    def _dispatch(self) -> Frame | None:
        frame = Frame(self._event or "message", "\n".join(self._data)) if self._data else None
        self._event, self._data = "", []
        return frame


@dataclass(frozen=True)
class Done:
    conversation_id: str
    message_id: str
    content: str
    refusal_reason: RefusalReason | None
    citations: tuple[str, ...]
    flags: AnswerFlags


@dataclass(frozen=True)
class StreamError:
    code: str
    message_id: str | None


def _object(data: str, event: str) -> Json:
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError:
        decoded = None
    if not isinstance(decoded, dict):
        raise MalformedEventError(f"{event}: data is not a JSON object")
    return cast(Json, decoded)


def _get(obj: Json, key: str, event: str) -> object:
    if key not in obj:
        raise MalformedEventError(f"{event}: missing field {key}")
    return obj[key]


def _bad(event: str, key: str) -> MalformedEventError:
    return MalformedEventError(f"{event}: field {key} has the wrong type or value")


def _id(obj: Json, key: str, event: str) -> str:
    value = _get(obj, key, event)
    if not isinstance(value, str) or not value:
        raise _bad(event, key)
    return value


def _bool(obj: Json, key: str, event: str) -> bool:
    value = _get(obj, key, event)
    if not isinstance(value, bool):
        raise _bad(event, key)
    return value


def _optional_code(obj: Json, key: str, event: str) -> str | None:
    value = _get(obj, key, event)
    if value is not None and not (isinstance(value, str) and _CODE.fullmatch(value)):
        raise _bad(event, key)
    return value


def _citations(obj: Json) -> tuple[str, ...]:
    value = _get(obj, "citations", "done")
    if not isinstance(value, list):
        raise _bad("done", "citations")
    tags = cast(list[object], value)
    if not all(isinstance(tag, str) and _TAG.fullmatch(tag) for tag in tags):
        raise _bad("done", "citations")
    return tuple(cast(list[str], tags))


def _flags(obj: Json) -> AnswerFlags:
    value = _get(obj, "flags", "done")
    if not isinstance(value, dict):
        raise _bad("done", "flags")
    flags = cast(Json, value)
    late, uncited, truncated = (_bool(flags, key, "done.flags") for key in _BOOLEAN_FLAGS)
    invalid = _get(flags, "invalid_citations", "done.flags")
    if isinstance(invalid, bool) or not isinstance(invalid, int) or invalid < 0:
        raise _bad("done.flags", "invalid_citations")
    fallback = _optional_code(flags, "rewrite_fallback", "done.flags")
    return AnswerFlags(late, uncited, invalid, fallback, truncated)


def _refusal_reason(obj: Json) -> RefusalReason | None:
    outcome = _get(obj, "outcome", "done")
    if outcome not in _OUTCOMES:
        raise _bad("done", "outcome")
    refusal = _bool(obj, "refusal", "done")
    raw = _get(obj, "refusal_reason", "done")
    if raw is not None and raw not in tuple(RefusalReason):
        raise _bad("done", "refusal_reason")
    reason = None if raw is None else RefusalReason(cast(str, raw))
    if len({refusal, reason is not None, outcome == "refused"}) != 1:
        raise MalformedEventError("done: outcome, refusal and refusal_reason disagree")
    return reason


def parse_done(data: str) -> Done:
    obj = _object(data, "done")
    content = _get(obj, "content", "done")
    if not isinstance(content, str):
        raise _bad("done", "content")
    return Done(
        conversation_id=_id(obj, "conversation_id", "done"),
        message_id=_id(obj, "message_id", "done"),
        content=content,
        refusal_reason=_refusal_reason(obj),
        citations=_citations(obj),
        flags=_flags(obj),
    )


def parse_error(data: str) -> StreamError:
    obj = _object(data, "error")
    code = _get(obj, "code", "error")
    if not isinstance(code, str) or not _CODE.fullmatch(code):
        raise _bad("error", "code")
    message_id = _get(obj, "message_id", "error")
    if message_id is not None and not isinstance(message_id, str):
        raise _bad("error", "message_id")
    return StreamError(code=code, message_id=message_id)

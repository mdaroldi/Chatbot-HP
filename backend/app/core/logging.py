"""Structured JSON logging: one JSON object per line on stdout."""

import json
import logging
import sys
import traceback
from datetime import UTC, datetime
from typing import Any

from app.core.request_context import get_request_id

# Attributes every LogRecord has; anything else on a record came from `extra=`.
# uvicorn's `color_message` duplicates the message with ANSI escapes.
_STANDARD_RECORD_ATTRS = frozenset(
    logging.LogRecord("", logging.INFO, "", 0, "", (), None).__dict__
) | {"message", "asctime", "color_message"}

_CORE_KEYS = ("timestamp", "level", "logger", "message", "request_id")

# Loggers that install their own handlers; they must go through the root JSON handler.
_THIRD_PARTY_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")
# These leak request data when enabled: full URLs with query strings (HTTP clients,
# uvicorn's access log, replaced by app.access) and SQL with bound parameters
# (SQLAlchemy logs whenever its logger is enabled, whatever the engine's echo flag).
_QUIET_LOGGERS = ("httpx", "httpcore", "uvicorn.access", "sqlalchemy")


def _qualified_name(cls: type[BaseException]) -> str:
    module = cls.__module__
    return cls.__qualname__ if module == "builtins" else f"{module}.{cls.__qualname__}"


def serialize_exception(
    exc: BaseException,
    *,
    include_messages: bool,
    ancestors: frozenset[int] = frozenset(),
) -> dict[str, object]:
    """Describe an exception as type plus frame locations, following its whole chain.

    Messages, notes and source lines are left out by default because they routinely
    carry user input (personal data). Frames keep only file, line and function: the
    source line of `raise ValueError(user_text)` would leak too. `ancestors` holds the
    exceptions on the current path, so cyclic chains terminate.
    """
    path = ancestors | {id(exc)}
    data: dict[str, object] = {
        "type": _qualified_name(type(exc)),
        "frames": [
            {"file": frame.filename, "line": frame.lineno, "function": frame.name}
            for frame in traceback.extract_tb(exc.__traceback__)
        ],
    }
    if include_messages:
        data["message"] = str(exc)
        notes = getattr(exc, "__notes__", None)
        if notes:
            data["notes"] = [str(note) for note in notes]

    def nested(other: BaseException) -> dict[str, object]:
        return serialize_exception(other, include_messages=include_messages, ancestors=path)

    if exc.__cause__ is not None and id(exc.__cause__) not in path:
        data["cause"] = nested(exc.__cause__)
    elif (
        not exc.__suppress_context__
        and exc.__context__ is not None
        and id(exc.__context__) not in path
    ):
        data["context"] = nested(exc.__context__)
    if isinstance(exc, BaseExceptionGroup):
        data["exceptions"] = [nested(member) for member in exc.exceptions]
    return data


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON so log pipelines can parse them reliably.

    json.dumps escapes newlines and quotes, so messages can never split a record over
    several lines. Extra fields never override the core keys. Exceptions are
    serialized without their messages unless explicitly enabled (see Settings).
    """

    def __init__(self, *, include_exception_messages: bool = False) -> None:
        super().__init__()
        self.include_exception_messages = include_exception_messages

    def format(self, record: logging.LogRecord) -> str:
        # Any: values are arbitrary user-supplied extras, serialized with default=str.
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key not in _CORE_KEYS:
                payload[key] = value
        if record.exc_info and record.exc_info[1] is not None:
            exc = record.exc_info[1]
            payload["exception_type"] = _qualified_name(type(exc))
            payload["exception"] = serialize_exception(
                exc, include_messages=self.include_exception_messages
            )
        return json.dumps(payload, default=str)


def configure_logging(level: str, *, include_exception_messages: bool = False) -> None:
    """Install a single stdout JSON handler on the root logger (idempotent).

    Exception messages stay out of logs unless `include_exception_messages` is set.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(include_exception_messages=include_exception_messages))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in _THIRD_PARTY_LOGGERS:
        third_party = logging.getLogger(name)
        third_party.handlers.clear()
        third_party.propagate = True
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

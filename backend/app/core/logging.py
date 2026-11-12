"""Structured JSON logging: one JSON object per line on stdout."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from app.core.request_context import get_request_id

# Attributes every LogRecord has; anything else on a record came from `extra=`.
_STANDARD_RECORD_ATTRS = frozenset(
    logging.LogRecord("", logging.INFO, "", 0, "", (), None).__dict__
) | {"message", "asctime"}

_CORE_KEYS = ("timestamp", "level", "logger", "message", "request_id")

# Loggers that install their own handlers; they must go through the root JSON handler.
_THIRD_PARTY_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")
# HTTP clients log full request URLs (query strings included) at INFO.
_QUIET_LOGGERS = ("httpx", "httpcore")


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON so log pipelines can parse them reliably.

    json.dumps escapes newlines and quotes, so messages and tracebacks can never
    split a record over several lines. Extra fields never override the core keys.
    """

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
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception_type"] = record.exc_info[0].__name__
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str) -> None:
    """Install a single stdout JSON handler on the root logger (idempotent)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in _THIRD_PARTY_LOGGERS:
        third_party = logging.getLogger(name)
        third_party.handlers.clear()
        third_party.propagate = True
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

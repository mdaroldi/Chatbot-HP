"""Behavior of the one-line JSON log formatter and logging setup."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

import pytest

from app.core.logging import JsonFormatter, configure_logging
from app.core.request_context import bound_request_id

REQUIRED_KEYS = {"timestamp", "level", "logger", "message", "request_id"}


def make_record(
    msg: str,
    *args: object,
    exc_info: Any = None,  # Any: mirrors logging's own loosely typed exc_info
    **extra: object,
) -> logging.LogRecord:
    record = logging.LogRecord("app.test", logging.INFO, __file__, 10, msg, args, exc_info)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def render(record: logging.LogRecord) -> dict[str, Any]:
    line = JsonFormatter().format(record)
    assert "\n" not in line
    assert "\r" not in line
    parsed: dict[str, Any] = json.loads(line)
    return parsed


def test_output_is_single_line_json_with_required_keys() -> None:
    parsed = render(make_record("hello %s", "world"))

    assert parsed.keys() >= REQUIRED_KEYS
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "app.test"
    assert parsed["message"] == "hello world"


def test_timestamp_is_iso8601_in_utc() -> None:
    parsed = render(make_record("tick"))

    timestamp = datetime.fromisoformat(parsed["timestamp"])
    assert timestamp.utcoffset() == UTC.utcoffset(None)


def test_request_id_is_null_outside_a_request() -> None:
    assert render(make_record("idle"))["request_id"] is None


def test_request_id_comes_from_the_current_request_context() -> None:
    with bound_request_id("req-12345678"):
        parsed = render(make_record("inside"))

    assert parsed["request_id"] == "req-12345678"


def test_newlines_and_quotes_in_message_stay_on_one_line() -> None:
    message = 'line one\nline two\r\n"quoted" and \\backslash'

    parsed = render(make_record(message))

    assert parsed["message"] == message


def fail_with_multiline_error() -> None:
    raise ValueError("boom\nwith newline")


def test_exception_is_serialized_on_the_same_line() -> None:
    try:
        fail_with_multiline_error()
    except ValueError:
        record = make_record("failed", exc_info=sys.exc_info())

    parsed = render(record)

    assert parsed["exception_type"] == "ValueError"
    assert "Traceback (most recent call last)" in parsed["exception"]
    assert "ValueError: boom" in parsed["exception"]


def test_extra_fields_are_included() -> None:
    parsed = render(make_record("done", method="GET", status=200, duration_ms=1.5))

    assert parsed["method"] == "GET"
    assert parsed["status"] == 200
    assert parsed["duration_ms"] == 1.5


def test_non_serializable_extra_falls_back_to_string() -> None:
    parsed = render(make_record("odd", when=datetime(2026, 1, 2, tzinfo=UTC)))

    assert parsed["when"] == "2026-01-02 00:00:00+00:00"


def test_uvicorn_color_message_is_dropped() -> None:
    parsed = render(make_record("Started", color_message="\x1b[36mStarted\x1b[0m"))

    assert "color_message" not in parsed


def test_extra_fields_cannot_overwrite_core_keys() -> None:
    parsed = render(make_record("real", level="FAKE", timestamp="never"))

    assert parsed["level"] == "INFO"
    assert parsed["timestamp"] != "never"


def test_configure_logging_installs_one_json_handler_at_level(
    restore_logging: logging.Logger,
) -> None:
    configure_logging("DEBUG")
    configure_logging("WARNING")

    assert restore_logging.level == logging.WARNING
    assert len(restore_logging.handlers) == 1
    assert isinstance(restore_logging.handlers[0].formatter, JsonFormatter)


def test_configure_logging_routes_uvicorn_loggers_through_root() -> None:
    uvicorn_logger = logging.getLogger("uvicorn.error")
    uvicorn_logger.addHandler(logging.NullHandler())

    configure_logging("INFO")

    assert uvicorn_logger.handlers == []
    assert uvicorn_logger.propagate is True


@pytest.mark.parametrize("name", ["httpx", "httpcore", "uvicorn.access"])
def test_configure_logging_silences_loggers_that_log_full_urls(name: str) -> None:
    """These log full URLs with query strings at INFO; our access log replaces them."""
    configure_logging("DEBUG")

    assert logging.getLogger(name).level == logging.WARNING


def test_configure_logging_writes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")

    logging.getLogger("app.test").info("to stdout")

    line = capsys.readouterr().out.strip()
    assert json.loads(line)["message"] == "to stdout"

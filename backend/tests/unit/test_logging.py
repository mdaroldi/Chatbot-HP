"""Behavior of the one-line JSON log formatter and logging setup."""

import json
import logging
import sys
from collections.abc import Callable
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


def render(record: logging.LogRecord, *, include_messages: bool = False) -> dict[str, Any]:
    line = JsonFormatter(include_exception_messages=include_messages).format(record)
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


USER_TEXT = "secret user text"


class CustomError(Exception):
    pass


def raise_secret() -> None:
    raise ValueError(USER_TEXT)


def capture(fn: Callable[[], None]) -> logging.LogRecord:
    try:
        fn()
    except BaseException:
        return make_record("failed", exc_info=sys.exc_info())
    raise AssertionError("expected an exception")


def test_exception_is_serialized_on_the_same_line() -> None:
    def multiline() -> None:
        raise ValueError("boom\nwith newline")

    parsed = render(capture(multiline), include_messages=True)

    assert parsed["exception"]["message"] == "boom\nwith newline"


def test_exception_logs_type_and_frames_but_not_message_by_default() -> None:
    record = capture(raise_secret)
    line = JsonFormatter().format(record)
    parsed = json.loads(line)

    assert USER_TEXT not in line
    assert parsed["exception_type"] == "ValueError"
    exception = parsed["exception"]
    assert exception["type"] == "ValueError"
    assert "message" not in exception
    assert [frame["function"] for frame in exception["frames"]][-2:] == [
        "capture",
        "raise_secret",
    ]


def test_frames_carry_location_only_never_source_lines() -> None:
    """The source line of `raise ValueError(USER_TEXT)` would leak a literal."""
    frames = render(capture(raise_secret))["exception"]["frames"]

    assert all(frame.keys() == {"file", "line", "function"} for frame in frames)
    assert frames[-1]["file"].endswith("test_logging.py")
    assert isinstance(frames[-1]["line"], int)


def test_explicit_cause_is_included_and_redacted() -> None:
    def chained() -> None:
        try:
            raise_secret()
        except ValueError as exc:
            raise RuntimeError("outer secret user text") from exc

    record = capture(chained)
    line = JsonFormatter().format(record)
    exception = json.loads(line)["exception"]

    assert "secret user text" not in line
    assert exception["type"] == "RuntimeError"
    assert exception["cause"]["type"] == "ValueError"
    assert exception["cause"]["frames"][-1]["function"] == "raise_secret"
    assert "context" not in exception


def test_implicit_context_is_included_and_redacted() -> None:
    def during_handling() -> None:
        try:
            raise_secret()
        except ValueError:
            raise KeyError("context secret user text")  # noqa: B904 - implicit context on purpose

    line = JsonFormatter().format(capture(during_handling))
    exception = json.loads(line)["exception"]

    assert "secret user text" not in line
    assert exception["type"] == "KeyError"
    assert exception["context"]["type"] == "ValueError"


def test_suppressed_context_is_omitted() -> None:
    def suppressed() -> None:
        try:
            raise_secret()
        except ValueError:
            raise KeyError("x") from None

    exception = render(capture(suppressed))["exception"]

    assert exception["type"] == "KeyError"
    assert "context" not in exception
    assert "cause" not in exception


def test_exception_group_members_are_included_and_redacted() -> None:
    def group() -> None:
        try:
            raise_secret()
        except ValueError as inner:
            raise ExceptionGroup("group secret user text", [inner, CustomError(USER_TEXT)])  # noqa: B904

    line = JsonFormatter().format(capture(group))
    exception = json.loads(line)["exception"]

    assert "secret user text" not in line
    assert exception["type"] == "ExceptionGroup"
    assert [member["type"] for member in exception["exceptions"]] == [
        "ValueError",
        "tests.unit.test_logging.CustomError",
    ]


def test_exception_notes_are_redacted_by_default() -> None:
    def noted() -> None:
        error = ValueError("x")
        error.add_note(USER_TEXT)
        raise error

    line = JsonFormatter().format(capture(noted))

    assert USER_TEXT not in line


def test_cyclic_exception_chain_terminates() -> None:
    def cyclic() -> None:
        first, second = ValueError("a"), KeyError("b")
        first.__context__ = second
        second.__context__ = first
        raise first

    exception = render(capture(cyclic))["exception"]

    assert exception["type"] == "ValueError"
    assert exception["context"]["type"] == "KeyError"
    assert "context" not in exception["context"]


def test_flag_restores_messages_and_notes_across_the_chain() -> None:
    def chained() -> None:
        error = ValueError(USER_TEXT)
        error.add_note("a note")
        try:
            raise error
        except ValueError as exc:
            raise RuntimeError("outer") from exc

    exception = render(capture(chained), include_messages=True)["exception"]

    assert exception["message"] == "outer"
    assert exception["cause"]["message"] == USER_TEXT
    assert exception["cause"]["notes"] == ["a note"]


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


@pytest.mark.parametrize("include", [False, True])
def test_configure_logging_passes_exception_message_flag_to_formatter(
    restore_logging: logging.Logger, include: bool
) -> None:
    configure_logging("INFO", include_exception_messages=include)

    formatter = restore_logging.handlers[0].formatter
    assert isinstance(formatter, JsonFormatter)
    assert formatter.include_exception_messages is include


def test_configure_logging_writes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")

    logging.getLogger("app.test").info("to stdout")

    line = capsys.readouterr().out.strip()
    assert json.loads(line)["message"] == "to stdout"

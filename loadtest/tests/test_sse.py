"""The SSE reader follows the backend's contract (ADR 0010): `event:` then one `data:`
line of JSON, a blank line ends the event, `:` lines are keepalives."""

import pytest

from hpdocs_load.sse import (
    SseEvent,
    SseProtocolError,
    parse_events,
    summarize_stream,
    timed_events,
)


def test_parses_events_in_order_and_skips_keepalives() -> None:
    lines = [
        "event: token",
        'data: {"text": "Hello"}',
        "",
        ": keepalive",
        "event: done",
        'data: {"outcome": "completed"}',
        "",
    ]

    assert list(parse_events(lines)) == [
        SseEvent("token", {"text": "Hello"}),
        SseEvent("done", {"outcome": "completed"}),
    ]


def test_repeated_blank_lines_between_events_are_ignored() -> None:
    lines = ["", "event: token", 'data: {"text": "a"}', "", "", ""]

    assert list(parse_events(lines)) == [SseEvent("token", {"text": "a"})]


def test_an_empty_stream_has_no_events() -> None:
    assert list(parse_events([])) == []


@pytest.mark.parametrize(
    "lines",
    [
        pytest.param(['data: {"text": "a"}', ""], id="data-without-event"),
        pytest.param(["event: token", ""], id="event-without-data"),
        pytest.param(["event: token", "data: not json", ""], id="invalid-json"),
        pytest.param(["event: token", "data: [1, 2]", ""], id="data-not-an-object"),
        pytest.param(["event: token", 'data: {"a": 1}', 'data: {"b": 2}', ""], id="two-data"),
        pytest.param(["event: token", "event: done", ""], id="two-event-lines"),
        pytest.param(["event: token", 'data: {"text": "a"}'], id="unterminated"),
        pytest.param(["event: token"], id="unterminated-without-data"),
        pytest.param(["retry: 10", ""], id="unknown-field"),
    ],
)
def test_malformed_streams_raise(lines: list[str]) -> None:
    with pytest.raises(SseProtocolError):
        list(parse_events(lines))


def _timed(*pairs: tuple[float, str, dict[str, object]]) -> list[tuple[float, SseEvent]]:
    return [(t, SseEvent(name, data)) for t, name, data in pairs]


def test_summary_of_a_completed_answer() -> None:
    events = _timed(
        (10.4, "token", {"text": "a"}),
        (10.6, "token", {"text": "b"}),
        (10.7, "sources", {"sources": []}),
        (10.9, "done", {"outcome": "completed"}),
    )

    result = summarize_stream(10.0, events)

    assert result.outcome == "completed"
    assert result.error_code is None
    assert result.ttft_ms == pytest.approx(400.0)
    assert result.total_ms == pytest.approx(900.0)
    assert result.tokens == 2
    assert result.reached_done


def test_a_refusal_reaches_done_and_keeps_its_outcome() -> None:
    events = _timed((1.5, "token", {"text": "x"}), (2.0, "done", {"outcome": "refused"}))

    result = summarize_stream(1.0, events)

    assert result.reached_done
    assert result.outcome == "refused"


def test_a_mid_stream_error_keeps_its_code_and_never_reaches_done() -> None:
    events = _timed((2.0, "token", {"text": "x"}), (3.0, "error", {"code": "llm_timeout"}))

    result = summarize_stream(1.0, events)

    assert not result.reached_done
    assert result.outcome is None
    assert result.error_code == "llm_timeout"
    assert result.ttft_ms == pytest.approx(1000.0)
    assert result.total_ms == pytest.approx(2000.0)


def test_a_stream_that_ends_without_done_or_error_is_a_protocol_error() -> None:
    result = summarize_stream(0.0, _timed((0.5, "token", {"text": "x"})))

    assert not result.reached_done
    assert result.error_code == "no_done"
    assert result.total_ms == pytest.approx(500.0)


def test_an_empty_stream_is_a_protocol_error_with_no_ttft() -> None:
    result = summarize_stream(0.0, [])

    assert result.error_code == "no_done"
    assert result.ttft_ms is None
    assert result.total_ms is None
    assert result.tokens == 0


def test_events_after_done_are_a_protocol_error() -> None:
    events = _timed(
        (1.0, "token", {"text": "x"}),
        (2.0, "done", {"outcome": "completed"}),
        (3.0, "token", {"text": "y"}),
    )

    with pytest.raises(SseProtocolError):
        summarize_stream(0.0, events)


@pytest.mark.parametrize(
    ("name", "data"),
    [
        pytest.param("done", {"outcome": 1}, id="done-outcome-not-a-string"),
        pytest.param("done", {}, id="done-without-outcome"),
        pytest.param("error", {"code": None}, id="error-code-not-a-string"),
        pytest.param("surprise", {}, id="unknown-event"),
    ],
)
def test_malformed_events_raise(name: str, data: dict[str, object]) -> None:
    with pytest.raises(SseProtocolError):
        summarize_stream(0.0, _timed((1.0, name, data)))


def test_timed_events_carry_the_time_of_the_line_that_ends_them() -> None:
    lines = [
        (1.0, "event: token"),
        (1.0, 'data: {"text": "a"}'),
        (1.2, ""),
        (1.5, ": keepalive"),
        (2.0, "event: done"),
        (2.1, 'data: {"outcome": "completed"}'),
        (2.1, ""),
    ]

    assert list(timed_events(lines)) == [
        (1.2, SseEvent("token", {"text": "a"})),
        (2.1, SseEvent("done", {"outcome": "completed"})),
    ]


def test_timed_events_raise_on_a_broken_stream() -> None:
    with pytest.raises(SseProtocolError):
        list(timed_events([(1.0, "event: token"), (1.1, "")]))

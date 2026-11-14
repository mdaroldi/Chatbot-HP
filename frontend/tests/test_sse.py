"""The SSE parser turns the answer stream into events, whatever the chunking.

Framing follows the WHATWG event-stream rules the backend writes to
(`event: <name>\\ndata: <json>\\n\\n`, `: keepalive\\n\\n`), with one deliberate
difference: a last frame missing its blank line is delivered at close(), not dropped.
JSON is not decoded here; the API client does that.
"""

from collections.abc import Iterator

import pytest

from hpdocs_ui.sse import (
    MAX_PENDING_FRAME_BYTES,
    SseEvent,
    SseFrameTooLargeError,
    SseParser,
    iter_events,
)

# What the backend's frame() and KEEPALIVE produce (backend/app/api/messages.py).
BACKEND_STREAM = (
    'event: token\ndata: {"text": "Open the "}\n\n'
    ": keepalive\n\n"
    'event: token\ndata: {"text": "ink door \u2013 é"}\n\n'
    'event: sources\ndata: {"sources": []}\n\n'
    'event: done\ndata: {"outcome": "completed"}\n\n'
)
BACKEND_EVENTS = [
    SseEvent("token", '{"text": "Open the "}'),
    SseEvent("token", '{"text": "ink door \u2013 é"}'),
    SseEvent("sources", '{"sources": []}'),
    SseEvent("done", '{"outcome": "completed"}'),
]


def parse_all(*chunks: str) -> list[SseEvent]:
    parser = SseParser()
    events: list[SseEvent] = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.close())
    return events


def test_the_backend_stream_in_one_chunk_gives_its_events_in_order() -> None:
    assert SseParser().feed(BACKEND_STREAM) == BACKEND_EVENTS


@pytest.mark.parametrize("cut", range(1, len(BACKEND_STREAM)))
def test_any_split_into_two_chunks_gives_the_same_events(cut: int) -> None:
    assert parse_all(BACKEND_STREAM[:cut], BACKEND_STREAM[cut:]) == BACKEND_EVENTS


def test_one_character_at_a_time_with_empty_chunks_gives_the_same_events() -> None:
    chunks = [piece for char in BACKEND_STREAM for piece in (char, "")]

    assert parse_all(*chunks) == BACKEND_EVENTS


@pytest.mark.parametrize("newline", ["\r\n", "\r"], ids=["crlf", "cr"])
def test_crlf_and_lone_cr_end_lines_like_lf(newline: str) -> None:
    stream = BACKEND_STREAM.replace("\n", newline)

    assert parse_all(stream) == BACKEND_EVENTS


@pytest.mark.parametrize("cut", range(1, len(BACKEND_STREAM.replace("\n", "\r\n"))))
def test_a_crlf_split_between_chunks_is_one_line_end(cut: int) -> None:
    stream = BACKEND_STREAM.replace("\n", "\r\n")

    assert parse_all(stream[:cut], stream[cut:]) == BACKEND_EVENTS


def test_a_cr_at_the_end_of_a_chunk_waits_for_the_next_one() -> None:
    """Only the next character says whether a final CR is a blank line or half a CRLF."""
    parser = SseParser()

    assert parser.feed("data: a\r") == []
    assert parser.feed("\n\r") == []
    assert parser.feed("\ndata: b\r\r") == [SseEvent("message", "a")]
    assert parser.feed("x") == [SseEvent("message", "b")]


def test_a_comment_inside_a_frame_is_ignored() -> None:
    stream = 'event: token\n: keepalive\ndata: {"text": "x"}\n\n'

    assert SseParser().feed(stream) == [SseEvent("token", '{"text": "x"}')]


def test_multi_line_data_is_joined_with_newlines() -> None:
    assert SseParser().feed("event: token\ndata: one\ndata:\ndata: three\n\n") == [
        SseEvent("token", "one\n\nthree")
    ]


def test_the_event_name_defaults_to_message_and_resets_after_each_frame() -> None:
    assert SseParser().feed("data: a\n\nevent: token\ndata: b\n\ndata: c\n\n") == [
        SseEvent("message", "a"),
        SseEvent("token", "b"),
        SseEvent("message", "c"),
    ]


@pytest.mark.parametrize(
    ("line", "data"),
    [
        ("data:x", "x"),
        ("data: x", "x"),
        ("data:  x", " x"),
        ("data: a: b", "a: b"),
        ("data", ""),
        ("data:", ""),
    ],
)
def test_one_leading_space_is_dropped_from_a_value(line: str, data: str) -> None:
    assert SseParser().feed(f"{line}\n\n") == [SseEvent("message", data)]


def test_a_frame_without_data_is_not_delivered_and_its_name_does_not_leak() -> None:
    assert SseParser().feed("event: token\n\ndata: x\n\n") == [SseEvent("message", "x")]


def test_unknown_fields_and_wrongly_cased_names_are_ignored() -> None:
    stream = "id: 7\nretry: 10\nfoo: bar\nEvent: token\nDATA: no\nevent: done\ndata: x\n\n"

    assert SseParser().feed(stream) == [SseEvent("done", "x")]


def test_unknown_event_names_are_passed_up_for_the_client_to_ignore() -> None:
    assert SseParser().feed("event: future\ndata: {}\n\n") == [SseEvent("future", "{}")]


def test_a_frame_is_held_until_its_blank_line() -> None:
    parser = SseParser()

    assert parser.feed("event: done\ndata: {}\n") == []
    assert parser.feed("\n") == [SseEvent("done", "{}")]
    assert parser.close() == []


@pytest.mark.parametrize(
    "tail",
    ["event: done\ndata: {}\n", "event: done\ndata: {}", "event: done\ndata: {}\r"],
    ids=["no-blank-line", "no-line-end", "trailing-cr"],
)
def test_close_delivers_a_last_frame_missing_its_blank_line(tail: str) -> None:
    parser = SseParser()

    assert parser.feed('event: token\ndata: {"text": "x"}\n\n' + tail) == [
        SseEvent("token", '{"text": "x"}')
    ]
    assert parser.close() == [SseEvent("done", "{}")]


def test_close_drops_a_trailing_frame_with_no_data() -> None:
    parser = SseParser()

    assert parser.feed("data: x\n\nevent: done\n") == [SseEvent("message", "x")]
    assert parser.close() == []


def test_iter_events_yields_each_event_as_its_chunk_arrives_and_flushes_at_the_end() -> None:
    seen: list[str] = []

    def source() -> Iterator[str]:
        for chunk in ("data: a\n\nda", "ta: b\n\ndata: c"):
            seen.append(chunk)
            yield chunk

    events = iter_events(source())

    assert next(events) == SseEvent("message", "a")
    assert seen == ["data: a\n\nda"]
    assert list(events) == [SseEvent("message", "b"), SseEvent("message", "c")]
    assert seen == ["data: a\n\nda", "ta: b\n\ndata: c"]


def test_the_pending_frame_cap_is_one_mebibyte() -> None:
    assert MAX_PENDING_FRAME_BYTES == 1024 * 1024


def frame_of(size: int) -> str:
    """A data frame whose text before the blank line is exactly `size` UTF-8 bytes."""
    prefix = "data: "
    filler = "é" * ((size - len(prefix) - 1) // 2)
    filler += "x" * (size - len(prefix) - 1 - 2 * len(filler))
    text = prefix + filler + "\n"
    assert len(text.encode()) == size
    return text


def test_a_frame_at_the_cap_is_delivered() -> None:
    frame = frame_of(MAX_PENDING_FRAME_BYTES)

    events = SseParser().feed(frame + "\n")

    assert [len(event.data.encode()) for event in events] == [MAX_PENDING_FRAME_BYTES - 7]


def test_one_byte_over_the_cap_raises_before_the_blank_line() -> None:
    parser = SseParser()

    with pytest.raises(SseFrameTooLargeError):
        parser.feed(frame_of(MAX_PENDING_FRAME_BYTES + 1))


def test_one_unterminated_line_over_the_cap_raises_while_it_arrives() -> None:
    parser = SseParser()
    body = "x" * (MAX_PENDING_FRAME_BYTES - len("data: "))

    assert parser.feed("data: ") == []
    for start in range(0, len(body), 65536):
        assert parser.feed(body[start : start + 65536]) == []
    with pytest.raises(SseFrameTooLargeError):
        parser.feed("x")


def test_many_small_frames_beyond_the_cap_in_total_never_trip_it() -> None:
    parser = SseParser()
    frame = 'event: token\ndata: {"text": "' + "x" * 1000 + '"}\n\n'
    count = 3 * MAX_PENDING_FRAME_BYTES // len(frame)

    events = [event for _ in range(count) for event in parser.feed(frame)]

    assert len(events) == count

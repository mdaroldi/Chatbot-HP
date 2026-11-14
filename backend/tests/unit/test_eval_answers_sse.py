"""SSE parsing for the answer harness: frames from real wire bytes, and a strict reader
of the `done` and `error` events that fails loudly on a missing or mistyped field.

Two sources of bytes: hand-written frames, and the backend's own `frame()` encoder (the
producer the harness must read), so the parser is not checked only against itself.
"""

import json

import pytest

from app.api.messages import KEEPALIVE, frame
from app.domain.chat import AnswerFlags, RefusalReason
from app.evaluation.answers.sse import (
    Done,
    Frame,
    FrameParser,
    MalformedEventError,
    StreamError,
    parse_done,
    parse_error,
)


def lines(raw: bytes) -> list[str]:
    """How httpx's aiter_lines splits a body: on line breaks, without them."""
    return raw.decode().splitlines()


def parse_all(raw: bytes) -> list[Frame]:
    parser = FrameParser()
    frames = [parser.feed(line) for line in lines(raw)]
    return [f for f in frames if f is not None]


DONE: dict[str, object] = {
    "conversation_id": "c-1",
    "user_message_id": "u-1",
    "message_id": "m-1",
    "outcome": "completed",
    "refusal": False,
    "refusal_reason": None,
    "content": "Clear the tray [S1].",
    "citations": ["S1"],
    "flags": {
        "late_refusal_marker": False,
        "uncited": False,
        "invalid_citations": 0,
        "rewrite_fallback": None,
        "truncated": False,
    },
}


# ------------------------------------------------------------------ frames


def test_hand_written_frames_are_split_on_blank_lines() -> None:
    raw = b'event: token\ndata: {"text": "Hi"}\n\nevent: done\ndata: {}\n\n'

    assert parse_all(raw) == [Frame("token", '{"text": "Hi"}'), Frame("done", "{}")]


def test_the_backend_encoder_round_trips_including_newlines_in_the_text() -> None:
    raw = frame("token", {"text": "line one\nline two"}) + frame("done", DONE)

    frames = parse_all(raw)

    assert [f.event for f in frames] == ["token", "done"]
    assert json.loads(frames[0].data) == {"text": "line one\nline two"}
    assert json.loads(frames[1].data) == DONE


def test_keepalive_comments_are_ignored() -> None:
    raw = KEEPALIVE + frame("token", {"text": "a"}) + KEEPALIVE + KEEPALIVE

    assert parse_all(raw) == [Frame("token", '{"text": "a"}')]


def test_multi_line_data_is_joined_with_newlines() -> None:
    raw = b"event: note\ndata: one\ndata: two\n\n"

    assert parse_all(raw) == [Frame("note", "one\ntwo")]


def test_a_frame_without_an_event_name_is_a_message_event() -> None:
    assert parse_all(b"data: x\n\n") == [Frame("message", "x")]


def test_a_field_without_a_space_after_the_colon_is_read_the_same() -> None:
    assert parse_all(b"event:done\ndata:{}\n\n") == [Frame("done", "{}")]


def test_a_blank_line_without_data_dispatches_nothing() -> None:
    assert parse_all(b"\n\nevent: token\n\n") == []


def test_unknown_fields_are_ignored() -> None:
    assert parse_all(b"id: 7\nretry: 10\nevent: t\ndata: d\n\n") == [Frame("t", "d")]


def test_an_unfinished_frame_is_not_dispatched() -> None:
    assert parse_all(b'event: done\ndata: {"a": 1}') == []


# ------------------------------------------------------------------ done


def test_a_completed_answer_is_read_with_its_flags() -> None:
    done = parse_done(json.dumps(DONE))

    assert done == Done(
        conversation_id="c-1",
        message_id="m-1",
        content="Clear the tray [S1].",
        refusal_reason=None,
        citations=("S1",),
        flags=AnswerFlags(),
    )


def test_a_refusal_is_read_with_its_reason() -> None:
    data = {**DONE, "outcome": "refused", "refusal": True, "refusal_reason": "below_threshold"}
    data["citations"] = []

    done = parse_done(json.dumps(data))

    assert done.refusal_reason is RefusalReason.BELOW_THRESHOLD
    assert done.citations == ()


def test_flags_are_read_from_the_done_event() -> None:
    flags = {
        "late_refusal_marker": True,
        "uncited": True,
        "invalid_citations": 2,
        "rewrite_fallback": "timeout",
        "truncated": True,
    }

    done = parse_done(json.dumps({**DONE, "flags": flags}))

    assert done.flags == AnswerFlags(True, True, 2, "timeout", True)


def test_extra_fields_are_ignored_by_the_tolerant_reader() -> None:
    data = {**DONE, "new_field": 1, "flags": {**DONE["flags"], "new_flag": True}}  # type: ignore[dict-item]

    assert parse_done(json.dumps(data)).message_id == "m-1"


@pytest.mark.parametrize(
    "field",
    [
        "conversation_id",
        "message_id",
        "outcome",
        "refusal",
        "refusal_reason",
        "content",
        "citations",
        "flags",
    ],
)
def test_a_missing_field_fails_loudly_naming_it(field: str) -> None:
    data = {k: v for k, v in DONE.items() if k != field}

    with pytest.raises(MalformedEventError, match=field):
        parse_done(json.dumps(data))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("message_id", 7),
        ("message_id", ""),
        ("conversation_id", None),
        ("content", None),
        ("refusal", "no"),
        ("refusal", 0),
        ("citations", "S1"),
        ("citations", [1]),
        ("citations", ["X1"]),
        ("refusal_reason", "tired"),
        ("outcome", "failed"),
        ("flags", []),
    ],
)
def test_a_mistyped_field_fails_loudly_naming_it(field: str, value: object) -> None:
    with pytest.raises(MalformedEventError, match=field):
        parse_done(json.dumps({**DONE, field: value}))


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("uncited", None),
        ("uncited", 1),
        ("late_refusal_marker", "false"),
        ("truncated", None),
        ("invalid_citations", -1),
        ("invalid_citations", True),
        ("invalid_citations", 1.5),
        ("rewrite_fallback", 3),
        ("rewrite_fallback", "Not A Code"),
    ],
)
def test_a_mistyped_flag_fails_loudly_naming_it(flag: str, value: object) -> None:
    flags = {**DONE["flags"], flag: value}  # type: ignore[dict-item]

    with pytest.raises(MalformedEventError, match=flag):
        parse_done(json.dumps({**DONE, "flags": flags}))


def test_a_missing_flag_fails_loudly_naming_it() -> None:
    flags = {k: v for k, v in DONE["flags"].items() if k != "uncited"}  # type: ignore[attr-defined]

    with pytest.raises(MalformedEventError, match="uncited"):
        parse_done(json.dumps({**DONE, "flags": flags}))


@pytest.mark.parametrize(
    "changes",
    [
        {"refusal": True},  # a refusal flag without a reason
        {"refusal_reason": "model_refused"},  # a reason on a completed answer
        {"outcome": "refused"},  # a refused outcome without the refusal fields
    ],
)
def test_inconsistent_refusal_fields_fail_loudly(changes: dict[str, object]) -> None:
    with pytest.raises(MalformedEventError, match="refusal"):
        parse_done(json.dumps({**DONE, **changes}))


@pytest.mark.parametrize("data", ["not json", "[]", '"text"', "null"])
def test_a_done_that_is_not_a_json_object_fails_loudly(data: str) -> None:
    with pytest.raises(MalformedEventError, match="object"):
        parse_done(data)


def test_a_malformed_done_message_never_quotes_the_answer() -> None:
    private_text = "the user's private words"

    with pytest.raises(MalformedEventError) as caught:
        parse_done(json.dumps({**DONE, "content": private_text, "citations": private_text}))

    assert private_text not in str(caught.value)


# ------------------------------------------------------------------ error


def test_an_error_event_gives_its_code_and_message_id() -> None:
    data = {
        "code": "llm_error",
        "message": "The model failed.",
        "request_id": "r",
        "message_id": "m-9",
    }

    assert parse_error(json.dumps(data)) == StreamError(code="llm_error", message_id="m-9")


def test_an_error_event_may_have_no_message_id() -> None:
    data = {"code": "internal_error", "message": "x", "request_id": "r", "message_id": None}

    assert parse_error(json.dumps(data)) == StreamError(code="internal_error", message_id=None)


@pytest.mark.parametrize(
    "data",
    [
        {"message_id": "m"},
        {"code": 5, "message_id": "m"},
        {"code": "Bad Code", "message_id": "m"},
        {"code": "llm_error"},
        {"code": "llm_error", "message_id": 3},
    ],
)
def test_a_malformed_error_event_fails_loudly(data: dict[str, object]) -> None:
    with pytest.raises(MalformedEventError):
        parse_error(json.dumps(data))


def test_an_error_event_that_is_not_an_object_fails_loudly() -> None:
    with pytest.raises(MalformedEventError, match="object"):
        parse_error("[1]")

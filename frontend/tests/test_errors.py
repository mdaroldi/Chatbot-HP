"""API errors carry codes only; the UI text comes from a fixed table, never from a
response body.
"""

import pytest

from hpdocs_ui.errors import (
    BACKEND_UNREACHABLE,
    BAD_RESPONSE,
    GENERIC_ERROR_TEXT,
    STREAM_TIMEOUT,
    ApiError,
    error_text,
    parse_retry_after,
    shown_request_id,
)

WORDING = {
    BACKEND_UNREACHABLE: "The chat service can't be reached right now.",
    "conversation_busy": (
        "An answer for this conversation is still being written. "
        "Wait for it to finish, then refresh."
    ),
    "conversation_not_found": "This conversation doesn't exist or was deleted.",
    "llm_unavailable": "The language model is unavailable right now. Try again in a moment.",
    "llm_timeout": "The language model took too long to answer. Try again.",
    "llm_error": "The language model couldn't answer this question.",
    "models_not_ready": "The search models are still loading. Try again in a minute.",
    "retrieval_unavailable": "Searching the manuals is unavailable right now.",
    "validation_error": "This message can't be sent: it is empty or too long.",
    "answer_withdrawn": "The model withdrew this answer.",
    STREAM_TIMEOUT: "The answer took too long and was stopped.",
    BAD_RESPONSE: "The chat service sent a response this page couldn't read.",
}


@pytest.mark.parametrize(("code", "text"), WORDING.items())
def test_each_known_code_has_its_fixed_wording(code: str, text: str) -> None:
    assert error_text(ApiError(code)) == text


@pytest.mark.parametrize(
    "code",
    [
        "internal_error",
        "unknown_document",
        "client_id_required",
        "invalid_cursor",
        "not_found",
        "something_new",
        "",
    ],
)
def test_other_codes_get_the_generic_wording(code: str) -> None:
    assert error_text(ApiError(code)) == GENERIC_ERROR_TEXT
    assert GENERIC_ERROR_TEXT == "Something went wrong. Try again."


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (7, "You're sending requests too quickly. Try again in 7 seconds."),
        (1, "You're sending requests too quickly. Try again in 1 second."),
        (None, "You're sending requests too quickly. Try again in a moment."),
    ],
)
def test_rate_limited_shows_the_retry_after_seconds(seconds: int | None, text: str) -> None:
    assert error_text(ApiError("rate_limited", status=429, retry_after_seconds=seconds)) == text


def test_an_api_error_holds_its_code_status_and_request_id_and_says_only_the_code() -> None:
    error = ApiError("llm_error", status=502, request_id="rid-12345678")

    assert (error.code, error.status, error.request_id) == ("llm_error", 502, "rid-12345678")
    assert error.retry_after_seconds is None
    assert str(error) == "llm_error"
    assert error.args == ("llm_error",)


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [("7", 7), ("1", 1), (" 12 ", 12), ("3600", 3600)],
)
def test_retry_after_in_whole_seconds_is_parsed(raw: str, seconds: int) -> None:
    assert parse_retry_after(raw) == seconds


@pytest.mark.parametrize(
    "raw", [None, "", "0", "-3", "1.5", "soon", "Wed, 21 Oct 2026 07:28:00 GMT", "9" * 30]
)
def test_unusable_retry_after_values_are_dropped(raw: str | None) -> None:
    assert parse_retry_after(raw) is None


def test_request_ids_are_shown_only_when_they_match_the_allowlist() -> None:
    assert shown_request_id("rid-12345678") == "rid-12345678"
    assert shown_request_id("rid 1234\n<b>x</b>") is None
    assert shown_request_id(None) is None

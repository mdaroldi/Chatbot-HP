"""API failures as codes, and the fixed text the page shows for each.

The text is chosen here by code, never taken from a response: a body could carry
anything, and the backend's own messages are written for developers.
"""

from collections.abc import Mapping

from hpdocs_ui.identifiers import trusted_identifier

# Codes the frontend assigns itself, next to the backend's.
BACKEND_UNREACHABLE = "backend_unreachable"
STREAM_TIMEOUT = "stream_timeout"
BAD_RESPONSE = "bad_response"
RATE_LIMITED = "rate_limited"

GENERIC_ERROR_TEXT = "Something went wrong. Try again."

ERROR_TEXT: Mapping[str, str] = {
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

_RATE_LIMITED_TEXT = "You're sending requests too quickly. Try again in {wait}."
# Longer than any Retry-After the backend sends (one token refill), bounding int().
_MAX_RETRY_AFTER_DIGITS = 6


class ApiError(Exception):
    """A failed call, as a code plus the facts the page may show.

    There is deliberately no message: str(error) is the code, so logging or showing
    an ApiError can never leak response text.
    """

    def __init__(
        self,
        code: str,
        *,
        status: int | None = None,
        request_id: str | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.status = status
        self.request_id = request_id
        self.retry_after_seconds = retry_after_seconds


def error_text(error: ApiError) -> str:
    if error.code == RATE_LIMITED:
        seconds = error.retry_after_seconds
        if seconds is None:
            wait = "a moment"
        else:
            wait = f"{seconds} second" if seconds == 1 else f"{seconds} seconds"
        return _RATE_LIMITED_TEXT.format(wait=wait)
    return ERROR_TEXT.get(error.code, GENERIC_ERROR_TEXT)


def parse_retry_after(raw: str | None) -> int | None:
    """Whole positive seconds (what the backend sends), else None: the HTTP-date form
    and anything odd are dropped rather than guessed at.
    """
    value = (raw or "").strip()
    if not value.isdecimal() or len(value) > _MAX_RETRY_AFTER_DIGITS:
        return None
    seconds = int(value)
    return seconds if seconds > 0 else None


def shown_request_id(raw: str | None) -> str | None:
    """The request ID for the small print, only if it is a well-formed identifier."""
    return trusted_identifier(raw)

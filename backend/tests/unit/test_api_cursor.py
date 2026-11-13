"""The list cursor: an opaque token for a keyset position, rejected whole when malformed."""

import base64
import json
from datetime import UTC, datetime

import pytest

from app.api.cursor import InvalidCursorError, decode_cursor, encode_cursor
from app.domain.chat_store import ConversationCursor

CURSOR = ConversationCursor(
    datetime(2026, 11, 14, 9, 0, 0, 123456, tzinfo=UTC), "00000000-0000-4000-8000-000000000001"
)


def token(payload: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")


def test_a_cursor_round_trips_exactly() -> None:
    encoded = encode_cursor(CURSOR)

    assert decode_cursor(encoded) == CURSOR
    assert encoded.isascii()
    assert "=" not in encoded


def test_a_cursor_is_url_safe() -> None:
    encoded = encode_cursor(CURSOR)

    assert encoded
    assert set(encoded) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not base64 !",
        token("just a string"),
        token([CURSOR.updated_at.isoformat()]),
        token([CURSOR.updated_at.isoformat(), CURSOR.id, "extra"]),
        token([1, CURSOR.id]),
        token([CURSOR.updated_at.isoformat(), 7]),
        token(["yesterday", CURSOR.id]),
        token(["2026-11-14T09:00:00", CURSOR.id]),  # no time zone
        token([CURSOR.updated_at.isoformat(), "not-a-uuid"]),
        base64.urlsafe_b64encode(b"\xff\xfe").decode(),
        base64.urlsafe_b64encode(b"[1,").decode(),
    ],
)
def test_a_malformed_cursor_is_rejected(raw: str) -> None:
    with pytest.raises(InvalidCursorError) as raised:
        decode_cursor(raw)

    assert (raised.value.status_code, raised.value.code) == (400, "invalid_cursor")
    assert raw not in raised.value.message or raw == ""


def test_the_id_is_normalised_to_the_canonical_uuid_form() -> None:
    upper = token([CURSOR.updated_at.isoformat(), CURSOR.id.upper()])

    assert decode_cursor(upper).id == CURSOR.id

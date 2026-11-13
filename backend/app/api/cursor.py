"""The conversation list cursor: an opaque, URL-safe token for a keyset position.

Clients pass it back unchanged. It holds only the last item's update time and ID, both
of which the client has already seen; the store still filters by client ID, so a
cursor from another client reveals nothing.
"""

import base64
import binascii
import json
import uuid
from datetime import datetime
from http import HTTPStatus

from app.core.errors import AppError
from app.domain.chat_store import ConversationCursor


class InvalidCursorError(AppError):
    status_code = HTTPStatus.BAD_REQUEST
    code = "invalid_cursor"

    def __init__(self) -> None:
        super().__init__("The cursor is not valid; start again without one.")


def encode_cursor(cursor: ConversationCursor) -> str:
    payload = json.dumps([cursor.updated_at.isoformat(), cursor.id]).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_cursor(raw: str) -> ConversationCursor:
    """Parse a cursor from encode_cursor; anything else is InvalidCursorError."""
    try:
        payload = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
        updated_at, conversation_id = payload
        moment = datetime.fromisoformat(updated_at)
        canonical = str(uuid.UUID(conversation_id))
    except (binascii.Error, ValueError, TypeError, AttributeError):
        # ValueError: bad JSON, UTF-8, unpacking, dates, UUIDs; TypeError: a non-string
        # date; AttributeError: uuid.UUID given a non-string.
        raise InvalidCursorError from None
    if moment.tzinfo is None:
        raise InvalidCursorError
    return ConversationCursor(moment, canonical)

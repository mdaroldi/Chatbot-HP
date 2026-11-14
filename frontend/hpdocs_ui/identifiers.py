"""Client and request IDs, under the backend's allowlist for identifiers.

The client ID is the only credential an anonymous browser has: whoever holds it
reads its conversations. It is generated here, checked before it is sent, and
never logged.
"""

import re
import uuid

# The backend's rule for X-Request-ID and X-Client-ID (app/core/middleware.py).
_IDENTIFIER = re.compile(r"[A-Za-z0-9-]{8,64}")


def new_client_id() -> str:
    """A random UUID4: 122 bits, so IDs cannot be guessed or enumerated."""
    return str(uuid.uuid4())


def trusted_identifier(raw: str | None) -> str | None:
    """The value if it matches the allowlist, else None: never repaired or echoed."""
    if raw is not None and _IDENTIFIER.fullmatch(raw):
        return raw
    return None

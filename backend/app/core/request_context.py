"""Per-request context shared by logging and error handling."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    """Request ID of the request being served, or None outside a request."""
    return _request_id.get()


@contextmanager
def bound_request_id(request_id: str) -> Iterator[None]:
    """Bind a request ID for the enclosed block and always restore the previous value.

    Resetting with the token (instead of setting None) keeps nested or concurrent
    contexts correct even if a block exits through an exception.
    """
    token = _request_id.set(request_id)
    try:
        yield
    finally:
        _request_id.reset(token)

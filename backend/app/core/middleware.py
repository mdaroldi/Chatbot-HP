"""Pure ASGI middleware for request IDs, access logs and the last-resort error boundary.

Pure ASGI (not BaseHTTPMiddleware) so streaming responses pass through untouched:
the middleware only rewrites the `http.response.start` message.
"""

import json
import logging
import re
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.request_context import bound_request_id

REQUEST_ID_HEADER = "x-request-id"
_REQUEST_ID_HEADER_BYTES = REQUEST_ID_HEADER.encode("ascii")
# Strict allowlist: anything else (CR/LF, spaces, non-ASCII) could forge log lines
# or headers, so it is replaced rather than sanitized.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9-]{8,64}")

_access_logger = logging.getLogger("app.access")
_logger = logging.getLogger(__name__)


def resolve_request_id(raw: bytes | None) -> str:
    """Trust a client-supplied ID only if it matches the allowlist; else make a UUID4."""
    if raw is not None:
        candidate = raw.decode("latin-1")
        if _VALID_REQUEST_ID.fullmatch(candidate):
            return candidate
    return str(uuid.uuid4())


def _incoming_request_id(scope: Scope) -> bytes | None:
    for name, value in scope["headers"]:
        if name.lower() == _REQUEST_ID_HEADER_BYTES:
            return bytes(value)
    return None


def _internal_error_body(request_id: str) -> bytes:
    error = {
        "code": "internal_error",
        "message": "Internal server error.",
        "request_id": request_id,
    }
    return json.dumps({"error": error}).encode()


class RequestIdMiddleware:
    """Assign a request ID, expose it to logs and responses, and log each request.

    It is also the outermost boundary for unhandled exceptions: they are logged once
    with the request ID and turned into a generic 500, so clients never see internals.
    If the response already started (e.g. mid-stream), the exception is re-raised so
    the server aborts the connection instead of sending a corrupt response.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = resolve_request_id(_incoming_request_id(scope))
        header = (_REQUEST_ID_HEADER_BYTES, request_id.encode("ascii"))
        status = 500
        response_started = False
        started_at = time.perf_counter()
        finished_at: float | None = None

        async def send_with_request_id(message: Message) -> None:
            nonlocal status, response_started, finished_at
            if message["type"] == "http.response.body" and not message.get("more_body"):
                await send(message)
                # Duration ends at the final body chunk, so streamed responses are timed
                # completely while post-response work (background tasks) is excluded.
                finished_at = time.perf_counter()
                return
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != _REQUEST_ID_HEADER_BYTES
                ]
                message = {**message, "headers": [*headers, header]}
            await send(message)

        with bound_request_id(request_id):
            try:
                await self.app(scope, receive, send_with_request_id)
            except Exception:
                _logger.exception("unhandled exception")
                if response_started:
                    raise
                await self._send_internal_error(send_with_request_id, request_id)
            finally:
                ended_at = finished_at if finished_at is not None else time.perf_counter()
                _access_logger.info(
                    "request completed",
                    extra={
                        "method": scope["method"],
                        "path": scope["path"],
                        "status": status,
                        "duration_ms": round((ended_at - started_at) * 1000, 3),
                    },
                )

    @staticmethod
    async def _send_internal_error(send: Send, request_id: str) -> None:
        body = _internal_error_body(request_id)
        await send(
            {
                "type": "http.response.start",
                "status": 500,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

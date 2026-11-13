"""Domain-agnostic error types and the single JSON error shape used by every response."""

from collections.abc import Mapping, Sequence
from http import HTTPStatus
from typing import cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.request_context import get_request_id

# Validation details expose only where and why; `input` and `ctx` can carry user text.
_SAFE_VALIDATION_KEYS = ("loc", "msg", "type")
_EXTRA_FORBIDDEN = "extra_forbidden"

ErrorDetail = Mapping[str, object]


class AppError(Exception):
    """Base for errors the API reports deliberately.

    Subclasses set `status_code` and `code`. The message is sent to clients, so it
    must never contain user content, secrets or internal details.
    """

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "app_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def error_body(
    code: str,
    message: str,
    request_id: str | None,
    details: Sequence[ErrorDetail] | None = None,
) -> dict[str, dict[str, object]]:
    """Build the one error shape all responses share, so clients parse a single format."""
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
            "details": list(details or []),
        }
    }


def _json_error(
    status: int,
    code: str,
    message: str,
    details: Sequence[ErrorDetail] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=error_body(code, message, get_request_id(), details),
        headers=headers,
    )


def _status_code_slug(status: int) -> str:
    return HTTPStatus(status).phrase.lower().replace(" ", "_").replace("-", "_")


# Starlette types every handler as (Request, Exception); each one below is registered
# for exactly one exception class, so the casts restate what registration guarantees.


async def _handle_app_error(_: Request, exc: Exception) -> JSONResponse:
    error = cast(AppError, exc)
    return _json_error(error.status_code, error.code, error.message)


async def _handle_http_exception(_: Request, raised: Exception) -> JSONResponse:
    exc = cast(StarletteHTTPException, raised)
    phrase = HTTPStatus(exc.status_code).phrase
    # Only plain-text details are shown; structured details may hold internals.
    message = exc.detail if isinstance(exc.detail, str) and exc.detail else phrase
    return _json_error(
        exc.status_code, _status_code_slug(exc.status_code), message, headers=exc.headers
    )


def _safe_detail(error: Mapping[str, object]) -> ErrorDetail:
    detail = {key: error[key] for key in _SAFE_VALIDATION_KEYS if key in error}
    loc = detail.get("loc")
    # The last element of a forbidden extra field's loc is the client's own key.
    if error.get("type") == _EXTRA_FORBIDDEN and isinstance(loc, tuple | list) and loc:
        detail["loc"] = list(loc[:-1])
    return detail


async def _handle_validation_error(_: Request, raised: Exception) -> JSONResponse:
    exc = cast(RequestValidationError, raised)
    details = [_safe_detail(error) for error in exc.errors()]
    return _json_error(
        HTTPStatus.UNPROCESSABLE_ENTITY, "validation_error", "Request validation failed.", details
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Map handled errors to the shared shape.

    Unhandled exceptions are not registered here on purpose: Starlette runs the
    `Exception` handler outside user middleware, where the request ID is gone.
    RequestIdMiddleware turns them into the same shape instead.
    """
    app.add_exception_handler(AppError, _handle_app_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)

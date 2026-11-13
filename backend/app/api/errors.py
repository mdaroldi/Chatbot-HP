"""Domain errors of the API's services, mapped to HTTP responses in one place.

Each maps to a fixed message: the exceptions name stages and exception types, which
are for logs (the service already logged them), not for clients.
"""

from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.errors import error_body
from app.core.request_context import get_request_id
from app.domain.retrieval import ModelsNotReadyError, RetrievalUnavailableError

_MAPPED: dict[type[Exception], tuple[int, str, str]] = {
    RetrievalUnavailableError: (
        HTTPStatus.SERVICE_UNAVAILABLE,
        "retrieval_unavailable",
        "Retrieval is unavailable.",
    ),
    ModelsNotReadyError: (
        HTTPStatus.SERVICE_UNAVAILABLE,
        "models_not_ready",
        "The query models are not ready.",
    ),
}


async def _handle_domain_error(_: Request, exc: Exception) -> JSONResponse:
    # Starlette dispatches by MRO, so a subclass arrives here too.
    status, code, message = next(v for t, v in _MAPPED.items() if isinstance(exc, t))
    return JSONResponse(status_code=status, content=error_body(code, message, get_request_id()))


def register_domain_error_handlers(app: FastAPI) -> None:
    for error in _MAPPED:
        app.add_exception_handler(error, _handle_domain_error)

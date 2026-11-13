"""Domain errors of the API's services, mapped to HTTP responses in one place.

Each maps to a fixed message: the exceptions name stages and exception types, which
are for logs (the service already logged them), not for clients.
"""

from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.errors import error_body
from app.core.request_context import get_request_id
from app.domain.chat import UnknownDocumentError
from app.domain.chat_store import ConversationBusyError, ConversationNotFoundError
from app.domain.llm import (
    LLMEmptyAnswerError,
    LLMError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
)
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
    # Another client's conversation is "not found" too: 404, never 403.
    ConversationNotFoundError: (
        HTTPStatus.NOT_FOUND,
        ConversationNotFoundError.code,
        ConversationNotFoundError.message,
    ),
    ConversationBusyError: (
        HTTPStatus.CONFLICT,
        ConversationBusyError.code,
        ConversationBusyError.message,
    ),
    # The index and this service disagree on the corpus: a deployment fault, not the
    # client's, and retrying will not help.
    UnknownDocumentError: (
        HTTPStatus.INTERNAL_SERVER_ERROR,
        UnknownDocumentError.code,
        UnknownDocumentError.message,
    ),
    # Raised before the first token (after it, they are SSE error events). Transient
    # upstream trouble is 503; a request the model rejected or answered empty is 502.
    LLMUnavailableError: (
        HTTPStatus.SERVICE_UNAVAILABLE,
        LLMUnavailableError.code,
        LLMUnavailableError.message,
    ),
    LLMTimeoutError: (
        HTTPStatus.SERVICE_UNAVAILABLE,
        LLMTimeoutError.code,
        LLMTimeoutError.message,
    ),
    LLMRequestError: (HTTPStatus.BAD_GATEWAY, LLMRequestError.code, LLMRequestError.message),
    LLMEmptyAnswerError: (
        HTTPStatus.BAD_GATEWAY,
        LLMEmptyAnswerError.code,
        LLMEmptyAnswerError.message,
    ),
    LLMError: (HTTPStatus.BAD_GATEWAY, LLMError.code, LLMError.message),
}


async def _handle_domain_error(_: Request, exc: Exception) -> JSONResponse:
    # Starlette dispatches by MRO, so a subclass arrives here too; the nearest mapped
    # class wins (an LLMTimeoutError is not reported as its base LLMError).
    status, code, message = next(_MAPPED[t] for t in type(exc).__mro__ if t in _MAPPED)
    return JSONResponse(status_code=status, content=error_body(code, message, get_request_id()))


def register_domain_error_handlers(app: FastAPI) -> None:
    for error in _MAPPED:
        app.add_exception_handler(error, _handle_domain_error)

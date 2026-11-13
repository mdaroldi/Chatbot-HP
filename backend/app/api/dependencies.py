"""FastAPI dependencies: how routes obtain settings and infrastructure adapters."""

from collections.abc import Awaitable, Callable, Sequence
from http import HTTPStatus

from fastapi import Request

from app.core.errors import AppError
from app.core.middleware import trusted_identifier
from app.core.settings import Settings
from app.domain.chat_store import ChatStore
from app.domain.health import DependencyProbe
from app.services.chat import ChatService
from app.services.retrieval import RetrievalService

CLIENT_ID_HEADER = "x-client-id"

RetrievalServiceFactory = Callable[[], Awaitable[RetrievalService]]
ChatServiceFactory = Callable[[], Awaitable[ChatService]]


def get_app_settings(request: Request) -> Settings:
    """Settings the app was built with (not re-read from the environment)."""
    settings: Settings = request.app.state.settings
    return settings


def get_readiness_probes(request: Request) -> Sequence[DependencyProbe]:
    """Probes registered on the app; tests override this with fakes."""
    probes: Sequence[DependencyProbe] = request.app.state.readiness_probes
    return probes


def get_client_id(request: Request) -> str | None:
    """The anonymous browser ID, under the request-ID rule; Phase 3 rate-limits by it.

    An invalid value counts as absent, so it is never echoed, logged or keyed on.
    """
    return trusted_identifier(request.headers.get(CLIENT_ID_HEADER))


async def get_retrieval_service(request: Request) -> RetrievalService:
    """A service over the loaded models, built per request by the composition root
    (main), so this layer never imports infra; ModelsNotReadyError until they load.
    """
    factory: RetrievalServiceFactory = request.app.state.retrieval_service_factory
    return await factory()


class ClientIdRequiredError(AppError):
    status_code = HTTPStatus.BAD_REQUEST
    code = "client_id_required"

    def __init__(self) -> None:
        super().__init__("A valid X-Client-ID header is required.")


def require_client_id(request: Request) -> str:
    """The client ID the chat endpoints scope every read and write by.

    Unlike get_client_id, absence is an error here: without it there is no owner. The
    rejected value is never echoed or logged.
    """
    client_id = get_client_id(request)
    if client_id is None:
        raise ClientIdRequiredError
    return client_id


def get_chat_store(request: Request) -> ChatStore:
    """The conversation store the lifespan built; tests override this with a fake."""
    store: ChatStore = request.app.state.chat_store
    return store


async def get_chat_service(request: Request) -> ChatService:
    """A chat service over the per-request retrieval service, built by the composition
    root (main); ModelsNotReadyError until the query models load.
    """
    factory: ChatServiceFactory = request.app.state.chat_service_factory
    return await factory()

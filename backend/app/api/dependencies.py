"""FastAPI dependencies: how routes obtain settings and infrastructure adapters."""

from collections.abc import Awaitable, Callable, Sequence

from fastapi import Request

from app.core.middleware import trusted_identifier
from app.core.settings import Settings
from app.domain.health import DependencyProbe
from app.services.retrieval import RetrievalService

CLIENT_ID_HEADER = "x-client-id"

RetrievalServiceFactory = Callable[[], Awaitable[RetrievalService]]


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

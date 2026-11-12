"""Application factory. Run with: uvicorn app.main:create_app --factory --no-server-header"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import REQUEST_ID_HEADER, RequestIdMiddleware
from app.core.settings import Settings, get_settings
from app.infra.lifecycle import open_infrastructure

_CORS_METHODS = ["GET", "POST", "DELETE"]
_CORS_HEADERS = ["Content-Type", "X-Request-ID", "X-Client-ID"]


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with open_infrastructure(app.state.settings) as infrastructure:
        app.state.infrastructure = infrastructure
        app.state.readiness_probes = infrastructure.probes
        yield


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app from explicit settings; nothing is read from the environment at import.

    Middleware order (outermost first): request ID, then CORS, then routes, so CORS
    preflights and rejections also get a request ID and an access log line.
    """
    settings = settings or get_settings()
    configure_logging(
        settings.log_level,
        include_exception_messages=settings.log_include_exception_messages,
    )

    docs = settings.api_docs_enabled
    app = FastAPI(
        title="HP Docs Chatbot API",
        version="0.1.0",
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
        lifespan=_lifespan,
    )
    app.state.settings = settings
    # Real probes are registered by the lifespan; none exist before startup.
    app.state.readiness_probes = ()
    register_exception_handlers(app)
    app.include_router(health_router)

    # add_middleware wraps the current stack: the last one added is the outermost.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_methods=_CORS_METHODS,
        allow_headers=_CORS_HEADERS,
        allow_credentials=False,
        expose_headers=[REQUEST_ID_HEADER],
    )
    app.add_middleware(RequestIdMiddleware)
    return app

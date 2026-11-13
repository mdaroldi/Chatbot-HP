"""Application factory. Run with: uvicorn app.main:create_app --factory --no-server-header"""

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from concurrent.futures import Executor
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.conversations import router as conversations_router
from app.api.errors import register_domain_error_handlers
from app.api.health import router as health_router
from app.api.search import build_search_router
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import REQUEST_ID_HEADER, RequestIdMiddleware
from app.core.settings import Settings, get_settings
from app.domain.retrieval import ProductName
from app.infra.chat_store import SqlChatStore
from app.infra.lifecycle import Infrastructure, open_infrastructure
from app.infra.qdrant_search import QdrantSearchStore
from app.infra.query_models import QueryModels
from app.ingestion.corpus import load_corpus
from app.services.retrieval import RetrievalService, product_names

ModelsFactory = Callable[[Settings, Executor], QueryModels]

_CORS_METHODS = ["GET", "POST", "DELETE"]
_CORS_HEADERS = ["Content-Type", "X-Request-ID", "X-Client-ID"]
# A pending answer older than this is left over from a crash. It must outlast any live
# request: the LLM total timeout (120 s) plus rewrite, retrieval and a margin.
PENDING_ANSWER_ABANDON_AFTER = timedelta(minutes=5)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    models_factory: ModelsFactory = app.state.query_models_factory
    async with open_infrastructure(
        app.state.settings, models_factory=models_factory
    ) as infrastructure:
        app.state.infrastructure = infrastructure
        app.state.readiness_probes = infrastructure.probes
        app.state.retrieval_service_factory = _retrieval_service_factory(
            infrastructure, app.state.settings, app.state.products
        )
        app.state.chat_store = SqlChatStore(
            infrastructure.session_factory, abandon_after=PENDING_ANSWER_ABANDON_AFTER
        )
        yield


def _retrieval_service_factory(
    infrastructure: Infrastructure, settings: Settings, products: Sequence[ProductName]
) -> Callable[[], Awaitable[RetrievalService]]:
    """Build the service per request from the loaded models; the adapters are cheap
    wrappers over clients the infrastructure owns, so nothing is opened here.
    """
    store = QdrantSearchStore(
        infrastructure.qdrant,
        settings.qdrant_collection_alias,
        timeout_seconds=settings.qdrant_search_timeout_seconds,
    )

    async def build() -> RetrievalService:
        models = infrastructure.models
        embedder = models.query_embedder()  # ModelsNotReadyError until they load
        reranker = await models.reranker()
        return RetrievalService(embedder, store, reranker, products=products)

    return build


def create_app(
    settings: Settings | None = None, *, query_models_factory: ModelsFactory | None = None
) -> FastAPI:
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
    # Injectable so tests never load real models; the default reads the baked store.
    app.state.query_models_factory = query_models_factory or QueryModels.from_settings
    # Real probes are registered by the lifespan; none exist before startup.
    app.state.readiness_probes = ()
    # The corpus allowlist names the searchable documents and the product rule's names.
    corpus = load_corpus()
    app.state.products = product_names(corpus)
    register_exception_handlers(app)
    register_domain_error_handlers(app)
    app.include_router(health_router)
    app.include_router(conversations_router)
    if settings.search_endpoint_enabled:
        app.include_router(build_search_router(settings, [doc.doc_id for doc in corpus]))

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

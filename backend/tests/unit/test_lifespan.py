"""Behavior of infrastructure startup and shutdown, including partial failures."""

import asyncio
import logging
from concurrent.futures import Executor, ThreadPoolExecutor
from datetime import timedelta
from typing import cast

import httpx
import pytest
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.settings import Environment, Settings
from app.infra.chat_store import SqlChatStore
from app.infra.inference import CpuAllowance, CpuSource
from app.infra.lifecycle import open_infrastructure
from app.infra.llm_anthropic import AnthropicProvider
from app.infra.llm_stub import StubProvider
from app.infra.probes import PostgresProbe, QdrantProbe
from app.infra.qdrant_search import IndexProbe, QdrantIndexIdentity
from app.infra.query_models import ModelNames, ModelsProbe, ModelsState, QueryModels
from app.main import create_app, pending_answer_abandon_after
from app.services.chat import ChatService
from tests.unit.conftest import SettingsFactory
from tests.unit.model_fakes import BlockingSource, Builds, FakeSource


class SpyEngine:
    def __init__(self) -> None:
        self._disposed = False

    async def dispose(self) -> None:
        self._disposed = True

    def disposed(self) -> bool:
        # A method, not an attribute, so mypy does not keep a stale narrowed value.
        return self._disposed


def is_closed(client: httpx.AsyncClient) -> bool:
    """Read the flag through a call so mypy does not reuse an earlier narrowing."""
    return client.is_closed


def is_shut_down(executor: Executor | None) -> bool:
    return isinstance(executor, ThreadPoolExecutor) and executor._shutdown


class SpyQdrant:
    """Stands in for AsyncQdrantClient: the lifespan only builds and closes it."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class Factories:
    """Records what open_infrastructure created so the test can inspect cleanup."""

    def __init__(self, *, fail_http_client: bool = False, source: FakeSource | None = None) -> None:
        self.engine = SpyEngine()
        self.client: httpx.AsyncClient | None = None
        self.fail_http_client = fail_http_client
        self.source = source or FakeSource()
        self.models: QueryModels | None = None
        self.executor: Executor | None = None
        self.qdrant = SpyQdrant()

    def qdrant_factory(self, _: Settings) -> AsyncQdrantClient:
        return cast(AsyncQdrantClient, self.qdrant)

    def engine_factory(self, _: Settings) -> AsyncEngine:
        return cast(AsyncEngine, self.engine)

    def http_client_factory(self, _: Settings) -> httpx.AsyncClient:
        if self.fail_http_client:
            raise RuntimeError("could not build client")
        self.client = httpx.AsyncClient()
        return self.client

    def models_factory(self, _: Settings, executor: Executor) -> QueryModels:
        builds = Builds()
        self.executor = executor
        self.models = QueryModels(
            self.source,
            ModelNames("demo/dense", "demo/bm25", "demo/rerank-a"),
            executor,
            build_embedder=builds.embedder,
            build_reranker=builds.reranker,
            warm_up_candidates=20,
        )
        return self.models


def within(cpus: float) -> CpuAllowance:
    return CpuAllowance(cpus, CpuSource.CGROUP)


async def test_open_infrastructure_builds_probes_and_closes_everything_on_exit(
    make_settings: SettingsFactory,
) -> None:
    factories = Factories()

    async with open_infrastructure(
        make_settings(),
        engine_factory=factories.engine_factory,
        http_client_factory=factories.http_client_factory,
        models_factory=factories.models_factory,
        qdrant_client_factory=factories.qdrant_factory,
        cpu_allowance_factory=lambda: within(8),
    ) as infra:
        assert [probe.name for probe in infra.probes] == ["postgres", "qdrant", "models", "index"]
        assert factories.client is not None
        assert not is_closed(factories.client)
        assert not factories.engine.disposed()

    assert is_closed(factories.client)
    assert factories.engine.disposed()
    assert is_shut_down(factories.executor)
    assert factories.qdrant.closed


async def test_engine_is_disposed_when_startup_fails_halfway(
    make_settings: SettingsFactory,
) -> None:
    factories = Factories(fail_http_client=True)

    with pytest.raises(RuntimeError, match="could not build client"):
        async with open_infrastructure(
            make_settings(),
            engine_factory=factories.engine_factory,
            http_client_factory=factories.http_client_factory,
            models_factory=factories.models_factory,
            qdrant_client_factory=factories.qdrant_factory,
            cpu_allowance_factory=lambda: within(8),
        ):
            pytest.fail("startup should not have completed")

    assert factories.engine.disposed()


async def test_everything_is_closed_when_a_later_startup_step_fails(
    make_settings: SettingsFactory,
) -> None:
    factories = Factories()

    with pytest.raises(LookupError):
        async with open_infrastructure(
            make_settings(),
            engine_factory=factories.engine_factory,
            http_client_factory=factories.http_client_factory,
            models_factory=factories.models_factory,
            qdrant_client_factory=factories.qdrant_factory,
            cpu_allowance_factory=lambda: within(8),
        ):
            raise LookupError

    assert factories.client is not None
    assert is_closed(factories.client)
    assert factories.engine.disposed()
    assert is_shut_down(factories.executor)
    assert factories.qdrant.closed


# ------------------------------------------------------------------ query models (Phase 2)


async def test_the_models_load_in_the_background_and_become_ready(
    make_settings: SettingsFactory,
) -> None:
    factories = Factories()

    async with open_infrastructure(
        make_settings(inference_workers=3),
        engine_factory=factories.engine_factory,
        http_client_factory=factories.http_client_factory,
        models_factory=factories.models_factory,
        qdrant_client_factory=factories.qdrant_factory,
        cpu_allowance_factory=lambda: within(8),
    ) as infra:
        assert infra.models is factories.models
        await asyncio.wait_for(infra.models_loaded, timeout=2)
        assert infra.models.state is ModelsState.READY
        assert isinstance(factories.executor, ThreadPoolExecutor)
        assert factories.executor._max_workers == 3


async def test_startup_does_not_wait_for_the_models(make_settings: SettingsFactory) -> None:
    """/health answers at once; /ready stays red with models: loading."""
    source = BlockingSource()
    factories = Factories(source=source)

    async with open_infrastructure(
        make_settings(),
        engine_factory=factories.engine_factory,
        http_client_factory=factories.http_client_factory,
        models_factory=factories.models_factory,
        qdrant_client_factory=factories.qdrant_factory,
        cpu_allowance_factory=lambda: within(8),
    ) as infra:
        await asyncio.wait_for(source.started.wait(), timeout=2)
        assert infra.models.state is ModelsState.LOADING
        task = infra.models_loaded

    assert task.cancelled()
    assert is_shut_down(factories.executor)


async def test_more_inference_threads_than_cpus_is_logged_at_startup(
    make_settings: SettingsFactory, caplog: pytest.LogCaptureFixture
) -> None:
    factories = Factories()

    with caplog.at_level(logging.WARNING, logger="app.infra.inference"):
        async with open_infrastructure(
            make_settings(inference_workers=2, inference_onnx_threads=2),
            engine_factory=factories.engine_factory,
            http_client_factory=factories.http_client_factory,
            models_factory=factories.models_factory,
            qdrant_client_factory=factories.qdrant_factory,
            cpu_allowance_factory=lambda: within(2),
        ):
            pass

    assert [r.__dict__["inference_threads"] for r in caplog.records] == [4]


async def test_app_lifespan_registers_real_probes_and_closes_them_on_shutdown(
    make_settings: SettingsFactory,
) -> None:
    """Real engine and HTTP client factories: creating them opens no connections."""
    factories = Factories()
    application = create_app(make_settings(), query_models_factory=factories.models_factory)

    async with application.router.lifespan_context(application):
        infra = application.state.infrastructure
        probes = application.state.readiness_probes
        assert [type(probe) for probe in probes] == [
            PostgresProbe,
            QdrantProbe,
            ModelsProbe,
            IndexProbe,
        ]
        assert not is_closed(infra.http_client)

    assert is_closed(infra.http_client)


async def test_app_lifespan_provides_the_sql_chat_store(make_settings: SettingsFactory) -> None:
    """Built over the infrastructure's session factory; creating it opens no connection."""
    application = create_app(make_settings(), query_models_factory=Factories().models_factory)

    async with application.router.lifespan_context(application):
        store = application.state.chat_store
        assert isinstance(store, SqlChatStore)
        assert store.sessions is application.state.infrastructure.session_factory


def test_a_pending_answer_is_abandoned_only_after_the_longest_live_request(
    make_settings: SettingsFactory,
) -> None:
    """Rewrite deadline plus the answer's total deadline plus a minute for retrieval."""
    defaults = make_settings()
    longer = make_settings(llm_rewrite_timeout_seconds=10, llm_total_timeout_seconds=300)

    assert pending_answer_abandon_after(defaults) == timedelta(seconds=185)
    assert pending_answer_abandon_after(longer) == timedelta(seconds=370)


async def test_the_chat_store_abandons_pending_answers_after_the_derived_window(
    make_settings: SettingsFactory,
) -> None:
    settings = make_settings(llm_total_timeout_seconds=200)
    application = create_app(settings, query_models_factory=Factories().models_factory)

    async with application.router.lifespan_context(application):
        store = application.state.chat_store
        assert store.abandon_after == pending_answer_abandon_after(settings)


@pytest.mark.parametrize(
    ("env", "honoured"), [(Environment.DEVELOPMENT, True), (Environment.PRODUCTION, False)]
)
async def test_app_lifespan_builds_the_chat_service_over_the_stub_and_the_live_index(
    make_settings: SettingsFactory, env: Environment, honoured: bool
) -> None:
    factories = Factories()
    settings = make_settings(app_env=env, llm_stub_token_delay_ms=7)
    application = create_app(settings, query_models_factory=factories.models_factory)

    async with application.router.lifespan_context(application):
        await asyncio.wait_for(application.state.infrastructure.models_loaded, timeout=2)
        service = await application.state.chat_service_factory()
        assert isinstance(service, ChatService)
        assert isinstance(service.index_identity, QdrantIndexIdentity)
        provider = service.provider
        assert isinstance(provider, StubProvider)
        assert provider.honours_directives is honoured
        assert provider.token_delay_ms == 7


async def test_app_lifespan_builds_the_chat_service_over_the_anthropic_adapter_and_closes_it(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[AnthropicProvider] = []

    async def record_close(self: AnthropicProvider) -> None:
        closed.append(self)

    monkeypatch.setattr(AnthropicProvider, "aclose", record_close)
    factories = Factories()
    settings = make_settings(
        llm_provider="anthropic", anthropic_api_key=SecretStr("placeholder-not-a-key")
    )
    application = create_app(settings, query_models_factory=factories.models_factory)

    async with application.router.lifespan_context(application):
        await asyncio.wait_for(application.state.infrastructure.models_loaded, timeout=2)
        service = await application.state.chat_service_factory()
        provider = service.provider
        assert isinstance(provider, AnthropicProvider)
        assert closed == []

    assert closed == [provider]

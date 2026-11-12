"""Behavior of infrastructure startup and shutdown, including partial failures."""

from typing import cast

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.settings import Settings
from app.infra.lifecycle import open_infrastructure
from app.infra.probes import PostgresProbe, QdrantProbe
from app.main import create_app
from tests.unit.conftest import SettingsFactory


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


class Factories:
    """Records what open_infrastructure created so the test can inspect cleanup."""

    def __init__(self, *, fail_http_client: bool = False) -> None:
        self.engine = SpyEngine()
        self.client: httpx.AsyncClient | None = None
        self.fail_http_client = fail_http_client

    def engine_factory(self, _: Settings) -> AsyncEngine:
        return cast(AsyncEngine, self.engine)

    def http_client_factory(self, _: Settings) -> httpx.AsyncClient:
        if self.fail_http_client:
            raise RuntimeError("could not build client")
        self.client = httpx.AsyncClient()
        return self.client


async def test_open_infrastructure_builds_probes_and_closes_everything_on_exit(
    make_settings: SettingsFactory,
) -> None:
    factories = Factories()

    async with open_infrastructure(
        make_settings(),
        engine_factory=factories.engine_factory,
        http_client_factory=factories.http_client_factory,
    ) as infra:
        assert [probe.name for probe in infra.probes] == ["postgres", "qdrant"]
        assert factories.client is not None
        assert not is_closed(factories.client)
        assert not factories.engine.disposed()

    assert is_closed(factories.client)
    assert factories.engine.disposed()


async def test_engine_is_disposed_when_startup_fails_halfway(
    make_settings: SettingsFactory,
) -> None:
    factories = Factories(fail_http_client=True)

    with pytest.raises(RuntimeError, match="could not build client"):
        async with open_infrastructure(
            make_settings(),
            engine_factory=factories.engine_factory,
            http_client_factory=factories.http_client_factory,
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
        ):
            raise LookupError

    assert factories.client is not None
    assert is_closed(factories.client)
    assert factories.engine.disposed()


async def test_app_lifespan_registers_real_probes_and_closes_them_on_shutdown(
    make_settings: SettingsFactory,
) -> None:
    """Real factories: creating an engine and client opens no connections."""
    application = create_app(make_settings())

    async with application.router.lifespan_context(application):
        infra = application.state.infrastructure
        probes = application.state.readiness_probes
        assert [type(probe) for probe in probes] == [PostgresProbe, QdrantProbe]
        assert not is_closed(infra.http_client)

    assert is_closed(infra.http_client)

"""Behavior of the Qdrant and Postgres probes and the Qdrant HTTP client factory."""

from collections.abc import Callable
from typing import cast

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from app.domain.health import DependencyUnavailableError, FailureReason, ProbeStatus
from app.infra.http import create_qdrant_http_client
from app.infra.probes import PostgresProbe, QdrantProbe
from app.services.readiness import check_readiness
from tests.unit.conftest import SettingsFactory

Handler = Callable[[httpx.Request], httpx.Response]
QDRANT = "http://qdrant.test:6333/"


def qdrant_probe(handler: Handler, base_url: str = QDRANT) -> QdrantProbe:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
    return QdrantProbe(client, base_url)


async def readiness_detail(probe: QdrantProbe) -> FailureReason | None:
    report = await check_readiness([probe], timeout_seconds=1)
    assert report.results[0].status is ProbeStatus.DOWN
    return report.results[0].detail


# ------------------------------------------------------------------ Qdrant probe


async def test_qdrant_200_is_ok_and_hits_readyz_with_get() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text="all shards are ready")

    probe = qdrant_probe(handler)
    await probe.ping()

    assert probe.name == "qdrant"
    assert [(r.method, str(r.url)) for r in seen] == [("GET", "http://qdrant.test:6333/readyz")]


async def test_qdrant_base_path_is_preserved() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200)

    await qdrant_probe(handler, base_url="http://proxy.test/qdrant").ping()

    assert seen == ["http://proxy.test/qdrant/readyz"]


@pytest.mark.parametrize("status", [503, 404, 500, 204])
async def test_qdrant_any_status_other_than_200_is_down(status: int) -> None:
    probe = qdrant_probe(lambda _: httpx.Response(status))

    with pytest.raises(DependencyUnavailableError, match=str(status)):
        await probe.ping()
    assert await readiness_detail(probe) is FailureReason.UNREACHABLE


async def test_qdrant_redirects_are_not_followed() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(307, headers={"location": "http://elsewhere.test/readyz"})

    probe = qdrant_probe(handler)

    with pytest.raises(DependencyUnavailableError, match="307"):
        await probe.ping()
    assert seen == ["/readyz"]


async def test_qdrant_connection_error_is_down_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    assert await readiness_detail(qdrant_probe(handler)) is FailureReason.UNREACHABLE


@pytest.mark.parametrize("error", [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout])
async def test_qdrant_client_timeouts_are_down_timeout(error: type[httpx.TimeoutException]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error("too slow", request=request)

    probe = qdrant_probe(handler)

    with pytest.raises(TimeoutError):
        await probe.ping()
    assert await readiness_detail(probe) is FailureReason.TIMEOUT


async def test_qdrant_client_has_explicit_timeouts_and_no_redirects(
    make_settings: SettingsFactory,
) -> None:
    async with create_qdrant_http_client(make_settings(readiness_timeout_seconds=1.5)) as client:
        assert client.timeout == httpx.Timeout(connect=1.5, read=1.5, write=1.5, pool=1.5)
        assert client.follow_redirects is False


# ------------------------------------------------------------------ Postgres probe


class FakeConnection:
    def __init__(self, statements: list[str], error: Exception | None) -> None:
        self.statements = statements
        self.error = error

    async def __aenter__(self) -> "FakeConnection":
        if self.error is not None:
            raise self.error
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def execute(self, statement: object) -> None:
        self.statements.append(str(statement))


class FakeEngine:
    """Just enough of AsyncEngine for the probe: connect() as an async context."""

    def __init__(self, error: Exception | None = None) -> None:
        self.statements: list[str] = []
        self.error = error

    def connect(self) -> FakeConnection:
        return FakeConnection(self.statements, self.error)


async def test_postgres_probe_runs_select_one() -> None:
    engine = FakeEngine()
    probe = PostgresProbe(cast(AsyncEngine, engine))

    await probe.ping()

    assert probe.name == "postgres"
    assert engine.statements == ["SELECT 1"]


async def test_postgres_connection_failure_propagates_as_down_unreachable() -> None:
    probe = PostgresProbe(cast(AsyncEngine, FakeEngine(error=ConnectionRefusedError())))

    report = await check_readiness([probe], timeout_seconds=1)

    assert report.results[0].detail is FailureReason.UNREACHABLE

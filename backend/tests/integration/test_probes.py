"""Probes against real Postgres and Qdrant containers."""

import pytest
from pydantic import HttpUrl, SecretStr

from app.domain.health import FailureReason, ProbeResult, ProbeStatus
from app.infra.db import create_engine
from app.infra.http import create_qdrant_http_client
from app.infra.probes import PostgresProbe, QdrantProbe
from app.services.readiness import check_readiness
from tests.integration.conftest import SettingsFactory

pytestmark = pytest.mark.integration


async def postgres_result(settings_factory: SettingsFactory, **overrides: object) -> ProbeResult:
    engine = create_engine(settings_factory(**overrides))
    try:
        report = await check_readiness([PostgresProbe(engine)], timeout_seconds=5)
    finally:
        await engine.dispose()
    return report.results[0]


async def test_postgres_up_is_ok(make_settings: SettingsFactory) -> None:
    result = await postgres_result(make_settings)

    assert result.status is ProbeStatus.OK


async def test_postgres_wrong_password_is_down_unreachable(make_settings: SettingsFactory) -> None:
    result = await postgres_result(make_settings, postgres_password=SecretStr("wrong-password"))

    assert (result.status, result.detail) == (ProbeStatus.DOWN, FailureReason.UNREACHABLE)


async def test_postgres_closed_port_is_down_unreachable(
    make_settings: SettingsFactory, closed_port: int
) -> None:
    result = await postgres_result(make_settings, postgres_port=closed_port)

    assert (result.status, result.detail) == (ProbeStatus.DOWN, FailureReason.UNREACHABLE)


async def qdrant_result(settings_factory: SettingsFactory, **overrides: object) -> ProbeResult:
    settings = settings_factory(**overrides)
    async with create_qdrant_http_client(settings) as client:
        report = await check_readiness(
            [QdrantProbe(client, str(settings.qdrant_url))], timeout_seconds=5
        )
    return report.results[0]


async def test_qdrant_up_is_ok(make_settings: SettingsFactory) -> None:
    result = await qdrant_result(make_settings)

    assert result.status is ProbeStatus.OK


async def test_qdrant_closed_port_is_down_unreachable(
    make_settings: SettingsFactory, closed_port: int
) -> None:
    result = await qdrant_result(
        make_settings, qdrant_url=HttpUrl(f"http://127.0.0.1:{closed_port}")
    )

    assert (result.status, result.detail) == (ProbeStatus.DOWN, FailureReason.UNREACHABLE)

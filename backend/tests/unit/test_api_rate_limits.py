"""Rate limits on the chat endpoints (by client ID) and /v1/search (client ID, else IP)."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from app.api.dependencies import get_chat_store, get_retrieval_service
from app.api.rate_limits import RateLimiters
from app.core.middleware import REQUEST_ID_HEADER
from app.core.settings import Settings
from app.main import create_app
from tests.unit.chat_fakes import InMemoryChatStore
from tests.unit.conftest import SettingsFactory
from tests.unit.test_api_search import Pipeline

REQUEST_ID = "test-request-0001"
CLIENT = "client-aaaa-0001"
OTHER = "client-bbbb-0002"


class Clock:
    def __init__(self) -> None:
        self.now = 50.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


def build(settings: Settings, clock: Clock, store: InMemoryChatStore) -> FastAPI:
    app = create_app(settings)
    app.state.rate_limiters = RateLimiters.from_settings(settings, clock=clock)
    app.dependency_overrides[get_chat_store] = lambda: store
    app.dependency_overrides[get_retrieval_service] = Pipeline().service
    return app


@pytest.fixture
def store() -> InMemoryChatStore:
    return InMemoryChatStore()


@pytest.fixture
async def http(
    make_settings: SettingsFactory, clock: Clock, store: InMemoryChatStore
) -> AsyncIterator[httpx.AsyncClient]:
    settings = make_settings(
        rate_limit_conversations_per_minute=60,
        rate_limit_conversations_burst=2,
        rate_limit_search_per_minute=6,
        rate_limit_search_burst=1,
        search_endpoint_enabled=True,
    )
    transport = httpx.ASGITransport(app=build(settings, clock, store), client=("10.0.0.7", 1234))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={REQUEST_ID_HEADER: REQUEST_ID}
    ) as client:
        yield client


def as_client(client_id: str) -> dict[str, str]:
    return {"X-Client-ID": client_id}


async def search(http: httpx.AsyncClient, headers: dict[str, str] | None = None) -> httpx.Response:
    return await http.post("/v1/search", json={"query": "synthetic question"}, headers=headers)


def assert_rate_limited(response: httpx.Response, retry_after: str) -> None:
    assert response.status_code == 429
    assert response.headers["Retry-After"] == retry_after
    assert response.json() == {
        "error": {
            "code": "rate_limited",
            "message": "Too many requests; retry after the time in Retry-After.",
            "request_id": REQUEST_ID,
            "details": [],
        }
    }


# Conversations


async def test_conversation_calls_beyond_the_burst_are_429_with_retry_after(
    http: httpx.AsyncClient, store: InMemoryChatStore
) -> None:
    for _ in range(2):
        assert (await http.get("/v1/conversations", headers=as_client(CLIENT))).status_code == 200

    response = await http.get("/v1/conversations", headers=as_client(CLIENT))

    assert_rate_limited(response, "1")
    assert store.calls == ["list_conversations", "list_conversations"]


async def test_every_conversation_route_shares_the_clients_bucket(
    http: httpx.AsyncClient, store: InMemoryChatStore
) -> None:
    created = (await http.post("/v1/conversations", headers=as_client(CLIENT))).json()
    path = f"/v1/conversations/{created['id']}"
    assert (await http.get(path, headers=as_client(CLIENT))).status_code == 200

    assert (await http.delete(path, headers=as_client(CLIENT))).status_code == 429
    assert (await http.post("/v1/conversations", headers=as_client(CLIENT))).status_code == 429
    assert "delete_conversation" not in store.calls


async def test_one_client_exhausting_its_bucket_does_not_limit_another(
    http: httpx.AsyncClient,
) -> None:
    statuses = [
        (await http.get("/v1/conversations", headers=as_client(CLIENT))).status_code
        for _ in range(3)
    ]

    assert statuses == [200, 200, 429]
    assert (await http.get("/v1/conversations", headers=as_client(OTHER))).status_code == 200


async def test_the_bucket_refills_with_time(http: httpx.AsyncClient, clock: Clock) -> None:
    for _ in range(2):
        await http.get("/v1/conversations", headers=as_client(CLIENT))
    assert (await http.get("/v1/conversations", headers=as_client(CLIENT))).status_code == 429

    clock.now += 1

    assert (await http.get("/v1/conversations", headers=as_client(CLIENT))).status_code == 200


async def test_a_request_without_a_client_id_is_400_and_spends_no_token(
    http: httpx.AsyncClient,
) -> None:
    for _ in range(5):
        assert (await http.get("/v1/conversations")).status_code == 400

    statuses = [
        (await http.get("/v1/conversations", headers=as_client(CLIENT))).status_code
        for _ in range(3)
    ]
    assert statuses == [200, 200, 429]  # the full burst of 2 was still there


# Search


async def test_search_is_limited_per_client_id(http: httpx.AsyncClient) -> None:
    assert (await search(http, as_client(CLIENT))).status_code == 200

    assert_rate_limited(await search(http, as_client(CLIENT)), "10")
    assert (await search(http, as_client(OTHER))).status_code == 200


@pytest.mark.parametrize("header", [None, "bad id!"])
async def test_search_without_a_valid_client_id_is_limited_per_ip(
    http: httpx.AsyncClient, header: str | None
) -> None:
    headers = None if header is None else {"X-Client-ID": header}
    assert (await search(http, headers)).status_code == 200

    assert (await search(http)).status_code == 429  # same IP, no header
    assert (await search(http, as_client(CLIENT))).status_code == 200  # its own bucket


async def test_the_retry_hint_is_rounded_up_to_whole_seconds(
    http: httpx.AsyncClient, clock: Clock
) -> None:
    await search(http, as_client(CLIENT))
    clock.now += 9.2

    assert_rate_limited(await search(http, as_client(CLIENT)), "1")


async def test_search_and_conversations_have_separate_buckets(http: httpx.AsyncClient) -> None:
    await search(http, as_client(CLIENT))
    assert (await search(http, as_client(CLIENT))).status_code == 429

    assert (await http.get("/v1/conversations", headers=as_client(CLIENT))).status_code == 200


async def test_the_app_builds_its_limiters_from_settings(make_settings: SettingsFactory) -> None:
    app = create_app(make_settings(rate_limit_conversations_burst=1))
    app.dependency_overrides[get_chat_store] = InMemoryChatStore
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.get("/v1/conversations", headers=as_client(CLIENT))
        second = await client.get("/v1/conversations", headers=as_client(CLIENT))

    assert (first.status_code, second.status_code) == (200, 429)


def test_limiters_take_their_limits_and_key_cap_from_settings(
    make_settings: SettingsFactory,
) -> None:
    limiters = RateLimiters.from_settings(
        make_settings(
            rate_limit_conversations_per_minute=7,
            rate_limit_conversations_burst=3,
            rate_limit_search_per_minute=5,
            rate_limit_search_burst=2,
            rate_limit_max_keys=11,
        )
    )

    assert (limiters.conversations.limit.per_minute, limiters.conversations.limit.burst) == (7, 3)
    assert (limiters.search.limit.per_minute, limiters.search.limit.burst) == (5, 2)
    assert limiters.conversations.max_keys == limiters.search.max_keys == 11


# The search endpoint in production


def startup_warnings(output: str) -> list[dict[str, Any]]:
    lines = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    return [line for line in lines if line["level"] == "WARNING"]


def test_enabling_search_in_production_starts_with_a_warning(
    make_settings: SettingsFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    app = create_app(make_settings(app_env="production", search_endpoint_enabled=True))

    warnings = startup_warnings(capsys.readouterr().out)
    assert [w["message"] for w in warnings] == ["search endpoint enabled in production"]
    assert "/v1/search" in app.openapi()["paths"]


@pytest.mark.parametrize(
    ("env", "enabled"), [("production", False), ("development", True), ("test", True)]
)
def test_no_warning_unless_search_is_enabled_in_production(
    make_settings: SettingsFactory, capsys: pytest.CaptureFixture[str], env: str, enabled: bool
) -> None:
    create_app(make_settings(app_env=env, search_endpoint_enabled=enabled))

    assert startup_warnings(capsys.readouterr().out) == []

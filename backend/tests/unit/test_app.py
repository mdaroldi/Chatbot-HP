"""Behavior of the assembled application: errors, CORS, middleware order, docs, health."""

import importlib
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import app.main
from app.api.dependencies import get_readiness_probes
from app.core.errors import AppError
from app.core.logging import JsonFormatter
from app.core.middleware import REQUEST_ID_HEADER
from app.core.settings import LogLevel
from app.domain.health import DependencyUnavailableError, FailureReason
from app.main import create_app
from tests.unit.conftest import LogReader, SettingsFactory
from tests.unit.fakes import FakeProbe

REQUEST_ID = "test-request-0001"
FRONTEND = "http://localhost:8501"
PERSONAL_TEXT = "my-home-address-is-secret-street-42"


class Question(BaseModel):
    text: str = Field(max_length=10)
    count: int


class StrictQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class TeapotError(AppError):
    status_code = 418
    code = "teapot"


def add_test_routes(application: FastAPI) -> None:
    @application.post("/test/validate")
    async def validate(question: Question) -> dict[str, int]:
        return {"count": question.count}

    @application.post("/test/strict")
    async def strict(question: StrictQuestion) -> dict[str, str]:
        return {"text": question.text}

    @application.get("/test/app-error")
    async def app_error() -> None:
        raise TeapotError("I refuse to brew coffee.")

    @application.get("/test/http-error")
    async def http_error() -> None:
        raise HTTPException(status_code=409, detail="Already exists.")

    @application.get("/test/http-error-structured")
    async def http_error_structured() -> None:
        raise HTTPException(status_code=400, detail={"internal": "do not expose"})

    @application.get("/test/crash")
    async def crash() -> None:
        raise RuntimeError("database password is hunter2")


@pytest.fixture
def application(make_settings: SettingsFactory) -> FastAPI:
    built = create_app(make_settings(cors_allowed_origins=[FRONTEND]))
    add_test_routes(built)
    return built


@pytest.fixture
async def client(application: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={REQUEST_ID_HEADER: REQUEST_ID}
    ) as http:
        yield http


def assert_error(
    response: httpx.Response, status: int, code: str, message: str
) -> list[dict[str, Any]]:
    """Every error response has exactly this shape; returns `details` for extra checks."""
    assert response.status_code == status
    assert response.headers["content-type"] == "application/json"
    assert response.headers[REQUEST_ID_HEADER] == REQUEST_ID
    body = response.json()
    assert body.keys() == {"error"}
    assert body["error"].keys() == {"code", "message", "request_id", "details"}
    assert body["error"]["code"] == code
    assert body["error"]["message"] == message
    assert body["error"]["request_id"] == REQUEST_ID
    details: list[dict[str, Any]] = body["error"]["details"]
    return details


# ------------------------------------------------------------------ error shape


async def test_not_found_uses_error_shape(client: httpx.AsyncClient) -> None:
    response = await client.get("/does-not-exist")

    assert assert_error(response, 404, "not_found", "Not Found") == []


async def test_method_not_allowed_uses_error_shape_and_keeps_allow_header(
    client: httpx.AsyncClient,
) -> None:
    response = await client.delete("/health")

    assert assert_error(response, 405, "method_not_allowed", "Method Not Allowed") == []
    assert response.headers["allow"] == "GET"


async def test_validation_error_uses_error_shape_with_location_details(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post("/test/validate", json={"text": PERSONAL_TEXT})

    details = assert_error(response, 422, "validation_error", "Request validation failed.")
    assert sorted(tuple(d["loc"]) for d in details) == [("body", "count"), ("body", "text")]
    assert all(d.keys() == {"loc", "msg", "type"} for d in details)


async def test_validation_error_never_echoes_input_or_ctx(client: httpx.AsyncClient) -> None:
    response = await client.post("/test/validate", json={"text": PERSONAL_TEXT, "count": "x"})

    assert response.status_code == 422
    assert PERSONAL_TEXT not in response.text
    assert '"input"' not in response.text
    assert '"ctx"' not in response.text
    assert '"url"' not in response.text


async def test_an_unknown_field_name_is_never_echoed(client: httpx.AsyncClient) -> None:
    """The key of a forbidden extra field is client input too, so loc stops at its parent."""
    response = await client.post("/test/strict", json={"text": "ok", PERSONAL_TEXT: 1})

    details = assert_error(response, 422, "validation_error", "Request validation failed.")
    assert details == [
        {"loc": ["body"], "msg": "Extra inputs are not permitted", "type": "extra_forbidden"}
    ]
    assert PERSONAL_TEXT not in response.text


async def test_malformed_json_body_uses_validation_error_shape(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/test/validate",
        content=b'{"text": "' + PERSONAL_TEXT.encode(),
        headers={"content-type": "application/json"},
    )

    assert_error(response, 422, "validation_error", "Request validation failed.")
    assert PERSONAL_TEXT not in response.text


async def test_app_error_uses_its_status_code_and_message(client: httpx.AsyncClient) -> None:
    response = await client.get("/test/app-error")

    assert assert_error(response, 418, "teapot", "I refuse to brew coffee.") == []


async def test_http_exception_with_text_detail_uses_it_as_message(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/test/http-error")

    assert_error(response, 409, "conflict", "Already exists.")


async def test_http_exception_with_structured_detail_falls_back_to_status_phrase(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/test/http-error-structured")

    assert_error(response, 400, "bad_request", "Bad Request")
    assert "do not expose" not in response.text


async def test_unhandled_exception_uses_error_shape_without_internals(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/test/crash")

    assert assert_error(response, 500, "internal_error", "Internal server error.") == []
    assert "hunter2" not in response.text


# ------------------------------------------------------------------ CORS and middleware order


def preflight_headers(
    origin: str = FRONTEND, method: str = "POST", headers: str = "content-type"
) -> dict[str, str]:
    return {
        "origin": origin,
        "access-control-request-method": method,
        "access-control-request-headers": headers,
    }


async def test_allowed_origin_gets_exact_origin_and_exposed_request_id(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/health", headers={"origin": FRONTEND})

    assert response.headers["access-control-allow-origin"] == FRONTEND
    assert response.headers["access-control-expose-headers"].lower() == "x-request-id"
    assert "access-control-allow-credentials" not in response.headers


@pytest.mark.parametrize(
    "origin",
    ["http://evil.example", "http://localhost:8501.evil.example", "https://localhost:8501"],
)
async def test_other_origins_get_no_cors_headers(client: httpx.AsyncClient, origin: str) -> None:
    response = await client.get("/health", headers={"origin": origin})

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


async def test_preflight_allows_configured_methods_and_headers(
    client: httpx.AsyncClient,
) -> None:
    response = await client.options(
        "/health", headers=preflight_headers(headers="content-type,x-request-id,x-client-id")
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == FRONTEND
    allowed_methods = set(response.headers["access-control-allow-methods"].split(", "))
    assert allowed_methods == {"GET", "POST", "DELETE"}
    allowed_headers = {
        h.lower() for h in response.headers["access-control-allow-headers"].split(", ")
    }
    assert {"content-type", "x-request-id", "x-client-id"} <= allowed_headers
    assert "access-control-allow-credentials" not in response.headers


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"method": "PUT"}, id="method-not-allowed"),
        pytest.param({"headers": "authorization"}, id="header-not-allowed"),
        pytest.param({"origin": "http://evil.example"}, id="origin-not-allowed"),
    ],
)
async def test_preflight_rejects_anything_not_configured(
    client: httpx.AsyncClient, overrides: dict[str, str]
) -> None:
    response = await client.options("/health", headers=preflight_headers(**overrides))

    assert response.status_code == 400


async def test_preflight_carries_request_id_and_is_access_logged(
    client: httpx.AsyncClient, json_logs: LogReader
) -> None:
    response = await client.options("/health", headers=preflight_headers())

    assert response.headers[REQUEST_ID_HEADER] == REQUEST_ID
    access = [line for line in json_logs() if line["logger"] == "app.access"]
    assert [(a["method"], a["path"], a["status"]) for a in access] == [("OPTIONS", "/health", 200)]
    assert access[0]["request_id"] == REQUEST_ID


async def test_rejected_preflight_still_carries_request_id(client: httpx.AsyncClient) -> None:
    response = await client.options("/health", headers=preflight_headers(method="PUT"))

    assert response.status_code == 400
    assert response.headers[REQUEST_ID_HEADER] == REQUEST_ID


# ------------------------------------------------------------------ health and docs


async def test_health_returns_constant_ok(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_health_does_not_touch_dependencies(make_settings: SettingsFactory) -> None:
    """Unreachable Postgres and Qdrant must not affect liveness."""
    settings = make_settings(
        postgres_host="unreachable.invalid", qdrant_url="http://unreachable.invalid:1"
    )
    transport = httpx.ASGITransport(app=create_app(settings))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        response = await http.get("/health")

    assert response.json() == {"status": "ok"}


def ready_client(
    make_settings: SettingsFactory, probes: list[FakeProbe], timeout: float = 0.5
) -> httpx.AsyncClient:
    application = create_app(make_settings(readiness_timeout_seconds=timeout))
    application.dependency_overrides[get_readiness_probes] = lambda: probes
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://test",
        headers={REQUEST_ID_HEADER: REQUEST_ID},
    )


async def test_ready_returns_200_when_all_dependencies_are_ok(
    make_settings: SettingsFactory,
) -> None:
    async with ready_client(make_settings, [FakeProbe("postgres"), FakeProbe("qdrant")]) as http:
        response = await http.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"postgres": {"status": "ok"}, "qdrant": {"status": "ok"}},
    }
    assert response.headers[REQUEST_ID_HEADER] == REQUEST_ID


async def test_ready_returns_503_status_document_with_request_id(
    make_settings: SettingsFactory,
) -> None:
    probes = [
        FakeProbe("postgres", error=OSError("connection to 10.0.0.5 refused")),
        FakeProbe("qdrant"),
    ]
    async with ready_client(make_settings, probes) as http:
        response = await http.get("/ready")

    assert response.status_code == 503
    assert response.headers["content-type"] == "application/json"
    assert response.headers[REQUEST_ID_HEADER] == REQUEST_ID
    assert response.json() == {
        "status": "not_ready",
        "checks": {
            "postgres": {"status": "down", "detail": "unreachable"},
            "qdrant": {"status": "ok"},
        },
    }
    assert "error" not in response.json()
    assert "10.0.0.5" not in response.text


async def test_ready_uses_the_configured_timeout(make_settings: SettingsFactory) -> None:
    async with ready_client(make_settings, [FakeProbe("qdrant", hang=True)], timeout=0.05) as http:
        response = await http.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {"qdrant": {"status": "down", "detail": "timeout"}}


async def test_ready_without_registered_probes_is_ready(client: httpx.AsyncClient) -> None:
    response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": {}}


async def test_ready_documents_the_503_status_document(client: httpx.AsyncClient) -> None:
    schema = (await client.get("/openapi.json")).json()

    responses = schema["paths"]["/ready"]["get"]["responses"]
    assert {"200", "503"} <= responses.keys()


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
async def test_api_docs_are_served_when_enabled(client: httpx.AsyncClient, path: str) -> None:
    response = await client.get(path)

    assert response.status_code == 200


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
async def test_api_docs_are_hidden_when_disabled(make_settings: SettingsFactory, path: str) -> None:
    transport = httpx.ASGITransport(app=create_app(make_settings(api_docs_enabled=False)))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        response = await http.get(path)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# ------------------------------------------------------------------ factory


def test_importing_main_reads_no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import must succeed even without the required POSTGRES_PASSWORD."""
    monkeypatch.delenv("POSTGRES_PASSWORD")

    module = importlib.reload(app.main)

    assert not any(isinstance(value, FastAPI) for value in vars(module).values())


def test_create_app_without_arguments_reads_settings_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_DOCS_ENABLED", "false")

    application = create_app()

    assert application.openapi_url is None


def test_create_app_passes_exception_message_flag_to_logging(
    make_settings: SettingsFactory,
) -> None:
    create_app(make_settings(log_include_exception_messages=True))

    formatter = logging.getLogger().handlers[0].formatter
    assert isinstance(formatter, JsonFormatter)
    assert formatter.include_exception_messages is True


def test_create_app_configures_logging_level(make_settings: SettingsFactory) -> None:
    create_app(make_settings(log_level=LogLevel.DEBUG))

    assert logging.getLogger().level == logging.DEBUG


async def test_ready_shows_what_probes_describe_and_why_one_is_down(
    make_settings: SettingsFactory,
) -> None:
    probes = [
        FakeProbe("models", info={"dense": "BAAI/bge-small-en-v1.5"}),
        FakeProbe(
            "index", error=DependencyUnavailableError("SECRET detail", reason=FailureReason.LOADING)
        ),
    ]
    async with ready_client(make_settings, probes) as http:
        response = await http.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {
        "models": {"status": "ok", "info": {"dense": "BAAI/bge-small-en-v1.5"}},
        "index": {"status": "down", "detail": "loading"},
    }
    assert "SECRET" not in response.text

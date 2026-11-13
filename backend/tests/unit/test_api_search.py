"""Behavior of POST /v1/search: contract, validation, errors, client ID and wiring."""

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Executor
from typing import Annotated, Any

import httpx
import pytest
from fastapi import Depends, FastAPI

from app.api.dependencies import get_client_id, get_retrieval_service
from app.core.middleware import REQUEST_ID_HEADER, resolve_request_id
from app.core.settings import Settings
from app.domain.ingestion import PageRange
from app.domain.retrieval import ModelsNotReadyError, RetrievalUnavailableError
from app.infra.query_models import ModelNames, QueryModels
from app.main import create_app
from app.services.retrieval import RetrievalService
from tests.unit.conftest import LogReader, SettingsFactory
from tests.unit.model_fakes import BlockingSource, Builds, FakeSource
from tests.unit.retrieval_fakes import FakeQueryEmbedder, FakeReranker, FakeSearchStore, make_hit

REQUEST_ID = "test-request-0001"
CLIENT_HEADER = "X-Client-ID"
CANARY = "CANARY-7f3a-secret-question"
ENVY = "envy-6000-user-guide"
OMEN = "omen-17-service-guide"
TIMING_KEYS = {"dense_embed", "sparse_embed", "search", "rerank", "total"}


class Pipeline:
    """The fakes behind one RetrievalService, kept so tests can inspect calls."""

    def __init__(self, hits: int = 3) -> None:
        # Texts of growing length: the fake reranker scores by length, reversing the order.
        self.store = FakeSearchStore(
            [
                make_hit(f"chunk-{n}", 1.0 - n / 10, text="x" * (n + 1), doc_id=OMEN)
                for n in range(hits)
            ]
        )
        self.embedder = FakeQueryEmbedder()
        self.reranker = FakeReranker()
        self.error: Exception | None = None

    async def service(self) -> RetrievalService:
        if self.error is not None:
            raise self.error
        return RetrievalService(self.embedder, self.store, self.reranker)


def build(settings: Settings, pipeline: Pipeline) -> FastAPI:
    application = create_app(settings)
    application.dependency_overrides[get_retrieval_service] = pipeline.service
    return application


def logs_after_build(request: pytest.FixtureRequest) -> LogReader:
    """create_app() replaces the root handlers, so capture starts after the app exists."""
    reader: LogReader = request.getfixturevalue("json_logs")
    return reader


def assert_logged_without(lines: list[dict[str, Any]], canary: str, message: str) -> None:
    """The capture saw the request (no vacuous pass) and no line holds the canary."""
    assert any(line["message"] == message for line in lines), [entry["message"] for entry in lines]
    assert all(canary not in str(line) for line in lines)


@pytest.fixture
def pipeline() -> Pipeline:
    return Pipeline()


@pytest.fixture
def enabled(make_settings: SettingsFactory) -> Callable[..., Settings]:
    def factory(**overrides: Any) -> Settings:  # Any: arbitrary field overrides
        return make_settings(search_endpoint_enabled=True, **overrides)

    return factory


async def post(
    application: FastAPI, body: object, *, headers: dict[str, str] | None = None
) -> httpx.Response:
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={REQUEST_ID_HEADER: REQUEST_ID}
    ) as http:
        if isinstance(body, bytes):
            sent = {**(headers or {}), "content-type": "application/json"}
            return await http.post("/v1/search", content=body, headers=sent)
        return await http.post("/v1/search", json=body, headers=headers)


MESSAGES = {
    "not_found": "Not Found",
    "validation_error": "Request validation failed.",
    "retrieval_unavailable": "Retrieval is unavailable.",
    "models_not_ready": "The query models are not ready.",
}


def assert_error(response: httpx.Response, status: int, code: str) -> list[dict[str, Any]]:
    """The unified error shape, with a fixed message per code (never request data)."""
    assert response.status_code == status
    assert response.headers["content-type"] == "application/json"
    body = response.json()
    assert body.keys() == {"error"}
    assert body["error"].keys() == {"code", "message", "request_id", "details"}
    assert body["error"]["code"] == code
    assert body["error"]["request_id"] == REQUEST_ID
    assert body["error"]["message"] == MESSAGES[code]
    details: list[dict[str, Any]] = body["error"]["details"]
    return details


# ------------------------------------------------------------------ mounting


async def test_the_endpoint_is_absent_by_default(
    make_settings: SettingsFactory, pipeline: Pipeline
) -> None:
    application = build(make_settings(), pipeline)

    response = await post(application, {"query": "hello"})

    assert_error(response, 404, "not_found")
    assert "/v1/search" not in application.openapi()["paths"]
    assert pipeline.store.requests == []


async def test_the_endpoint_is_documented_when_enabled(
    enabled: Callable[..., Settings], pipeline: Pipeline
) -> None:
    paths = build(enabled(), pipeline).openapi()["paths"]

    assert set(paths["/v1/search"]) == {"post"}
    assert {"200", "422", "503"} <= set(paths["/v1/search"]["post"]["responses"])


# ------------------------------------------------------------------ results


async def test_results_are_ranked_with_scores_sections_pages_and_timings(
    enabled: Callable[..., Settings], pipeline: Pipeline
) -> None:
    response = await post(build(enabled(), pipeline), {"query": "remove the bottom cover"})

    assert response.status_code == 200
    body = response.json()
    assert body.keys() == {"results", "relevance", "filter", "timings_ms", "config"}
    assert [r["chunk_id"] for r in body["results"]] == ["chunk-2", "chunk-1", "chunk-0"]
    assert body["results"][0] == {
        "chunk_id": "chunk-2",
        "doc_id": OMEN,
        "section_path": ["Removal and replacement", "Bottom cover"],
        "pages": {"pdf_start": 30, "pdf_end": 31, "printed_start": 28, "printed_end": 29},
        "content_type": "procedure",
        "has_warning": False,
        "score": 3.0,
        "scores": {"fused": pytest.approx(0.8), "rerank": 3.0},
        "text": "xxx",
    }
    assert body["relevance"] == {"signal": 3.0, "threshold": None, "flag": None}
    assert body["filter"] is None
    assert body["config"] == {"mode": "hybrid", "rerank": True, "rrf_k": 60, "candidates": 20}
    timings = body["timings_ms"]
    assert set(timings) == TIMING_KEYS
    assert all(isinstance(value, float) and value >= 0 for value in timings.values())


async def test_unset_printed_pages_are_null(
    enabled: Callable[..., Settings], pipeline: Pipeline
) -> None:
    hit = make_hit("unnumbered")
    pipeline.store.hits = [
        dataclasses.replace(hit, pages=PageRange(2, 3, None, None), has_warning=True)
    ]

    result = (await post(build(enabled(), pipeline), {"query": "q"})).json()["results"][0]

    assert result["pages"] == {
        "pdf_start": 2,
        "pdf_end": 3,
        "printed_start": None,
        "printed_end": None,
    }
    assert result["has_warning"] is True


async def test_top_k_defaults_to_the_setting_and_the_request_can_lower_it(
    enabled: Callable[..., Settings],
) -> None:
    application = build(enabled(retrieval_top_k=2), Pipeline(hits=4))

    default = (await post(application, {"query": "q"})).json()
    one = (await post(application, {"query": "q", "top_k": 1})).json()

    assert len(default["results"]) == 2
    assert [r["chunk_id"] for r in one["results"]] == ["chunk-3"]
    assert one["config"]["candidates"] == 20


async def test_without_rerank_the_store_order_and_score_are_returned(
    enabled: Callable[..., Settings], pipeline: Pipeline
) -> None:
    application = build(enabled(retrieval_rerank_enabled=False, retrieval_mode="dense"), pipeline)

    body = (await post(application, {"query": "q", "top_k": 2})).json()

    assert [r["chunk_id"] for r in body["results"]] == ["chunk-0", "chunk-1"]
    assert body["results"][0]["score"] == 1.0
    assert body["results"][0]["scores"] == {"fused": 1.0, "rerank": None}
    assert body["relevance"] == {"signal": None, "threshold": None, "flag": None}
    assert body["config"] == {"mode": "dense", "rerank": False, "rrf_k": 60, "candidates": 2}
    assert body["timings_ms"]["rerank"] is None
    assert body["timings_ms"]["sparse_embed"] is None
    assert pipeline.reranker.calls == []


@pytest.mark.parametrize(("threshold", "flag"), [(2.5, True), (3.5, False)])
async def test_the_relevance_flag_compares_the_top_score_with_the_threshold(
    enabled: Callable[..., Settings], pipeline: Pipeline, threshold: float, flag: bool
) -> None:
    application = build(enabled(retrieval_relevance_threshold=threshold), pipeline)

    body = (await post(application, {"query": "q"})).json()

    assert body["relevance"] == {"signal": 3.0, "threshold": threshold, "flag": flag}


async def test_a_requested_document_filters_the_search(
    enabled: Callable[..., Settings], pipeline: Pipeline
) -> None:
    body = (await post(build(enabled(), pipeline), {"query": "q", "doc_id": OMEN})).json()

    assert body["filter"] == {"doc_id": OMEN, "source": "request"}
    assert pipeline.store.requests[-1].doc_id == OMEN


async def test_an_empty_search_returns_no_results(
    enabled: Callable[..., Settings], pipeline: Pipeline
) -> None:
    pipeline.store.hits = []

    body = (await post(build(enabled(), pipeline), {"query": "q", "doc_id": ENVY})).json()

    assert body["results"] == []
    assert body["relevance"]["signal"] is None


async def test_the_query_is_stripped_before_retrieval(
    enabled: Callable[..., Settings], pipeline: Pipeline
) -> None:
    await post(build(enabled(), pipeline), {"query": "  reset the printer \n"})

    assert pipeline.embedder.dense_texts == ["reset the printer"]
    assert pipeline.reranker.calls[0][0] == "reset the printer"


async def test_the_query_never_reaches_the_logs(
    enabled: Callable[..., Settings], pipeline: Pipeline, request: pytest.FixtureRequest
) -> None:
    application = build(enabled(), pipeline)
    logs = logs_after_build(request)

    response = await post(application, {"query": CANARY})

    assert response.status_code == 200
    assert_logged_without(logs(), CANARY, "retrieval done")


# ------------------------------------------------------------------ validation

INVALID_BODIES = {
    "missing query": {"top_k": 3},
    "empty query": {"query": ""},
    "blank query": {"query": "   \n\t "},
    "query too long": {"query": CANARY + "x" * 40},
    "query not a string": {"query": [CANARY]},
    "top_k zero": {"query": CANARY, "top_k": 0},
    "top_k above the cap": {"query": CANARY, "top_k": 21},
    "top_k above the candidates": {"query": CANARY, "top_k": 11},
    "top_k not an integer": {"query": CANARY, "top_k": CANARY},
    "unknown doc_id": {"query": CANARY, "doc_id": CANARY},
    "blank doc_id": {"query": CANARY, "doc_id": ""},
    "unknown field": {"query": "q", CANARY: CANARY},
    "not an object": [CANARY],
}


@pytest.mark.parametrize("body", INVALID_BODIES.values(), ids=INVALID_BODIES.keys())
async def test_invalid_input_is_422_and_never_echoed_or_logged(
    enabled: Callable[..., Settings],
    pipeline: Pipeline,
    request: pytest.FixtureRequest,
    body: object,
) -> None:
    application = build(enabled(query_max_chars=40, retrieval_rerank_candidates=10), pipeline)
    logs = logs_after_build(request)

    response = await post(application, body)

    details = assert_error(response, 422, "validation_error")
    assert details
    assert all(detail.keys() <= {"loc", "msg", "type"} for detail in details)
    assert CANARY not in response.text
    assert_logged_without(logs(), CANARY, "request completed")
    assert pipeline.store.requests == []


async def test_a_malformed_body_is_422_and_never_echoed(
    enabled: Callable[..., Settings], pipeline: Pipeline
) -> None:
    response = await post(build(enabled(), pipeline), b'{"query": "' + CANARY.encode())

    assert_error(response, 422, "validation_error")
    assert CANARY not in response.text


async def test_a_query_at_the_limit_is_accepted(
    enabled: Callable[..., Settings], pipeline: Pipeline
) -> None:
    response = await post(build(enabled(query_max_chars=40), pipeline), {"query": "q" * 40})

    assert response.status_code == 200


async def test_top_k_may_equal_the_candidates(
    enabled: Callable[..., Settings], pipeline: Pipeline
) -> None:
    settings = enabled(retrieval_rerank_candidates=20)

    response = await post(build(settings, pipeline), {"query": "q", "top_k": 20})

    assert response.status_code == 200


async def test_every_corpus_document_is_an_accepted_doc_id(
    enabled: Callable[..., Settings], pipeline: Pipeline
) -> None:
    application = build(enabled(), pipeline)

    statuses = [
        (await post(application, {"query": "q", "doc_id": doc})).status_code for doc in (ENVY, OMEN)
    ]

    assert statuses == [200, 200]


# ------------------------------------------------------------------ errors


async def test_a_failed_stage_is_503_retrieval_unavailable(
    enabled: Callable[..., Settings], pipeline: Pipeline
) -> None:
    pipeline.store.error = RetrievalUnavailableError("search", "UnexpectedResponse")

    response = await post(build(enabled(), pipeline), {"query": CANARY})

    assert assert_error(response, 503, "retrieval_unavailable") == []
    assert "UnexpectedResponse" not in response.text
    assert CANARY not in response.text


async def test_any_adapter_failure_is_503_retrieval_unavailable(
    enabled: Callable[..., Settings], pipeline: Pipeline
) -> None:
    pipeline.reranker.error = RuntimeError(f"onnx failed on {CANARY}")

    response = await post(build(enabled(), pipeline), {"query": CANARY})

    assert assert_error(response, 503, "retrieval_unavailable") == []
    assert CANARY not in response.text


async def test_models_still_loading_is_503_models_not_ready(
    enabled: Callable[..., Settings], pipeline: Pipeline
) -> None:
    pipeline.error = ModelsNotReadyError("query models are loading")

    response = await post(build(enabled(), pipeline), {"query": "q"})

    assert assert_error(response, 503, "models_not_ready") == []


# ------------------------------------------------------------------ X-Client-ID

VALID_CLIENT_IDS = ["abcd1234", "A" * 64, "0f8e-4c2a-9b1d-client"]
INVALID_CLIENT_IDS = ["short", "x" * 65, "has space-in-it", "under_score_id", "semi;colon;id"]


@pytest.fixture
def client_id_app(make_settings: SettingsFactory) -> FastAPI:
    application = create_app(make_settings())

    @application.get("/test/client-id")
    async def echo(client_id: Annotated[str | None, Depends(get_client_id)]) -> dict[str, Any]:
        return {"client_id": client_id}

    return application


async def client_id_of(application: FastAPI, headers: dict[str, str]) -> object:
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        return (await http.get("/test/client-id", headers=headers)).json()["client_id"]


@pytest.mark.parametrize("value", VALID_CLIENT_IDS)
async def test_a_valid_client_id_is_used(client_id_app: FastAPI, value: str) -> None:
    assert await client_id_of(client_id_app, {CLIENT_HEADER: value}) == value


@pytest.mark.parametrize("value", INVALID_CLIENT_IDS)
async def test_an_invalid_client_id_counts_as_absent(client_id_app: FastAPI, value: str) -> None:
    assert await client_id_of(client_id_app, {CLIENT_HEADER: value}) is None


async def test_a_missing_client_id_is_absent(client_id_app: FastAPI) -> None:
    assert await client_id_of(client_id_app, {}) is None


async def test_client_ids_follow_the_request_id_rule(client_id_app: FastAPI) -> None:
    """Both headers share one allowlist; a value valid for one is valid for the other."""
    for value in [*VALID_CLIENT_IDS, *INVALID_CLIENT_IDS]:
        accepted = await client_id_of(client_id_app, {CLIENT_HEADER: value}) == value
        assert accepted == (resolve_request_id(value.encode()) == value), value


@pytest.mark.parametrize("value", INVALID_CLIENT_IDS)
async def test_an_invalid_client_id_is_never_echoed_or_logged(
    enabled: Callable[..., Settings], pipeline: Pipeline, request: pytest.FixtureRequest, value: str
) -> None:
    application = build(enabled(), pipeline)
    logs = logs_after_build(request)

    response = await post(application, {"query": "q"}, headers={CLIENT_HEADER: value + CANARY})

    assert response.status_code == 200
    assert CANARY not in response.text
    assert all(CANARY not in str(header) for header in response.headers.values())
    assert_logged_without(logs(), CANARY, "retrieval done")


# ------------------------------------------------------------------ wiring


def fake_models(source: FakeSource) -> Callable[[Settings, Executor], QueryModels]:
    def factory(_: Settings, executor: Executor) -> QueryModels:
        builds = Builds()
        return QueryModels(
            source,
            ModelNames("demo/dense", "demo/bm25", "demo/rerank-a"),
            executor,
            build_embedder=builds.embedder,
            build_reranker=builds.reranker,
        )

    return factory


@pytest.fixture
async def started(enabled: Callable[..., Settings]) -> AsyncIterator[FastAPI]:
    application = create_app(enabled(), query_models_factory=fake_models(FakeSource()))
    async with application.router.lifespan_context(application):
        await asyncio.wait_for(application.state.infrastructure.models_loaded, timeout=5)
        yield application


async def test_the_service_is_built_from_the_loaded_models(started: FastAPI) -> None:
    service = await started.state.retrieval_service_factory()

    assert isinstance(service, RetrievalService)
    assert service.reranker_model == "demo/rerank-a"


async def test_a_search_before_the_models_load_is_503_models_not_ready(
    enabled: Callable[..., Settings],
) -> None:
    application = create_app(enabled(), query_models_factory=fake_models(BlockingSource()))
    async with application.router.lifespan_context(application):
        response = await post(application, {"query": "q"})

    assert assert_error(response, 503, "models_not_ready") == []

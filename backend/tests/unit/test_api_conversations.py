"""Behavior of the /v1/conversations endpoints: client ID, ownership, shapes, errors."""

from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from app.api.dependencies import get_chat_store
from app.core.middleware import REQUEST_ID_HEADER
from app.domain.chat import AnswerRecord, Outcome, RefusalReason, Source
from app.domain.chat_store import ChatStore, ConversationBusyError
from app.domain.ingestion import PageRange
from app.main import create_app
from tests.unit.chat_fakes import T0, InMemoryChatStore
from tests.unit.conftest import LogReader, SettingsFactory

REQUEST_ID = "test-request-0001"
CLIENT = "client-aaaa-0001"
OTHER = "client-bbbb-0002"
CANARY = "CANARY-5e1d-personal-question"
MISSING = "00000000-0000-4000-8000-00000000dead"
SOURCE = Source(
    tag="S1",
    doc_id="synthetic-doc",
    document_title="Synthetic manual",
    section_path=("Chapter", "Section"),
    pages=PageRange(3, 4, 1, 2),
    has_warning=True,
    snippet="A short synthetic snippet.",
)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


@pytest.fixture
def store() -> InMemoryChatStore:
    return InMemoryChatStore(clock=Clock())


@pytest.fixture
def application(make_settings: SettingsFactory, store: InMemoryChatStore) -> FastAPI:
    app = create_app(make_settings())
    app.dependency_overrides[get_chat_store] = lambda: store
    return app


@pytest.fixture
async def http(application: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={REQUEST_ID_HEADER: REQUEST_ID}
    ) as client:
        yield client


def as_client(client_id: str) -> dict[str, str]:
    return {"X-Client-ID": client_id}


def test_the_fake_store_implements_the_port(store: InMemoryChatStore) -> None:
    port: ChatStore = store  # mypy checks the structural match
    assert port is store


async def create(http: httpx.AsyncClient, client_id: str = CLIENT) -> dict[str, Any]:
    response = await http.post("/v1/conversations", headers=as_client(client_id))
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def answered(store: InMemoryChatStore, client_id: str, conversation_id: str) -> None:
    turn = await store.start_turn(client_id, conversation_id, "Question?", title="Question")
    await store.finish_answer(
        client_id,
        turn.assistant.id,
        AnswerRecord(
            outcome=Outcome.COMPLETED,
            content="Answer [S1].",
            citations=("S1",),
            sources=(SOURCE,),
            model="stub",
            input_tokens=9,
        ),
    )


# X-Client-ID

ENDPOINTS = [
    ("POST", "/v1/conversations"),
    ("GET", "/v1/conversations"),
    ("GET", f"/v1/conversations/{MISSING}"),
    ("DELETE", f"/v1/conversations/{MISSING}"),
]


@pytest.mark.parametrize(("method", "path"), ENDPOINTS)
@pytest.mark.parametrize(
    "header", [None, "", "short", "has space here", "badéchars-1234", "x" * 65]
)
async def test_a_missing_or_invalid_client_id_is_400_and_reaches_no_store(
    http: httpx.AsyncClient, store: InMemoryChatStore, method: str, path: str, header: str | None
) -> None:
    # Bytes, as on the wire: httpx refuses non-ASCII str header values.
    headers: dict[bytes, bytes] = {} if header is None else {b"X-Client-ID": header.encode()}

    response = await http.request(method, path, headers=headers)

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "client_id_required",
            "message": "A valid X-Client-ID header is required.",
            "request_id": REQUEST_ID,
            "details": [],
        }
    }
    assert store.calls == []
    if header:
        assert header not in response.text


# Create, list


async def test_creating_returns_the_new_empty_conversation(http: httpx.AsyncClient) -> None:
    body = await create(http)

    assert set(body) == {"id", "title", "created_at", "updated_at"}
    assert body["title"] is None
    assert body["created_at"] == body["updated_at"]
    assert datetime.fromisoformat(body["created_at"]).tzinfo is not None


async def test_a_client_lists_only_its_own_conversations_newest_first(
    http: httpx.AsyncClient,
) -> None:
    older = await create(http)
    await create(http, OTHER)
    newer = await create(http)

    response = await http.get("/v1/conversations", headers=as_client(CLIENT))

    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body["items"]] == [newer["id"], older["id"]]
    assert body["next_cursor"] is None
    assert body["items"][0] == newer


async def test_the_list_pages_with_next_cursor(http: httpx.AsyncClient) -> None:
    created = [(await create(http))["id"] for _ in range(5)]

    seen: list[str] = []
    params: dict[str, str | int] = {"limit": 2}
    pages = 0
    while True:
        response = await http.get("/v1/conversations", params=params, headers=as_client(CLIENT))
        assert response.status_code == 200
        body = response.json()
        seen.extend(item["id"] for item in body["items"])
        pages += 1
        if body["next_cursor"] is None:
            break
        params = {"limit": 2, "cursor": body["next_cursor"]}

    assert seen == list(reversed(created))
    assert pages == 3


async def test_the_default_page_is_twenty(http: httpx.AsyncClient) -> None:
    for _ in range(21):
        await create(http)

    body = (await http.get("/v1/conversations", headers=as_client(CLIENT))).json()

    assert len(body["items"]) == 20
    assert body["next_cursor"] is not None


@pytest.mark.parametrize("limit", ["0", "101", "-1", "ten"])
async def test_a_limit_outside_one_to_one_hundred_is_a_validation_error(
    http: httpx.AsyncClient, store: InMemoryChatStore, limit: str
) -> None:
    response = await http.get(
        "/v1/conversations", params={"limit": limit}, headers=as_client(CLIENT)
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert store.calls == []


async def test_a_malformed_cursor_is_400_and_never_echoed(
    http: httpx.AsyncClient, store: InMemoryChatStore
) -> None:
    response = await http.get(
        "/v1/conversations", params={"cursor": CANARY}, headers=as_client(CLIENT)
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_cursor"
    assert CANARY not in response.text
    assert store.calls == []


async def test_an_oversized_cursor_is_a_validation_error(http: httpx.AsyncClient) -> None:
    response = await http.get(
        "/v1/conversations", params={"cursor": "a" * 257}, headers=as_client(CLIENT)
    )

    assert response.status_code == 422


# Get


async def test_get_returns_the_conversation_with_its_messages(
    http: httpx.AsyncClient, store: InMemoryChatStore
) -> None:
    conversation = await create(http)
    await answered(store, CLIENT, conversation["id"])

    response = await http.get(f"/v1/conversations/{conversation['id']}", headers=as_client(CLIENT))

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == conversation["id"]
    assert body["title"] == "Question"
    user, answer = body["messages"]
    assert set(user) == {
        "id",
        "role",
        "content",
        "created_at",
        "outcome",
        "refusal",
        "refusal_reason",
        "citations",
        "sources",
    }
    assert (user["role"], user["content"], user["outcome"], user["refusal"]) == (
        "user",
        "Question?",
        None,
        False,
    )
    assert (answer["role"], answer["outcome"], answer["citations"]) == (
        "assistant",
        "completed",
        ["S1"],
    )
    assert answer["sources"] == [
        {
            "tag": "S1",
            "document_id": "synthetic-doc",
            "document_title": "Synthetic manual",
            "section_path": ["Chapter", "Section"],
            "pages": {"pdf_start": 3, "pdf_end": 4, "printed_start": 1, "printed_end": 2},
            "has_warning": True,
            "snippet": "A short synthetic snippet.",
        }
    ]


async def test_get_shows_a_refusal_and_its_reason(
    http: httpx.AsyncClient, store: InMemoryChatStore
) -> None:
    conversation = await create(http)
    turn = await store.start_turn(CLIENT, conversation["id"], "Unrelated?", title="Unrelated")
    await store.finish_answer(
        CLIENT,
        turn.assistant.id,
        AnswerRecord(
            outcome=Outcome.REFUSED,
            content="Fixed refusal.",
            refusal_reason=RefusalReason.BELOW_THRESHOLD,
        ),
    )

    body = (
        await http.get(f"/v1/conversations/{conversation['id']}", headers=as_client(CLIENT))
    ).json()

    refusal = body["messages"][1]
    assert (refusal["refusal"], refusal["refusal_reason"]) == (True, "below_threshold")
    assert (refusal["citations"], refusal["sources"]) == ([], [])


@pytest.mark.parametrize("method", ["GET", "DELETE"])
async def test_another_clients_conversation_is_404_not_403(
    http: httpx.AsyncClient, store: InMemoryChatStore, method: str
) -> None:
    conversation = await create(http)

    response = await http.request(
        method, f"/v1/conversations/{conversation['id']}", headers=as_client(OTHER)
    )

    assert response.status_code == 404
    assert response.json()["error"] == {
        "code": "conversation_not_found",
        "message": "Conversation not found.",
        "request_id": REQUEST_ID,
        "details": [],
    }
    assert await store.get_thread(CLIENT, conversation["id"]) is not None


@pytest.mark.parametrize("method", ["GET", "DELETE"])
async def test_an_unknown_conversation_is_404(http: httpx.AsyncClient, method: str) -> None:
    response = await http.request(method, f"/v1/conversations/{MISSING}", headers=as_client(CLIENT))

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "conversation_not_found"


@pytest.mark.parametrize("method", ["GET", "DELETE"])
@pytest.mark.parametrize("bad_id", [CANARY, "123", "00000000-0000-4000-8000-00000000000g"])
async def test_a_conversation_id_must_be_a_uuid(
    http: httpx.AsyncClient, store: InMemoryChatStore, method: str, bad_id: str
) -> None:
    response = await http.request(method, f"/v1/conversations/{bad_id}", headers=as_client(CLIENT))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert bad_id not in response.text
    assert store.calls == []


async def test_the_id_reaches_the_store_in_canonical_form(http: httpx.AsyncClient) -> None:
    """The fake keys conversations by the lowercase form, as Postgres compares UUIDs."""
    conversation = await create(http)

    response = await http.get(
        f"/v1/conversations/{conversation['id'].upper()}", headers=as_client(CLIENT)
    )

    assert response.status_code == 200
    assert response.json()["id"] == conversation["id"]


# Delete


async def test_delete_is_204_and_the_conversation_is_gone(
    http: httpx.AsyncClient, store: InMemoryChatStore
) -> None:
    conversation = await create(http)
    await answered(store, CLIENT, conversation["id"])
    path = f"/v1/conversations/{conversation['id']}"

    response = await http.delete(path, headers=as_client(CLIENT))

    assert response.status_code == 204
    assert response.content == b""
    assert (await http.get(path, headers=as_client(CLIENT))).status_code == 404
    assert (await http.delete(path, headers=as_client(CLIENT))).status_code == 404


async def test_a_busy_conversation_maps_to_409_in_the_unified_shape(
    make_settings: SettingsFactory,
) -> None:
    """No route raises it before the message endpoint (commit 10); the mapping is here."""
    app = create_app(make_settings())

    @app.get("/busy")
    async def busy() -> None:
        raise ConversationBusyError

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/busy", headers={REQUEST_ID_HEADER: REQUEST_ID})

    assert response.status_code == 409
    assert response.json()["error"] == {
        "code": "conversation_busy",
        "message": "The conversation is still answering the previous message.",
        "request_id": REQUEST_ID,
        "details": [],
    }


# Privacy


async def test_message_content_and_titles_never_reach_the_logs(
    http: httpx.AsyncClient, store: InMemoryChatStore, json_logs: LogReader
) -> None:
    conversation = await create(http)
    await store.start_turn(CLIENT, conversation["id"], CANARY, title=CANARY)

    response = await http.get(f"/v1/conversations/{conversation['id']}", headers=as_client(CLIENT))
    await http.get("/v1/conversations", headers=as_client(CLIENT))

    assert CANARY in response.text  # the owner does get it back
    lines = json_logs()
    assert any(line["message"] == "request completed" for line in lines)
    assert all(CANARY not in str(line) for line in lines)
    assert all(CLIENT not in str(line) for line in lines)


async def test_routes_use_the_store_the_lifespan_put_on_the_app(
    make_settings: SettingsFactory,
) -> None:
    """Without an override, get_chat_store reads app.state.chat_store."""
    app = create_app(make_settings())
    stored = InMemoryChatStore()
    app.state.chat_store = stored

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/conversations", headers=as_client(CLIENT))

    assert response.status_code == 201
    assert stored.calls == ["create_conversation"]

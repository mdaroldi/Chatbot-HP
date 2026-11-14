"""The real Streamlit script, run with AppTest against a scripted ChatApi.

Every state is reached through the fake client and real clicks, never by setting
session state, so these tests exercise the same path a browser does.
"""

import pathlib
from collections.abc import Callable

import pytest
from streamlit.testing.v1 import AppTest

from hpdocs_ui import status, wiring
from hpdocs_ui.errors import ApiError
from hpdocs_ui.identifiers import trusted_identifier
from hpdocs_ui.sanitize import escape_markdown
from hpdocs_ui.settings import FrontendSettings
from tests.fakes import (
    CID,
    DOWN,
    NOT_READY,
    UP,
    A,
    B,
    C,
    FakeChatApi,
    FakeFetch,
    conversation,
    page,
)

APP = pathlib.Path(__file__).resolve().parents[1] / "app.py"
BACKEND = "http://backend.test:8000"

OpenApp = Callable[..., AppTest]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeChatApi:
    api = FakeChatApi()

    def build_api(settings: FrontendSettings, client_id: str) -> FakeChatApi:
        assert str(settings.backend_url) == BACKEND + "/"
        api.client_ids.append(client_id)
        return api

    monkeypatch.setenv("BACKEND_URL", BACKEND)
    monkeypatch.setattr(wiring, "build_api", build_api)
    return api


@pytest.fixture
def fetch(monkeypatch: pytest.MonkeyPatch) -> FakeFetch:
    fake_fetch = FakeFetch(UP)
    monkeypatch.setattr(status, "fetch_status", fake_fetch)
    return fake_fetch


@pytest.fixture
def open_app(fake: FakeChatApi, fetch: FakeFetch) -> OpenApp:
    def open_(**params: str) -> AppTest:
        app = AppTest.from_file(str(APP), default_timeout=30)
        for name, value in params.items():
            app.query_params[name] = value
        app.run()
        assert not app.exception
        # Every run builds the client and the first run checks the backend status.
        assert fake.client_ids
        assert fetch.calls
        return app

    return open_


def click(app: AppTest, key: str) -> AppTest:
    app.button(key=key).click().run()
    assert not app.exception
    return app


def sidebar_text(app: AppTest) -> list[str]:
    """Notices and errors in the sidebar. Success is left out: with the backend up, the
    status expander shows one.
    """
    side = app.sidebar
    return [str(e.value) for e in (*side.info, *side.warning, *side.error)]


def conversation_buttons(app: AppTest) -> list[tuple[str, str]]:
    return [
        (b.key.removeprefix("open-"), b.label)
        for b in app.sidebar.button
        if b.key and b.key.startswith("open-")
    ]


def is_selected(app: AppTest, conversation_id: str) -> bool:
    return bool(app.button(key=f"open-{conversation_id}").proto.type == "primary")


# --- backend status (replaces the Phase 0 status page) ---------------------------


def test_backend_up_shows_ready_status_in_the_sidebar_expander(
    open_app: OpenApp, fetch: FakeFetch
) -> None:
    app = open_app()

    (expander,) = app.sidebar.expander
    assert expander.label == "Backend status"
    assert [e.value for e in expander.success] == ["Backend is up and ready."]
    assert "postgres: ok" in [t.value for t in expander.text]
    assert fetch.calls == [(BACKEND + "/", "/health"), (BACKEND + "/", "/ready")]


def test_backend_down_shows_unreachable_without_internals(
    open_app: OpenApp, fetch: FakeFetch
) -> None:
    fetch.results = DOWN

    app = open_app()

    (expander,) = app.sidebar.expander
    assert [e.value for e in expander.error] == ["Backend is unreachable."]
    assert not expander.success
    assert not [t for t in expander.text if "Traceback" in str(t.value)]


def test_backend_not_ready_shows_a_warning_with_the_failing_dependency(
    open_app: OpenApp, fetch: FakeFetch
) -> None:
    fetch.results = NOT_READY

    app = open_app()

    (expander,) = app.sidebar.expander
    assert [e.value for e in expander.warning] == ["Backend is up but not ready."]
    assert "qdrant: down" in [t.value for t in expander.text]


def test_status_is_checked_once_per_session_and_again_on_refresh(
    open_app: OpenApp, fetch: FakeFetch
) -> None:
    app = open_app()
    app.run()
    assert len(fetch.calls) == 2

    click(app, "refresh-status")

    assert len(fetch.calls) == 4


# --- URL and client ID -----------------------------------------------------------


def test_a_first_visit_gets_a_new_client_id_in_the_url(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    app = open_app()

    client_id = app.query_params["cid"]
    assert trusted_identifier(client_id) == client_id
    assert "conv" not in app.query_params
    assert set(fake.client_ids) == {client_id}


def test_the_url_keeps_a_valid_client_and_its_conversation_in_canonical_form(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = page(conversation(A))

    app = open_app(cid=CID, conv=A.upper())

    assert (app.query_params["cid"], app.query_params["conv"]) == (CID, A)
    assert set(fake.client_ids) == {CID}
    assert is_selected(app, A)


def test_an_invalid_conversation_is_dropped_from_the_url(open_app: OpenApp) -> None:
    app = open_app(cid=CID, conv="not-a-uuid")

    assert app.query_params["cid"] == CID
    assert "conv" not in app.query_params


# --- the conversation list -------------------------------------------------------


def test_the_list_shows_each_title_escaped_and_untitled_ones_by_a_fixed_label(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    hostile = "**Bold** ![x](http://evil.invalid/a.png) <b>"
    fake.pages[None] = page(conversation(A, hostile), conversation(B, None), conversation(C, " "))

    app = open_app(cid=CID)

    assert conversation_buttons(app) == [
        (A, escape_markdown(hostile)),
        (B, "Untitled conversation"),
        (C, "Untitled conversation"),
    ]


def test_the_list_is_fetched_once_until_something_changes_it(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = page(conversation(A), conversation(B))
    app = open_app(cid=CID)

    app.run()
    click(app, f"open-{A}")

    assert fake.calls == [("list", None)]


def test_load_more_appends_the_next_page_and_disappears_at_the_end(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = page(conversation(A), next_cursor="cursor-1")
    fake.pages["cursor-1"] = page(conversation(B))
    app = open_app(cid=CID)

    click(app, "load-more")

    assert [key for key, _ in conversation_buttons(app)] == [A, B]
    assert fake.calls == [("list", None), ("list", "cursor-1")]
    assert not [b for b in app.sidebar.button if b.key == "load-more"]


def test_a_failed_list_shows_the_fixed_wording_and_is_fetched_again_on_the_next_run(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = ApiError("backend_unreachable")
    app = open_app(cid=CID)

    assert sidebar_text(app) == ["The chat service can't be reached right now."]

    fake.pages[None] = page(conversation(A))
    app.run()

    assert [key for key, _ in conversation_buttons(app)] == [A]
    assert sidebar_text(app) == []
    assert fake.count("list") == 2


def test_a_failed_load_more_keeps_the_list_and_shows_the_fixed_wording(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = page(conversation(A), next_cursor="cursor-1")
    fake.pages["cursor-1"] = ApiError("rate_limited", status=429, retry_after_seconds=3)
    app = open_app(cid=CID)

    click(app, "load-more")

    assert [key for key, _ in conversation_buttons(app)] == [A]
    assert sidebar_text(app) == ["You're sending requests too quickly. Try again in 3 seconds."]


# --- selection -------------------------------------------------------------------


def test_selecting_a_conversation_marks_it_and_writes_it_to_the_url(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = page(conversation(A), conversation(B))
    app = open_app(cid=CID)
    assert not is_selected(app, A)

    click(app, f"open-{B}")

    assert app.query_params["conv"] == B
    assert (is_selected(app, A), is_selected(app, B)) == (False, True)


def test_new_conversation_clears_the_selection_from_the_url(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = page(conversation(A))
    app = open_app(cid=CID, conv=A)

    click(app, "new-conversation")

    assert "conv" not in app.query_params
    assert not is_selected(app, A)
    assert fake.calls == [("list", None)]


# --- delete ----------------------------------------------------------------------


def test_delete_asks_for_confirmation_and_cancel_keeps_the_conversation(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = page(conversation(A))
    app = open_app(cid=CID, conv=A)

    click(app, f"delete-{A}")
    assert [w.value for w in app.sidebar.warning] == [
        "Delete this conversation? This can't be undone."
    ]

    click(app, f"cancel-delete-{A}")

    assert fake.count("delete") == 0
    assert not app.sidebar.warning
    assert [key for key, _ in conversation_buttons(app)] == [A]
    assert app.query_params["conv"] == A


def test_a_confirmed_delete_of_the_selected_conversation_removes_it_and_refetches(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = page(conversation(A), conversation(B))
    app = open_app(cid=CID, conv=A)
    click(app, f"delete-{A}")
    fake.pages[None] = page(conversation(B))

    click(app, f"confirm-delete-{A}")

    assert ("delete", A) in fake.calls
    assert fake.count("list") == 2
    assert [key for key, _ in conversation_buttons(app)] == [B]
    assert "conv" not in app.query_params
    assert sidebar_text(app) == ["The conversation was deleted."]


def test_deleting_another_conversation_keeps_the_selection(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = page(conversation(A), conversation(B))
    app = open_app(cid=CID, conv=A)
    click(app, f"delete-{B}")
    fake.pages[None] = page(conversation(A))

    click(app, f"confirm-delete-{B}")

    assert app.query_params["conv"] == A
    assert is_selected(app, A)
    assert [key for key, _ in conversation_buttons(app)] == [A]


def test_a_delete_answering_not_found_drops_the_conversation_with_the_not_found_notice(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = page(conversation(A), conversation(B))
    fake.delete_errors[A] = ApiError("conversation_not_found", status=404)
    app = open_app(cid=CID, conv=A)
    click(app, f"delete-{A}")
    fake.pages[None] = page(conversation(B))

    click(app, f"confirm-delete-{A}")

    assert [key for key, _ in conversation_buttons(app)] == [B]
    assert "conv" not in app.query_params
    assert sidebar_text(app) == ["This conversation doesn't exist or was deleted."]
    assert fake.count("list") == 2


def test_a_failed_delete_keeps_the_conversation_and_shows_the_fixed_wording(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = page(conversation(A))
    fake.delete_errors[A] = ApiError("backend_unreachable")
    app = open_app(cid=CID, conv=A)
    click(app, f"delete-{A}")

    click(app, f"confirm-delete-{A}")

    assert [key for key, _ in conversation_buttons(app)] == [A]
    assert app.query_params["conv"] == A
    assert sidebar_text(app) == ["The chat service can't be reached right now."]
    assert fake.count("list") == 1


def test_a_notice_is_shown_once(open_app: OpenApp, fake: FakeChatApi) -> None:
    fake.pages[None] = page(conversation(A))
    app = open_app(cid=CID)
    click(app, f"delete-{A}")
    fake.pages[None] = page()
    click(app, f"confirm-delete-{A}")
    assert sidebar_text(app) == ["The conversation was deleted."]

    app.run()

    assert sidebar_text(app) == []

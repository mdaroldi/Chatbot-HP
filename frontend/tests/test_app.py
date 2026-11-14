"""The real Streamlit script, run with AppTest against a scripted ChatApi.

Every state is reached through the fake client and real clicks, never by setting
session state, so these tests exercise the same path a browser does.
"""

import pathlib
from collections.abc import Callable, Iterator
from itertools import accumulate

import pytest
import streamlit as st
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.testing.v1 import AppTest

from hpdocs_ui import status, wiring
from hpdocs_ui.chat_view import PRIMING_TEXT
from hpdocs_ui.errors import ApiError
from hpdocs_ui.identifiers import trusted_identifier
from hpdocs_ui.sanitize import escape_markdown, sanitize_markdown
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
    done,
    page,
    tokens,
)
from tests.test_sanitize import assert_safe

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


class Sent:
    """Every message the page sends towards the browser, in order.

    AppTest keeps only the last version of each element, and Streamlit's queue merges
    updates to one element; recording at the queue's entry sees every intermediate
    state a browser could be shown. Strings in the log are markers set by the test.
    """

    def __init__(self) -> None:
        self.log: list[ForwardMsg | str] = []

    def mark(self, label: str) -> None:
        self.log.append(label)

    def watch(self, fake: FakeChatApi) -> None:
        """Marks where the page opens and closes the answer stream."""
        fake.on_stream_start = lambda: self.mark("stream started")
        fake.on_stream_closed = lambda: self.mark("stream closed")


# The proto is untyped (protobuf ships no stubs); fields are read with getattr.
Element = tuple[tuple[int, ...], str, object]


def elements(log: list[ForwardMsg | str]) -> Iterator[Element]:
    """(delta path, element kind, element proto) of each element drawn."""
    for item in log:
        if not isinstance(item, ForwardMsg) or item.WhichOneof("type") != "delta":
            continue
        if item.delta.WhichOneof("type") != "new_element":
            continue
        element = item.delta.new_element
        kind = element.WhichOneof("type")
        assert kind is not None
        yield tuple(item.metadata.delta_path), kind, getattr(element, kind)


def bubbles(log: list[ForwardMsg | str]) -> list[tuple[str, tuple[int, ...]]]:
    """(role, delta path) of each chat message block opened."""
    found = []
    for item in log:
        if isinstance(item, ForwardMsg) and item.WhichOneof("type") == "delta":
            block = item.delta.add_block
            if item.delta.WhichOneof("type") == "add_block" and block.HasField("chat_message"):
                found.append((block.chat_message.name, tuple(item.metadata.delta_path)))
    return found


def texts_in(log: list[ForwardMsg | str], role: str) -> list[str]:
    """The plain and Markdown texts drawn inside the chat messages of one role."""
    paths = [path for name, path in bubbles(log) if name == role]
    return [
        str(getattr(proto, "body", ""))
        for path, kind, proto in elements(log)
        if kind in {"text", "markdown"} and any(path[: len(p)] == p for p in paths)
    ]


def answer_updates(log: list[ForwardMsg | str]) -> list[str]:
    """Every body drawn into the answer's text slot, from the priming caption on (the
    placeholder itself draws an empty element first).
    """
    drawn = [
        (path, str(getattr(proto, "body", "")))
        for path, kind, proto in elements(log)
        if kind != "empty"
    ]
    (slot,) = {path for path, body in drawn if body == PRIMING_TEXT}
    return [body for path, body in drawn if path == slot]


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> Sent:
    recorder = Sent()
    enqueue = ForwardMsgQueue.enqueue

    def record(queue: ForwardMsgQueue, msg: ForwardMsg) -> None:
        copy = ForwardMsg()
        copy.CopyFrom(msg)
        recorder.log.append(copy)
        enqueue(queue, msg)

    monkeypatch.setattr(ForwardMsgQueue, "enqueue", record)
    return recorder


def ask(app: AppTest, question: str) -> AppTest:
    app.chat_input(key="question").set_value(question).run()
    assert not app.exception
    return app


def main_errors(app: AppTest) -> list[str]:
    return [str(e.value) for e in app.main.error]


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


# --- sending a question ----------------------------------------------------------

# An image split across tokens: some prefixes hold half an image, one a whole one.
EVIL_TOKENS = ("Press ", "!", "[x", "](http://evil", ".invalid/a.png)", " then OK [S1].")


def test_every_update_of_the_streaming_answer_shows_the_sanitized_text_so_far(
    open_app: OpenApp, fake: FakeChatApi, sent: Sent
) -> None:
    fake.created = [conversation(A, None)]
    fake.streams = [((*tokens(*EVIL_TOKENS), done("Press OK [S1].")), None)]
    sent.watch(fake)
    app = open_app(cid=CID)

    ask(app, "How do I print?")

    prefixes = list(accumulate(EVIL_TOKENS))
    with pytest.raises(AssertionError):
        assert_safe(prefixes[4])  # control: unsanitized, this prefix is an image
    updates = answer_updates(sent.log[: sent.log.index("stream closed")])
    # Streamlit strips a Markdown body before sending it.
    assert updates == [PRIMING_TEXT, *(sanitize_markdown(p).strip() for p in prefixes)]
    for body in updates:
        assert_safe(body)
    assert "evil.invalid" not in "".join(
        str(getattr(proto, "src", "")) for _, _, proto in elements(sent.log)
    )


def test_the_question_and_the_priming_caption_are_drawn_before_the_answer_is_read(
    open_app: OpenApp, fake: FakeChatApi, sent: Sent
) -> None:
    fake.created = [conversation(A, None)]
    fake.streams = [((*tokens("Hi."), done("Hi.")), None)]
    sent.watch(fake)
    app = open_app(cid=CID)

    ask(app, "  How do I *print*?  ")

    before = sent.log[: sent.log.index("stream started")]
    assert texts_in(before, "user") == ["How do I *print*?"]
    assert texts_in(before, "assistant") == [PRIMING_TEXT]
    assert fake.questions == ["How do I *print*?"]


def test_every_widget_is_disabled_while_the_answer_streams_and_enabled_after(
    open_app: OpenApp, fake: FakeChatApi, sent: Sent
) -> None:
    fake.pages[None] = page(conversation(B), next_cursor="cursor-1")
    fake.created = [conversation(A, None)]
    fake.streams = [((*tokens("Hi."), done("Hi.")), None)]
    app = open_app(cid=CID)
    click(app, f"delete-{B}")  # the inline confirmation adds two more buttons
    sent.log.clear()
    sent.watch(fake)

    ask(app, "How do I print?")

    sending = sent.log[: sent.log.index("stream closed")]
    widgets = [
        (kind, bool(getattr(proto, "disabled")))  # noqa: B009 - untyped proto
        for _, kind, proto in elements(sending)
        if "disabled" in getattr(proto, "DESCRIPTOR").fields_by_name  # noqa: B009 - untyped proto
    ]
    kinds = [kind for kind, _ in widgets]
    assert kinds.count("button") == 7  # new, open, delete, confirm, cancel, more, refresh
    assert kinds.count("chat_input") == 1
    assert all(disabled for _, disabled in widgets)
    assert not app.chat_input(key="question").disabled
    assert not [b for b in app.button if b.disabled]


def test_a_question_is_posted_once_however_often_the_page_reruns(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.created = [conversation(A, None)]
    fake.streams = [((*tokens("Hi."), done("Hi.")), None)]
    app = open_app(cid=CID)

    ask(app, "How do I print?")
    app.run()
    app.run()

    assert not app.exception
    assert (fake.count("create"), fake.count("stream")) == (1, 1)
    assert fake.questions == ["How do I print?"]
    assert fake.streams_closed == 1


def test_a_blank_question_sends_nothing(open_app: OpenApp, fake: FakeChatApi) -> None:
    app = open_app(cid=CID)

    ask(app, "   ")

    assert (fake.count("create"), fake.count("stream")) == (0, 0)


def test_the_input_refuses_text_longer_than_the_backend_accepts(
    open_app: OpenApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UI_MESSAGE_MAX_CHARS", "321")

    app = open_app(cid=CID)

    assert app.chat_input(key="question").max_chars == 321


def test_the_first_question_creates_the_conversation_and_puts_it_in_the_url_first(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.created = [conversation(A, None)]
    fake.streams = [((*tokens("Hi."), done("Hi.")), None)]
    url_when_reading: list[dict[str, str]] = []
    fake.on_stream_start = lambda: url_when_reading.append(st.query_params.to_dict())
    app = open_app(cid=CID)
    fake.pages[None] = page(conversation(A, "How do I print?"))

    ask(app, "How do I print?")

    # A reload while the answer streams finds the conversation in the URL.
    assert url_when_reading == [{"cid": CID, "conv": A}]
    assert [call for call in fake.calls if call[0] != "list"] == [("create", None), ("stream", A)]
    assert app.query_params["conv"] == A
    assert conversation_buttons(app) == [(A, escape_markdown("How do I print?"))]
    assert is_selected(app, A)
    assert fake.count("list") == 2


def test_a_question_in_the_open_conversation_creates_nothing(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = page(conversation(A))
    fake.streams = [((*tokens("Hi."), done("Hi.")), None)]
    app = open_app(cid=CID, conv=A)

    ask(app, "How do I print?")

    assert [call for call in fake.calls if call[0] != "list"] == [("stream", A)]
    assert app.query_params["conv"] == A


@pytest.mark.parametrize(
    ("code", "status_code", "wording"),
    [
        ("backend_unreachable", None, "The chat service can't be reached right now."),
        ("bad_response", 200, "The chat service sent a response this page couldn't read."),
    ],
)
def test_a_failed_create_sends_nothing_and_keeps_the_question_to_copy(
    open_app: OpenApp, fake: FakeChatApi, code: str, status_code: int | None, wording: str
) -> None:
    fake.created = [ApiError(code, status=status_code, request_id="req-12345678")]
    app = open_app(cid=CID)

    ask(app, "How do I print?")

    assert fake.count("stream") == 0
    assert main_errors(app) == [wording]
    assert [c.value for c in app.code] == ["How do I print?"]
    assert [c.value for c in app.caption] == [escape_markdown("Request ID: req-12345678")]
    assert not app.chat_message
    assert "conv" not in app.query_params


def test_a_send_refused_before_the_first_token_keeps_the_question_out_of_the_thread(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.created = [conversation(A, None)]
    fake.streams = [((), ApiError("llm_unavailable", status=503))]
    app = open_app(cid=CID)

    ask(app, "How do I print?")

    assert main_errors(app) == [
        "The language model is unavailable right now. Try again in a moment."
    ]
    assert [c.value for c in app.code] == ["How do I print?"]
    assert not app.caption  # no request ID to show
    assert not app.chat_message
    assert app.query_params["conv"] == A  # the conversation exists; the question can be resent


def test_a_failure_mid_stream_shows_its_wording_without_offering_the_stored_question(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.created = [conversation(A, None)]
    fake.streams = [(tokens("Press the"), ApiError("llm_error", status=200))]
    app = open_app(cid=CID)

    ask(app, "How do I print?")

    assert main_errors(app) == ["The language model couldn't answer this question."]
    assert not app.code


def test_a_send_answering_not_found_drops_the_conversation_and_keeps_the_question(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = page(conversation(A), conversation(B))
    fake.streams = [((), ApiError("conversation_not_found", status=404))]
    app = open_app(cid=CID, conv=A)
    fake.pages[None] = page(conversation(B))

    ask(app, "Still there?")

    assert "conv" not in app.query_params
    assert sidebar_text(app) == ["This conversation doesn't exist or was deleted."]
    assert main_errors(app) == ["This conversation doesn't exist or was deleted."]
    assert [c.value for c in app.code] == ["Still there?"]
    assert [key for key, _ in conversation_buttons(app)] == [B]
    assert fake.count("list") == 2
    assert fake.count("create") == 0


def test_the_last_error_is_cleared_by_the_next_question(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.created = [conversation(A, None)]
    fake.streams = [
        ((), ApiError("llm_timeout", status=503)),
        ((*tokens("Hi."), done("Hi.")), None),
    ]
    app = open_app(cid=CID)
    ask(app, "How do I print?")
    assert main_errors(app)

    ask(app, "How do I print?")

    assert main_errors(app) == []
    assert not app.code

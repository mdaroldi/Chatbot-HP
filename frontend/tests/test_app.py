"""The real Streamlit script, run with AppTest against a scripted ChatApi.

Every state is reached through the fake client and real clicks, never by setting
session state, so these tests exercise the same path a browser does.
"""

import logging
import pathlib
from collections.abc import Callable, Iterator
from itertools import accumulate

import pytest
import streamlit as st
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block, Node

from hpdocs_ui import status, wiring
from hpdocs_ui.api import Message, SourcesEvent, StreamEvent
from hpdocs_ui.chat_view import NOTICE_TEXT, PRIMING_TEXT, REFUSAL_TEXT, WARNING_BADGE
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
    answered,
    asked,
    conversation,
    detail,
    done,
    page,
    source,
    tokens,
    up,
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


def main_captions(app: AppTest) -> list[str]:
    return [str(c.value) for c in app.main.caption]


def request_ids(app: AppTest) -> list[str]:
    return [c for c in main_captions(app) if c.startswith("Request ID")]


def drawn(node: Node) -> list[tuple[str, str]]:
    """Everything a node draws, in order, as (type, proto) pairs."""
    found = [(node.type, str(node.proto))]
    if isinstance(node, Block):
        for child in node.children.values():
            found.extend(drawn(child))
    return found


def roles(app: AppTest) -> list[str]:
    return [str(m.name) for m in app.chat_message]


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

    fake.details[A] = detail(A)
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
    fake.details[A] = detail(A)
    click(app, f"open-{A}")

    assert [call for call in fake.calls if call[0] == "list"] == [("list", None)]


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

    fake.details[B] = detail(B)
    click(app, f"open-{B}")

    assert app.query_params["conv"] == B
    assert (is_selected(app, A), is_selected(app, B)) == (False, True)


def test_new_conversation_clears_the_selection_from_the_url(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = page(conversation(A))
    fake.details[A] = detail(A)
    app = open_app(cid=CID, conv=A)

    click(app, "new-conversation")

    assert "conv" not in app.query_params
    assert not is_selected(app, A)
    assert [call for call in fake.calls if call[0] == "list"] == [("list", None)]


# --- delete ----------------------------------------------------------------------


def test_delete_asks_for_confirmation_and_cancel_keeps_the_conversation(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = page(conversation(A))
    fake.details[A] = detail(A)
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
    fake.details[A] = detail(A)
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
    fake.details[A] = detail(A)
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
    fake.details[A] = detail(A)
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
    fake.details[A] = detail(A)
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
    # new, open, delete, confirm, cancel, more, refresh status, and the four examples
    assert kinds.count("button") == 11
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
    fake.details[A] = detail(A)
    app = open_app(cid=CID, conv=A)

    ask(app, "How do I print?")

    assert [call for call in fake.calls if call[0] != "list"] == [("get", A), ("stream", A)]
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
    assert request_ids(app) == [escape_markdown("Request ID: req-12345678")]
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
    assert request_ids(app) == []  # no request ID to show
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
    fake.details[A] = detail(A)
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


# --- the thread ------------------------------------------------------------------


def test_a_finished_answer_is_shown_once_however_often_the_page_reruns(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.created = [conversation(A, None)]
    fake.streams = [((*tokens("Press ", "OK [S1]."), source_event(), done("Press OK [S1].")), None)]
    app = open_app(cid=CID)

    ask(app, "How do I print?")
    app.run()
    app.run()

    assert fake.count("stream") == 1
    assert roles(app) == ["user", "assistant"]
    user, answer = app.chat_message
    assert [t.value for t in user.text] == ["How do I print?"]
    assert [m.value for m in answer.markdown] == [sanitize_markdown("Press OK [S1].")]
    assert fake.count("get") == 0  # the streamed answer needs no read back


def source_event() -> SourcesEvent:
    return SourcesEvent(sources=(source(),))


# Each case: what the stream gave, and what GET returns for the same stored turn.
PARITY: dict[str, tuple[tuple[tuple[StreamEvent, ...], ApiError | None], Message]] = {
    "completed": (
        ((*tokens("Press OK [S1]."), source_event(), done("Press OK [S1].")), None),
        answered("Press OK [S1].", citations=("S1",), sources=(source(),)),
    ),
    "uncited": (
        ((*tokens("Press OK."), source_event(), done("Press OK.", uncited=True)), None),
        answered("Press OK.", sources=(source(),)),
    ),
    "refused": (
        (
            (*tokens("BACKEND"), SourcesEvent(sources=()), done("x", reason="model_refused")),
            None,
        ),
        answered("BACKEND-REFUSAL", "refused", reason="model_refused"),
    ),
    "failed": (
        (tokens("Press the"), ApiError("llm_error", status=200)),
        answered("Press the", "failed"),
    ),
}


@pytest.mark.parametrize("case", PARITY)
def test_a_reload_draws_the_same_thread_as_the_stream_did(
    open_app: OpenApp, fake: FakeChatApi, case: str
) -> None:
    script, stored = PARITY[case]
    fake.created = [conversation(A, None)]
    fake.streams = [script]
    streamed = ask(open_app(cid=CID), "How do I *print*?")
    fake.details[A] = detail(A, asked("How do I *print*?"), stored)

    reloaded = open_app(cid=CID, conv=A)

    assert fake.calls.count(("get", A)) == 1
    assert roles(reloaded) == ["user", "assistant"]
    assert [drawn(m) for m in reloaded.chat_message] == [drawn(m) for m in streamed.chat_message]


@pytest.mark.parametrize(
    ("stored", "notice"),
    [
        (answered("Press OK."), NOTICE_TEXT["uncited"]),
        (answered("Press the", "failed"), NOTICE_TEXT["failed"]),
        (answered("Press", "cancelled"), NOTICE_TEXT["cancelled"]),
        (answered("", "pending"), NOTICE_TEXT["pending"]),
    ],
)
def test_an_answer_that_is_not_a_cited_completion_carries_its_notice(
    open_app: OpenApp, fake: FakeChatApi, stored: Message, notice: str
) -> None:
    fake.details[A] = detail(A, asked("How?"), stored)

    app = open_app(cid=CID, conv=A)

    _, answer = app.chat_message
    shown = [str(e.value) for e in (*answer.info, *answer.warning)]
    assert shown == [notice]


def test_a_refusal_shows_the_fixed_wording_and_no_notice(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.details[A] = detail(
        A, asked("Who won?"), answered("BACKEND-REFUSAL", "refused", reason="below_threshold")
    )

    app = open_app(cid=CID, conv=A)

    _, answer = app.chat_message
    assert [m.value for m in answer.markdown] == [REFUSAL_TEXT["below_threshold"]]
    assert not answer.info
    assert not answer.warning
    assert not answer.expander


def test_sources_are_listed_under_their_answer_with_warnings_and_plain_snippets(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    hostile = "**Load** ![i](http://evil.invalid/i.png) <b>"
    fake.details[A] = detail(
        A,
        asked(hostile),
        answered(
            "Press OK [S1] [S2].",
            citations=("S1", "S2"),
            sources=(
                source("S1", snippet=hostile),
                source("S2", title=hostile, section_path=(), has_warning=True, snippet="Hot."),
            ),
        ),
    )

    app = open_app(cid=CID, conv=A)

    user, answer = app.chat_message
    assert [t.value for t in user.text] == [hostile]
    assert not user.markdown
    (expander,) = answer.expander
    assert expander.label == "Sources"
    assert [c.value for c in expander.caption] == [
        escape_markdown("[S1] HP ENVY 6000 User Guide \u203a Print \u203a Print documents, p. 38"),
        escape_markdown(f"[S2] {hostile}, p. 38"),
    ]
    assert [t.value for t in expander.text] == [hostile, "Hot."]
    badges = [m.value for m in expander.markdown if WARNING_BADGE in str(m.value)]
    assert len(badges) == 1


def test_a_thread_is_read_once_when_its_conversation_is_selected(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = page(conversation(A), conversation(B))
    fake.details[A] = detail(A, asked("First?"), answered("One [S1].", citations=("S1",)))
    fake.details[B] = detail(B, asked("Second?"), answered("Two [S1].", citations=("S1",)))
    app = open_app(cid=CID, conv=A)
    app.run()

    click(app, f"open-{B}")
    app.run()

    assert [call for call in fake.calls if call[0] == "get"] == [("get", A), ("get", B)]
    assert [t.value for t in app.chat_message[0].text] == ["Second?"]


def test_a_thread_gone_clears_the_selection_with_the_not_found_notice(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.pages[None] = page(conversation(B))
    fake.details[A] = ApiError("conversation_not_found", status=404)

    app = open_app(cid=CID, conv=A)

    assert "conv" not in app.query_params
    assert sidebar_text(app) == ["This conversation doesn't exist or was deleted."]
    assert not app.chat_message
    assert [key for key, _ in conversation_buttons(app)] == [B]
    app.run()
    # Read after the thread, so the list read in the first run is already current.
    assert fake.count("list") == 1


def test_a_failed_thread_read_shows_its_wording_and_is_read_again_on_the_next_run(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.details[A] = ApiError("backend_unreachable", request_id="req-12345678")
    app = open_app(cid=CID, conv=A)

    assert main_errors(app) == ["The chat service can't be reached right now."]
    assert main_captions(app) == [escape_markdown("Request ID: req-12345678")]
    assert not app.chat_message
    assert app.query_params["conv"] == A

    fake.details[A] = detail(A, asked("First?"), answered("One [S1].", citations=("S1",)))
    app.run()

    assert main_errors(app) == []
    assert roles(app) == ["user", "assistant"]
    assert fake.count("get") == 2


def test_a_new_answer_is_drawn_after_the_thread_already_loaded(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.details[A] = detail(A, asked("First?"), answered("One [S1].", citations=("S1",)))
    fake.streams = [((*tokens("Two [S1]."), source_event(), done("Two [S1].")), None)]
    app = open_app(cid=CID, conv=A)

    ask(app, "Second?")

    assert roles(app) == ["user", "assistant", "user", "assistant"]
    questions = [t.value for m in app.chat_message if m.name == "user" for t in m.text]
    assert questions == ["First?", "Second?"]
    assert fake.count("get") == 1


def sources_open(app: AppTest) -> list[bool]:
    return [bool(e.proto.expanded) for e in app.expander if e.label == "Sources"]


def test_only_the_latest_answer_opens_its_sources_after_a_stream_and_after_a_reload(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    first = answered("One [S1].", citations=("S1",), sources=(source(),))
    fake.details[A] = detail(A, asked("First?"), first)
    fake.streams = [((*tokens("Two [S1]."), source_event(), done("Two [S1].")), None)]
    streamed = ask(open_app(cid=CID, conv=A), "Second?")
    second = answered("Two [S1].", citations=("S1",), sources=(source(),))
    fake.details[A] = detail(A, asked("First?"), first, asked("Second?"), second)

    reloaded = open_app(cid=CID, conv=A)

    assert sources_open(streamed) == [False, True]
    assert sources_open(reloaded) == [False, True]
    assert [drawn(m) for m in reloaded.chat_message] == [drawn(m) for m in streamed.chat_message]


def test_earlier_sources_stay_closed_when_the_latest_answer_has_none(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.details[A] = detail(
        A,
        asked("First?"),
        answered("One [S1].", citations=("S1",), sources=(source(),)),
        asked("Who won?"),
        answered("BACKEND-REFUSAL", "refused", reason="below_threshold"),
    )

    app = open_app(cid=CID, conv=A)

    assert sources_open(app) == [False]


# --- notices, the empty state and the error states (commit 9a) -------------------

EXAMPLES = [
    "How do I remove the bottom cover of the OMEN 17?",
    "What does error E4 mean on the HP ENVY 6000?",
    "What is the spare part number for the OMEN 17 bottom cover?",
    "Who won the 2022 World Cup?",
]
ANTHROPIC_PRIVACY = (
    "Your questions, recent messages and the manual excerpts found for them are sent "
    "to the Anthropic API to write the answers."
)
STUB_PRIVACY = "Answers come from the built-in test model; nothing is sent to Anthropic."
UNKNOWN_PRIVACY = (
    "When the cloud model is on, your questions, recent messages and the manual excerpts "
    "found for them are sent to the Anthropic API to write the answers."
)
LINK_NOTICE = "This page's link opens your conversations. Don't share it."
PRIVACY = {"anthropic": ANTHROPIC_PRIVACY, "stub": STUB_PRIVACY, None: UNKNOWN_PRIVACY}


def example_buttons(app: AppTest) -> list[str]:
    return [str(b.label) for b in app.main.button if b.key and b.key.startswith("example-")]


@pytest.mark.parametrize("llm", ["anthropic", "stub", None, "something-else"])
def test_the_sidebar_says_where_questions_go_and_that_the_link_is_private(
    open_app: OpenApp, fetch: FakeFetch, llm: str | None
) -> None:
    fetch.results = up(llm)

    app = open_app(cid=CID)

    expected = PRIVACY.get(llm, UNKNOWN_PRIVACY)
    assert [str(c.value) for c in app.sidebar.caption][:2] == [expected, LINK_NOTICE]


def test_the_privacy_notice_follows_the_provider_after_a_status_refresh(
    open_app: OpenApp, fetch: FakeFetch
) -> None:
    fetch.results = DOWN
    app = open_app(cid=CID)
    assert str(app.sidebar.caption[0].value) == UNKNOWN_PRIVACY

    fetch.results = up("stub")
    click(app, "refresh-status")

    assert str(app.sidebar.caption[0].value) == STUB_PRIVACY


@pytest.mark.parametrize("llm", ["anthropic", "stub", None])
def test_the_empty_state_offers_the_four_examples_and_repeats_the_privacy_line(
    open_app: OpenApp, fetch: FakeFetch, llm: str | None
) -> None:
    fetch.results = up(llm)

    app = open_app(cid=CID)

    assert example_buttons(app) == EXAMPLES
    assert PRIVACY[llm] in main_captions(app)
    assert not app.chat_message


@pytest.mark.parametrize("index", range(len(EXAMPLES)))
def test_an_example_is_sent_like_a_typed_question_and_the_empty_state_goes(
    open_app: OpenApp, fake: FakeChatApi, index: int
) -> None:
    fake.created = [conversation(A, None)]
    fake.streams = [((*tokens("Remove the screws."), done("Remove the screws.")), None)]
    app = open_app(cid=CID)

    click(app, f"example-{index}")
    app.run()

    assert fake.questions == [EXAMPLES[index]]
    assert (fake.count("create"), fake.count("stream")) == (1, 1)
    assert [t.value for t in app.chat_message[0].text] == [EXAMPLES[index]]
    assert example_buttons(app) == []
    assert app.query_params["conv"] == A


def test_an_empty_conversation_shows_the_examples_but_a_failed_read_does_not(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.details[A] = detail(A)
    fake.details[B] = ApiError("backend_unreachable")

    assert example_buttons(open_app(cid=CID, conv=A)) == EXAMPLES
    assert example_buttons(open_app(cid=CID, conv=B)) == []


def test_the_examples_are_disabled_while_an_answer_streams(
    open_app: OpenApp, fake: FakeChatApi, sent: Sent
) -> None:
    fake.created = [conversation(A, None)]
    fake.streams = [((*tokens("Hi."), done("Hi.")), None)]
    app = open_app(cid=CID)
    sent.log.clear()
    sent.watch(fake)

    click(app, "example-3")

    sending = sent.log[: sent.log.index("stream closed")]
    labels = {
        str(getattr(proto, "label")): bool(getattr(proto, "disabled"))  # noqa: B009 - untyped proto
        for _, kind, proto in elements(sending)
        if kind == "button"
    }
    assert {label: labels[label] for label in EXAMPLES} == dict.fromkeys(EXAMPLES, True)


def test_error_state_backend_down_offers_retry_which_checks_again(
    open_app: OpenApp, fake: FakeChatApi, fetch: FakeFetch
) -> None:
    fetch.results = DOWN
    fake.pages[None] = ApiError("backend_unreachable")
    app = open_app(cid=CID)

    assert "The chat service can't be reached right now." in main_errors(app)
    assert app.button(key="retry-backend").label == "Retry"

    fetch.results = UP
    fake.pages[None] = page(conversation(A))
    click(app, "retry-backend")

    assert main_errors(app) == []
    assert not [b for b in app.button if b.key == "retry-backend"]
    assert [key for key, _ in conversation_buttons(app)] == [A]
    assert len(fetch.calls) == 4


def test_error_state_a_backend_that_is_up_offers_no_retry(
    open_app: OpenApp, fetch: FakeFetch
) -> None:
    fetch.results = NOT_READY

    app = open_app(cid=CID)

    assert main_errors(app) == []
    assert not [b for b in app.button if b.key == "retry-backend"]


@pytest.mark.parametrize(
    ("error", "wording"),
    [
        (
            ApiError("rate_limited", status=429, retry_after_seconds=7),
            "You're sending requests too quickly. Try again in 7 seconds.",
        ),
        (
            ApiError("rate_limited", status=429),
            "You're sending requests too quickly. Try again in a moment.",
        ),
        (
            ApiError("llm_unavailable", status=503),
            "The language model is unavailable right now. Try again in a moment.",
        ),
        (
            ApiError("llm_timeout", status=503),
            "The language model took too long to answer. Try again.",
        ),
        (ApiError("llm_error", status=502), "The language model couldn't answer this question."),
        (
            ApiError("models_not_ready", status=503),
            "The search models are still loading. Try again in a minute.",
        ),
        (
            ApiError("retrieval_unavailable", status=503),
            "Searching the manuals is unavailable right now.",
        ),
        (
            ApiError("validation_error", status=422),
            "This message can't be sent: it is empty or too long.",
        ),
        (ApiError("unknown_document", status=500), "Something went wrong. Try again."),
        (ApiError("internal_error", status=500), "Something went wrong. Try again."),
        (ApiError("stream_timeout"), "The answer took too long and was stopped."),
    ],
)
def test_error_state_before_the_first_token_shows_fixed_wording_and_the_question(
    open_app: OpenApp, fake: FakeChatApi, error: ApiError, wording: str
) -> None:
    fake.details[A] = detail(A, asked("First?"), answered("One [S1].", citations=("S1",)))
    fake.streams = [((), error)]
    app = open_app(cid=CID, conv=A)

    ask(app, "How do I print?")

    assert main_errors(app) == [wording]
    assert [c.value for c in app.code] == ["How do I print?"]
    assert roles(app) == ["user", "assistant"]
    assert not [b for b in app.button if b.key == "refresh-busy"]


@pytest.mark.parametrize(
    ("code", "wording"),
    [
        ("llm_unavailable", "The language model is unavailable right now. Try again in a moment."),
        ("llm_timeout", "The language model took too long to answer. Try again."),
        ("internal_error", "Something went wrong. Try again."),
    ],
)
def test_error_state_mid_stream_keeps_the_partial_answer_marked_interrupted(
    open_app: OpenApp, fake: FakeChatApi, code: str, wording: str
) -> None:
    fake.created = [conversation(A, None)]
    fake.streams = [(tokens("Press the"), ApiError(code, status=200))]
    app = open_app(cid=CID)

    ask(app, "How do I print?")

    _, answer = app.chat_message
    assert [m.value for m in answer.markdown] == ["Press the"]
    assert [str(w.value) for w in answer.warning] == [NOTICE_TEXT["failed"]]
    assert main_errors(app) == [wording]


def test_error_state_busy_offers_refresh_which_rereads_the_thread(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.details[A] = detail(A, asked("First?"), answered("", "pending"))
    fake.streams = [((), ApiError("conversation_busy", status=409, request_id="req-12345678"))]
    app = open_app(cid=CID, conv=A)

    ask(app, "Second?")

    assert main_errors(app) == [
        "An answer for this conversation is still being written. "
        "Wait for it to finish, then refresh."
    ]
    assert escape_markdown("Request ID: req-12345678") in main_captions(app)
    assert fake.count("stream") == 1

    fake.details[A] = detail(A, asked("First?"), answered("One [S1].", citations=("S1",)))
    click(app, "refresh-busy")

    assert main_errors(app) == []
    assert fake.count("get") == 2
    assert fake.count("stream") == 1
    _, answer = app.chat_message
    assert [m.value for m in answer.markdown] == ["One [S1]."]


def test_a_pending_answer_offers_refresh_which_rereads_the_thread(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    fake.details[A] = detail(A, asked("First?"), answered("", "pending"))
    app = open_app(cid=CID, conv=A)
    _, answer = app.chat_message
    assert [str(i.value) for i in answer.info] == [NOTICE_TEXT["pending"]]
    assert not app.chat_input(key="question").disabled

    fake.details[A] = detail(A, asked("First?"), answered("One [S1].", citations=("S1",)))
    click(app, "refresh-answer-1")

    assert fake.count("get") == 2
    _, answer = app.chat_message
    assert not answer.info
    assert not [b for b in app.button if (b.key or "").startswith("refresh-answer")]


def test_a_pending_answer_refresh_is_disabled_while_an_answer_streams(
    open_app: OpenApp, fake: FakeChatApi, sent: Sent
) -> None:
    fake.details[A] = detail(A, asked("First?"), answered("", "pending"))
    fake.streams = [((*tokens("Hi."), done("Hi.")), None)]
    app = open_app(cid=CID, conv=A)
    sent.log.clear()
    sent.watch(fake)

    ask(app, "Second?")

    sending = sent.log[: sent.log.index("stream closed")]
    refresh = [
        bool(getattr(proto, "disabled"))  # noqa: B009 - untyped proto
        for _, kind, proto in elements(sending)
        if kind == "button" and str(getattr(proto, "label")) == "Refresh"  # noqa: B009
    ]
    assert refresh == [True]


# --- the top-level guard and the canary log audit (commit 9b) --------------------

CANARY = "CANARY-9b-3c1d7a"
CANARY_CID = "cid-CANARY-9b-0042"
GENERIC = "Something went wrong. Try again."
# Streamlit's loggers do not propagate, so the audit hooks each one directly.
AUDITED_PREFIXES = ("streamlit", "hpdocs_ui", "httpx", "httpcore")

Logged = Callable[[], str]


@pytest.fixture
def logged(caplog: pytest.LogCaptureFixture, capfd: pytest.CaptureFixture[str]) -> Iterator[Logged]:
    """Everything logged at DEBUG and up, with tracebacks, plus whatever reached the
    process's stdout and stderr (Streamlit's console handler writes to stderr).
    """
    caplog.set_level(logging.DEBUG)
    names = [
        name for name in logging.root.manager.loggerDict if name.split(".")[0] in AUDITED_PREFIXES
    ]
    hooked = [logging.getLogger(name) for name in {"streamlit", *names}]
    levels = [(logger, logger.level) for logger in hooked]
    for logger in hooked:
        logger.addHandler(caplog.handler)
        logger.setLevel(logging.DEBUG)
    assert "streamlit.error_util" in names  # where uncaught script errors are logged
    output: list[str] = []
    formatter = logging.Formatter()

    def text() -> str:
        captured = capfd.readouterr()
        output.extend((captured.out, captured.err))
        records = [
            "\n".join(
                (
                    record.getMessage(),
                    formatter.formatException(record.exc_info) if record.exc_info else "",
                    record.stack_info or "",
                )
            )
            for record in caplog.records
        ]
        return "\n".join((*records, *output))

    yield text
    for logger, level in levels:
        logger.removeHandler(caplog.handler)
        logger.setLevel(level)


def page_text(log: list[ForwardMsg | str]) -> str:
    return "\n".join(str(item) for item in log)


def open_broken(fake: FakeChatApi) -> AppTest:
    """The page with its first read failing, so it never reaches the status check."""
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.query_params["cid"] = CID
    app.run()
    assert not app.exception
    assert fake.client_ids
    return app


def test_the_audit_sees_what_streamlit_logs_for_an_unguarded_error(logged: Logged) -> None:
    def unguarded() -> None:
        raise RuntimeError("CANARY-9b-3c1d7a")

    app = AppTest.from_function(unguarded)
    app.run()

    assert app.exception
    assert CANARY in logged()


@pytest.mark.usefixtures("fetch")
def test_a_bug_in_the_page_shows_the_generic_wording_and_logs_only_its_type(
    fake: FakeChatApi, sent: Sent, logged: Logged
) -> None:
    fake.pages[None] = RuntimeError(CANARY)

    app = open_broken(fake)

    assert main_errors(app) == [GENERIC]
    assert CANARY not in page_text(sent.log)
    text = logged()
    assert CANARY not in text
    assert "RuntimeError" in text
    assert fake.count("list") == 2  # one rerun to draw an enabled page, then no loop
    app.run()
    assert main_errors(app) == [GENERIC]


@pytest.mark.usefixtures("fetch")
def test_after_a_passing_bug_the_page_recovers_and_the_wording_goes(fake: FakeChatApi) -> None:
    fake.pages[None] = RuntimeError(CANARY)
    app = open_broken(fake)
    assert main_errors(app) == [GENERIC]

    fake.pages[None] = page(conversation(A))
    app.run()

    assert main_errors(app) == []
    assert [key for key, _ in conversation_buttons(app)] == [A]


def test_a_bug_in_a_callback_shows_the_generic_wording_and_the_page_still_works(
    open_app: OpenApp, fake: FakeChatApi, sent: Sent, logged: Logged
) -> None:
    fake.pages[None] = page(conversation(A), conversation(B))
    fake.delete_errors[B] = RuntimeError(CANARY)
    app = open_app(cid=CID)
    click(app, f"delete-{B}")

    click(app, f"confirm-delete-{B}")

    assert main_errors(app) == [GENERIC]
    assert [key for key, _ in conversation_buttons(app)] == [A, B]
    assert CANARY not in page_text(sent.log)
    assert CANARY not in logged()
    app.run()
    assert main_errors(app) == []


def test_a_bug_while_an_answer_streams_leaves_an_enabled_page(
    open_app: OpenApp, fake: FakeChatApi, sent: Sent, logged: Logged
) -> None:
    fake.created = [conversation(A, None)]
    fake.streams = [(tokens("Press the"), RuntimeError(CANARY))]
    fake.details[A] = detail(A, asked("How do I print?"), answered("Press the", "cancelled"))
    app = open_app(cid=CID)

    ask(app, "How do I print?")

    assert main_errors(app) == [GENERIC]
    assert not app.chat_input(key="question").disabled
    assert not [b for b in app.button if b.disabled]
    assert (fake.count("stream"), fake.streams_closed) == (1, 1)
    assert CANARY not in page_text(sent.log)
    assert CANARY not in logged()


def test_after_a_bug_mid_answer_the_thread_is_read_again_from_the_backend(
    open_app: OpenApp, fake: FakeChatApi
) -> None:
    # The backend stored the turn (here as cancelled) while the page failed.
    fake.created = [conversation(A, None)]
    fake.streams = [(tokens("Press the"), RuntimeError(CANARY))]
    fake.details[A] = detail(A, asked("How do I print?"), answered("Press the", "cancelled"))
    app = open_app(cid=CID)

    ask(app, "How do I print?")

    assert main_errors(app) == [GENERIC]
    assert fake.count("get") == 1
    assert roles(app) == ["user", "assistant"]
    assert [str(w.value) for w in app.main.warning] == [NOTICE_TEXT["cancelled"]]
    app.run()
    assert fake.count("get") == 1  # read once, not on every run


@pytest.mark.usefixtures("fetch")
def test_a_bug_in_two_runs_in_a_row_still_reads_the_thread_again(fake: FakeChatApi) -> None:
    fake.details[A] = detail(A, asked("How do I print?"), answered("Press OK."))
    fake.pages[None] = RuntimeError(CANARY)
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.query_params.update(cid=CID, conv=A)
    app.run()
    assert not app.exception
    assert main_errors(app) == [GENERIC]
    assert fake.count("get") == 2  # the failed run, then its rerun, which failed too

    fake.pages[None] = page(conversation(A))
    app.run()

    assert main_errors(app) == []
    assert fake.count("get") == 3


def test_a_bug_before_the_page_state_exists_shows_the_generic_wording(
    monkeypatch: pytest.MonkeyPatch, logged: Logged
) -> None:
    monkeypatch.setenv("BACKEND_URL", f"not a url {CANARY}")
    app = AppTest.from_file(str(APP), default_timeout=30)

    app.run()

    assert not app.exception
    assert main_errors(app) == [GENERIC]
    assert CANARY not in logged()


def test_canary_never_reaches_the_logs_in_any_flow(
    open_app: OpenApp, fake: FakeChatApi, fetch: FakeFetch, logged: Logged
) -> None:
    fetch.results = up("anthropic")
    hostile = f"{CANARY} ![x](http://evil.invalid/{CANARY}.png)"
    answer: tuple[StreamEvent, ...] = (
        *tokens(f"Press OK {CANARY} [S1]."),
        source_event(),
        done(f"OK {CANARY} [S1]."),
    )
    fake.pages[None] = page(conversation(A, CANARY), conversation(B, CANARY))
    fake.created = [
        ApiError("bad_response", status=201),
        conversation(C, CANARY),
    ]
    fake.streams = [
        (answer, None),
        ((), ApiError("rate_limited", status=429, request_id="req-12345678")),
        (tokens(CANARY), ApiError("llm_error", status=200)),
        ((), ApiError("bad_response", status=200)),
        (tokens(CANARY), RuntimeError(CANARY)),
    ]
    fake.details[A] = detail(A, asked(CANARY), answered(CANARY, citations=("S1",)))
    fake.details[B] = ApiError("backend_unreachable")
    # Read again after the bug mid-answer, as the page guard asks.
    fake.details[C] = detail(C, asked(CANARY), answered(CANARY, "cancelled"))
    fake.delete_errors[A] = ApiError("internal_error", status=500)
    fake.delete_errors[B] = RuntimeError(CANARY)
    fake.pages["cursor"] = RuntimeError(CANARY)

    app = open_app(cid=CANARY_CID)
    ask(app, hostile)  # create answers bad_response
    ask(app, hostile)  # streams the answer
    for _ in range(4):
        ask(app, hostile)
    click(app, f"open-{A}")
    click(app, f"open-{B}")
    for conversation_id in (A, B):
        click(app, f"delete-{conversation_id}")
        click(app, f"confirm-delete-{conversation_id}")
    fake.pages[None] = RuntimeError(CANARY)
    click(app, "new-conversation")

    assert fake.questions == [hostile] * 5
    assert set(fake.client_ids) == {CANARY_CID}
    text = logged()
    assert CANARY not in text
    assert CANARY_CID not in text
    assert "evil.invalid" not in text

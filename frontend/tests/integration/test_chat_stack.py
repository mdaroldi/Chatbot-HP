"""Every answer outcome through the real page, the real HTTP client and the backend.

Each scenario uses its own client ID, so none shares a rate-limit bucket or a
conversation list with another.
"""

import logging
import time
import uuid

import pytest
from streamlit.testing.v1 import AppTest

from hpdocs_ui.api import Message, Token
from hpdocs_ui.chat_view import NOTICE_TEXT, REFUSAL_TEXT
from hpdocs_ui.errors import ERROR_TEXT, ApiError
from hpdocs_ui.http_api import HttpChatApi
from hpdocs_ui.settings import FrontendSettings
from tests.integration.conftest import OpenPage

pytestmark = pytest.mark.integration

IN_SCOPE = "How do I remove the bottom cover of the OMEN 17?"
OUT_OF_SCOPE = "Who won the 2022 World Cup?"
CANARY = "CANARY-ui-it-7b2e"
# The backend marks a closed stream cancelled after the request ends.
SETTLE_SECONDS = 15.0


def ask(app: AppTest, question: str) -> AppTest:
    app.chat_input(key="question").set_value(question).run()
    assert not app.exception
    return app


def main_errors(app: AppTest) -> list[str]:
    return [str(e.value) for e in app.main.error]


def warnings_in(app: AppTest) -> list[str]:
    return [str(w.value) for w in app.main.warning]


def answer_text(app: AppTest) -> list[str]:
    return [str(m.value) for m in app.chat_message[-1].markdown]


def last_answer(api: HttpChatApi, conversation_id: str) -> Message:
    messages = api.get_conversation(conversation_id).messages
    assert [m.role for m in messages][-2:] == ["user", "assistant"]
    return messages[-1]


def settled(api: HttpChatApi, conversation_id: str) -> Message:
    """The last answer once the backend stops writing it."""
    deadline = time.monotonic() + SETTLE_SECONDS
    answer = last_answer(api, conversation_id)
    while answer.outcome == "pending" and time.monotonic() < deadline:
        time.sleep(0.25)
        answer = last_answer(api, conversation_id)
    return answer


def test_a_question_streams_a_cited_answer_with_its_sources(
    open_page: OpenPage, client_id: str, api: HttpChatApi, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    app = open_page(client_id)

    ask(app, f"{IN_SCOPE} {CANARY}")

    assert main_errors(app) == []
    assert [str(m.name) for m in app.chat_message] == ["user", "assistant"]
    conversation_id = app.query_params["conv"]
    stored = last_answer(api, conversation_id)
    assert (stored.outcome, stored.refusal_reason) == ("completed", None)
    assert stored.citations
    assert stored.sources
    answer = app.chat_message[-1]
    (expander,) = answer.expander
    assert expander.label == "Sources"
    assert len(expander.caption) == len(stored.sources)
    assert not answer.warning
    assert CANARY not in caplog.text


def test_a_refusal_by_the_model_shows_its_wording(
    open_page: OpenPage, client_id: str, api: HttpChatApi
) -> None:
    app = open_page(client_id)

    ask(app, f"{IN_SCOPE} [stub:refuse]")

    assert main_errors(app) == []
    assert answer_text(app) == [REFUSAL_TEXT["model_refused"]]
    stored = last_answer(api, app.query_params["conv"])
    assert (stored.outcome, stored.refusal_reason) == ("refused", "model_refused")


def test_an_out_of_scope_question_is_refused_before_the_model(
    open_page: OpenPage, client_id: str, api: HttpChatApi
) -> None:
    app = open_page(client_id)

    ask(app, OUT_OF_SCOPE)

    assert answer_text(app) == [REFUSAL_TEXT["below_threshold"]]
    stored = last_answer(api, app.query_params["conv"])
    assert (stored.outcome, stored.refusal_reason) == ("refused", "below_threshold")


def test_a_failure_before_the_first_token_keeps_only_the_question_to_copy(
    open_page: OpenPage, client_id: str, api: HttpChatApi
) -> None:
    question = f"{IN_SCOPE} [stub:fail-before]"
    app = open_page(client_id)

    ask(app, question)

    assert main_errors(app) == [ERROR_TEXT["llm_unavailable"]]
    assert [c.value for c in app.code] == [question]
    assert not app.chat_message
    assert api.get_conversation(app.query_params["conv"]).messages == ()


def test_a_failure_mid_answer_keeps_the_partial_answer_marked_incomplete(
    open_page: OpenPage, client_id: str, api: HttpChatApi
) -> None:
    app = open_page(client_id)

    ask(app, f"{IN_SCOPE} [stub:fail-after]")

    assert main_errors(app) == [ERROR_TEXT["llm_unavailable"]]
    assert warnings_in(app) == [NOTICE_TEXT["failed"]]
    assert answer_text(app) != [""]
    stored = last_answer(api, app.query_params["conv"])
    assert stored.outcome == "failed"
    assert stored.content


def test_closing_the_stream_mid_answer_stores_it_cancelled_and_reopening_says_so(
    open_page: OpenPage, client_id: str, api: HttpChatApi
) -> None:
    conversation_id = api.create_conversation().id
    with api.stream_answer(conversation_id, IN_SCOPE) as events:
        first = next(e for e in events if isinstance(e, Token))
    assert first.text

    stored = settled(api, conversation_id)

    assert stored.outcome == "cancelled", (
        "the answer finished before the stream closed: the stack needs LLM_STUB_TOKEN_DELAY_MS > 0"
    )
    app = open_page(client_id, conversation_id)
    assert warnings_in(app) == [NOTICE_TEXT["cancelled"]]


def test_reloading_the_page_shows_the_same_thread(open_page: OpenPage, client_id: str) -> None:
    app = open_page(client_id)
    ask(app, IN_SCOPE)
    conversation_id = app.query_params["conv"]

    reloaded = open_page(client_id, conversation_id)

    assert reloaded.query_params["conv"] == conversation_id
    assert [str(m.name) for m in reloaded.chat_message] == ["user", "assistant"]
    assert [t.value for t in reloaded.chat_message[0].text] == [IN_SCOPE]
    assert answer_text(reloaded) == answer_text(app)


def test_another_client_cannot_open_the_conversation(open_page: OpenPage, api: HttpChatApi) -> None:
    conversation_id = api.create_conversation().id
    other_client = str(uuid.uuid4())
    with pytest.raises(ApiError) as raised:
        HttpChatApi(FrontendSettings(), other_client).get_conversation(conversation_id)
    assert (raised.value.code, raised.value.status) == ("conversation_not_found", 404)

    app = open_page(other_client, conversation_id)

    assert "conv" not in app.query_params
    assert [str(w.value) for w in app.sidebar.warning] == [ERROR_TEXT["conversation_not_found"]]
    assert not app.chat_message

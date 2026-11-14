"""View models: one MessageView for an answer, whether it was streamed or reloaded."""

from datetime import UTC, datetime
from typing import Literal

import pytest

from hpdocs_ui.api import (
    Conversation,
    ConversationDetail,
    Done,
    Flags,
    Message,
    Outcome,
    Pages,
    RefusalReason,
    Source,
    SourcesEvent,
    StreamEvent,
    Token,
)
from hpdocs_ui.chat_view import (
    NOTICE_TEXT,
    REFUSAL_TEXT,
    UNKNOWN_REFUSAL_TEXT,
    UNTITLED,
    WARNING_BADGE,
    ErrorView,
    MessageView,
    SourceView,
    answer_from_stream,
    conversation_label,
    error_view,
    message_view,
    page_label,
    source_heading,
    source_view,
    thread_view,
)
from hpdocs_ui.errors import ApiError
from hpdocs_ui.sanitize import escape_markdown

NOW = datetime(2026, 11, 14, 9, 0, tzinfo=UTC)
# What the backend streams and stores for every refusal; the page never shows it.
BACKEND_REFUSAL = "BACKEND-REFUSAL-TEXT"


def pages(
    pdf: tuple[int, int] = (7, 7), printed: tuple[int | None, int | None] = (None, None)
) -> Pages:
    return Pages(pdf_start=pdf[0], pdf_end=pdf[1], printed_start=printed[0], printed_end=printed[1])


def source(tag: str = "S1", *, has_warning: bool = False) -> Source:
    return Source(
        tag=tag,
        document_id="omen-17-mg",
        document_title="OMEN 17 Maintenance and Service Guide",
        section_path=("Removal and replacement", "Battery"),
        pages=pages((50, 51), (48, 49)),
        has_warning=has_warning,
        snippet="Disconnect the battery cable [S2] from the system board.",
    )


def flags(*, uncited: bool = False) -> Flags:
    return Flags(
        late_refusal_marker=False,
        uncited=uncited,
        invalid_citations=False,
        rewrite_fallback=False,
        truncated=False,
    )


def done(
    *,
    outcome: Outcome = "completed",
    content: str = "Remove the battery [S1].",
    citations: tuple[str, ...] = ("S1",),
    refusal_reason: RefusalReason | None = None,
    uncited: bool = False,
) -> Done:
    return Done(
        conversation_id="c",
        user_message_id="u",
        message_id="m",
        outcome=outcome,
        refusal=outcome == "refused",
        refusal_reason=refusal_reason,
        content=content,
        citations=citations,
        flags=flags(uncited=uncited),
    )


def message(  # noqa: PLR0913 - the fields of a stored message that vary by case
    *,
    role: str = "assistant",
    outcome: Outcome | None = "completed",
    content: str = "Remove the battery [S1].",
    citations: tuple[str, ...] = ("S1",),
    refusal_reason: RefusalReason | None = None,
    sources: tuple[Source, ...] = (),
) -> Message:
    return Message.model_validate(
        {
            "id": "m",
            "role": role,
            "content": content,
            "created_at": NOW,
            "outcome": outcome,
            "refusal": outcome == "refused",
            "refusal_reason": refusal_reason,
            "citations": citations,
            "sources": sources,
        }
    )


def tokens(*texts: str) -> list[StreamEvent]:
    return [Token(text=text) for text in texts]


def mid_stream(code: str) -> ApiError:
    return ApiError(code, status=200, request_id="req-12345678")


# --- page labels -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "label"),
    [
        (pages((44, 44), (42, 42)), "p. 42"),
        (pages((44, 45), (42, 43)), "pp. 42\u201343"),
        (pages((44, 45), (42, None)), "p. 42"),
        (pages((7, 7)), "PDF page 7"),
        (pages((7, 8)), "PDF pages 7\u20138"),
        (pages((7, 8), (None, 9)), "PDF pages 7\u20138"),
    ],
)
def test_the_page_label_prefers_printed_pages_and_falls_back_to_pdf_pages(
    value: Pages, label: str
) -> None:
    assert page_label(value) == label


# --- sources ---------------------------------------------------------------------


def test_a_source_view_carries_the_fields_the_page_shows() -> None:
    view = source_view(source("S3", has_warning=True))

    assert view == SourceView(
        tag="S3",
        document_title="OMEN 17 Maintenance and Service Guide",
        section="Removal and replacement \u203a Battery",
        pages="pp. 48\u201349",
        has_warning=True,
        snippet="Disconnect the battery cable [S2] from the system board.",
    )


def test_a_source_with_no_section_path_has_an_empty_section() -> None:
    bare = source().model_copy(update={"section_path": ()})

    assert source_view(bare).section == ""


def test_the_warning_badge_names_both_kinds_of_note() -> None:
    assert WARNING_BADGE == "WARNING/CAUTION"


# --- fixed wording ---------------------------------------------------------------


def test_the_refusal_wording_is_fixed_per_reason() -> None:
    assert REFUSAL_TEXT == {
        "below_threshold": (
            "I couldn't find this in the HP ENVY 6000 or OMEN 17 manuals, so I can't answer it."
        ),
        "model_refused": (
            "The manual excerpts I found don't answer this question, "
            "so I can't answer it from the documents."
        ),
        "provider_refused": "The model declined to answer this question.",
    }


def test_the_notice_wording_is_fixed_per_state() -> None:
    assert NOTICE_TEXT == {
        "uncited": "This answer cites no source from the manuals; check it against the documents.",
        "failed": "This answer was interrupted and is incomplete.",
        "cancelled": "This answer was stopped before it finished.",
        "pending": (
            "An answer is still being written for this conversation, or was interrupted. "
            "Refresh to see it."
        ),
    }


def test_a_refusal_without_a_reason_gets_a_neutral_text() -> None:
    view = message_view(message(outcome="refused", content=BACKEND_REFUSAL, citations=()))

    assert view.text == UNKNOWN_REFUSAL_TEXT == "I can't answer this question from the manuals."
    assert view.state == "refused"


# --- a reloaded thread (GET) -----------------------------------------------------


def test_a_user_message_shows_its_text_with_no_state_notice_or_sources() -> None:
    view = message_view(message(role="user", outcome=None, content="How do I scan?", citations=()))

    assert view == MessageView(role="user", text="How do I scan?", state=None, sources=())
    assert view.notice is None


def test_a_completed_answer_keeps_its_text_and_sources_in_backend_order() -> None:
    stored = message(citations=("S2", "S1"), sources=(source("S2"), source("S1")))

    view = message_view(stored)

    assert (view.role, view.text, view.state, view.notice) == (
        "assistant",
        "Remove the battery [S1].",
        "answered",
        None,
    )
    assert [s.tag for s in view.sources] == ["S2", "S1"]


def test_a_completed_answer_with_no_citations_is_uncited_after_a_reload() -> None:
    view = message_view(message(content="Some answer.", citations=()))

    assert (view.text, view.state) == ("Some answer.", "uncited")
    assert view.notice == NOTICE_TEXT["uncited"]


@pytest.mark.parametrize("reason", ["below_threshold", "model_refused", "provider_refused"])
def test_a_refusal_shows_the_page_wording_never_the_stored_text(reason: RefusalReason) -> None:
    view = message_view(
        message(outcome="refused", content=BACKEND_REFUSAL, citations=(), refusal_reason=reason)
    )

    assert (view.text, view.state, view.sources, view.notice) == (
        REFUSAL_TEXT[reason],
        "refused",
        (),
        None,
    )


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_an_interrupted_answer_keeps_its_partial_text_with_a_notice(
    outcome: Literal["failed", "cancelled"],
) -> None:
    view = message_view(message(outcome=outcome, content="Open the", citations=()))

    assert (view.text, view.state, view.notice) == ("Open the", outcome, NOTICE_TEXT[outcome])


@pytest.mark.parametrize("outcome", ["pending", None])
def test_an_answer_still_pending_is_empty_with_the_refresh_notice(outcome: Outcome | None) -> None:
    view = message_view(message(outcome=outcome, content="", citations=()))

    assert (view.text, view.state, view.notice) == ("", "pending", NOTICE_TEXT["pending"])


def test_a_thread_view_keeps_the_message_order() -> None:
    detail = ConversationDetail(
        id="c",
        title="Battery",
        created_at=NOW,
        updated_at=NOW,
        messages=(
            message(role="user", outcome=None, content="Q1", citations=()),
            message(content="A1", citations=()),
            message(role="user", outcome=None, content="Q2", citations=()),
        ),
    )

    assert [(v.role, v.text) for v in thread_view(detail)] == [
        ("user", "Q1"),
        ("assistant", "A1"),
        ("user", "Q2"),
    ]


def test_an_empty_thread_has_no_views() -> None:
    detail = ConversationDetail(id="c", title=None, created_at=NOW, updated_at=NOW, messages=())

    assert thread_view(detail) == ()


# --- a streamed answer -----------------------------------------------------------


def test_a_streamed_answer_shows_the_final_content_not_the_streamed_tokens() -> None:
    events = [*tokens("Remove the ", "battery [S9]."), SourcesEvent(sources=(source(),)), done()]

    view = answer_from_stream(events, None)

    assert view is not None
    assert (view.role, view.text, view.state) == (
        "assistant",
        "Remove the battery [S1].",
        "answered",
    )
    assert view.sources == (source_view(source()),)


def test_on_the_stream_the_uncited_flag_decides_not_the_citations() -> None:
    flagged = answer_from_stream([SourcesEvent(sources=()), done(uncited=True)], None)
    unflagged = answer_from_stream(
        [SourcesEvent(sources=()), done(citations=(), uncited=False)], None
    )

    assert flagged is not None
    assert unflagged is not None
    assert (flagged.state, unflagged.state) == ("uncited", "answered")


@pytest.mark.parametrize("reason", ["below_threshold", "model_refused", "provider_refused"])
def test_a_streamed_refusal_shows_the_page_wording(reason: RefusalReason) -> None:
    events = [
        *tokens(BACKEND_REFUSAL),
        SourcesEvent(sources=()),
        done(outcome="refused", content=BACKEND_REFUSAL, citations=(), refusal_reason=reason),
    ]

    view = answer_from_stream(events, None)

    assert view is not None
    assert (view.text, view.state) == (REFUSAL_TEXT[reason], "refused")


@pytest.mark.parametrize("status", [None, 404, 409, 422, 429, 500, 502, 503])
def test_a_failure_before_the_stream_started_leaves_no_answer(status: int | None) -> None:
    assert answer_from_stream([], ApiError("llm_unavailable", status=status)) is None


@pytest.mark.parametrize(
    "code",
    ["llm_error", "llm_timeout", "llm_unavailable", "internal_error", "backend_unreachable"],
)
def test_a_failure_mid_stream_keeps_the_partial_text_as_failed(code: str) -> None:
    view = answer_from_stream(tokens("Open the ", "cover"), mid_stream(code))

    assert view is not None
    assert (view.text, view.state, view.sources) == ("Open the cover", "failed", ())


def test_sources_received_before_a_mid_stream_failure_are_not_shown() -> None:
    events = [*tokens("Open the cover"), SourcesEvent(sources=(source(),))]

    view = answer_from_stream(events, mid_stream("bad_response"))

    assert view is not None
    assert (view.text, view.state, view.sources) == ("Open the cover", "failed", ())


def test_a_stream_stopped_by_the_total_deadline_is_cancelled() -> None:
    view = answer_from_stream(tokens("Open the"), mid_stream("stream_timeout"))

    assert view is not None
    assert (view.text, view.state) == ("Open the", "cancelled")


def test_a_withdrawn_answer_replaces_the_partial_text_with_the_provider_refusal() -> None:
    view = answer_from_stream(tokens("Partial answer"), mid_stream("answer_withdrawn"))

    assert view is not None
    assert (view.text, view.state, view.sources) == (
        REFUSAL_TEXT["provider_refused"],
        "refused",
        (),
    )


@pytest.mark.parametrize("events", [[], tokens("x"), [*tokens("x"), SourcesEvent(sources=())]])
def test_a_stream_that_ended_without_done_or_error_is_a_programming_error(
    events: list[StreamEvent],
) -> None:
    with pytest.raises(ValueError, match="done"):
        answer_from_stream(events, None)


# --- parity: the same answer streamed and reloaded (adjustment C) ----------------


PARITY_CASES: dict[str, tuple[list[StreamEvent], ApiError | None, Message]] = {
    "completed": (
        [*tokens("Remove the ", "battery [S1]."), SourcesEvent(sources=(source(),)), done()],
        None,
        message(sources=(source(),)),
    ),
    "refused": (
        [
            *tokens(BACKEND_REFUSAL),
            SourcesEvent(sources=()),
            done(
                outcome="refused",
                content=BACKEND_REFUSAL,
                citations=(),
                refusal_reason="below_threshold",
            ),
        ],
        None,
        message(
            outcome="refused",
            content=BACKEND_REFUSAL,
            citations=(),
            refusal_reason="below_threshold",
        ),
    ),
    "uncited": (
        [
            *tokens("An answer."),
            SourcesEvent(sources=()),
            done(content="An answer.", citations=(), uncited=True),
        ],
        None,
        message(content="An answer.", citations=()),
    ),
    "failed": (
        tokens("Open the ", "cover"),
        mid_stream("llm_error"),
        message(outcome="failed", content="Open the cover", citations=()),
    ),
    "withdrawn": (
        tokens("Partial"),
        mid_stream("answer_withdrawn"),
        message(
            outcome="refused",
            content=BACKEND_REFUSAL,
            citations=(),
            refusal_reason="provider_refused",
        ),
    ),
    "cancelled": (
        tokens("Open the"),
        mid_stream("stream_timeout"),
        message(outcome="cancelled", content="Open the", citations=()),
    ),
}


@pytest.mark.parametrize("case", PARITY_CASES)
def test_a_streamed_answer_and_its_reload_give_the_same_view(case: str) -> None:
    events, error, stored = PARITY_CASES[case]

    assert answer_from_stream(events, error) == message_view(stored)


# --- error view ------------------------------------------------------------------


def test_an_error_view_has_the_fixed_text_and_the_trusted_request_id() -> None:
    view = error_view(ApiError("llm_timeout", status=503, request_id="req-12345678"))

    assert view == ErrorView(
        text="The language model took too long to answer. Try again.",
        request_id="req-12345678",
        question=None,
    )


def test_an_error_view_drops_a_malformed_request_id() -> None:
    assert error_view(ApiError("llm_error", request_id="<b>bad</b>")).request_id is None


def test_an_error_view_keeps_the_question_so_it_can_be_resent() -> None:
    view = error_view(ApiError("rate_limited", status=429, retry_after_seconds=3), question="Q?")

    assert (view.text, view.question) == (
        "You're sending requests too quickly. Try again in 3 seconds.",
        "Q?",
    )


# --- conversation labels ---------------------------------------------------------


def _titled(title: str | None) -> Conversation:
    when = datetime(2026, 11, 14, tzinfo=UTC)
    return Conversation(id="c", title=title, created_at=when, updated_at=when)


def test_a_conversation_label_is_its_title_escaped_for_a_button() -> None:
    title = "How do I *scan* [S1]?"

    assert conversation_label(_titled(title)) == escape_markdown(title)


@pytest.mark.parametrize("title", [None, "", " \n\t "])
def test_a_conversation_without_a_title_has_a_fixed_label(title: str | None) -> None:
    assert conversation_label(_titled(title)) == UNTITLED == "Untitled conversation"


def test_a_source_heading_names_tag_document_section_and_pages() -> None:
    view = SourceView(
        tag="S2",
        document_title="OMEN 17 Service Guide",
        section="Removal \u203a Battery",
        pages="p. 54",
        has_warning=False,
        snippet="",
    )

    assert source_heading(view) == "[S2] OMEN 17 Service Guide \u203a Removal \u203a Battery, p. 54"


def test_a_source_heading_without_a_section_keeps_document_and_pages() -> None:
    view = SourceView(
        tag="S1",
        document_title="Guide",
        section="",
        pages="PDF page 3",
        has_warning=True,
        snippet="",
    )

    assert source_heading(view) == "[S1] Guide, PDF page 3"

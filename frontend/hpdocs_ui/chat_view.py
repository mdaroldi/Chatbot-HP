"""What the page shows for each message, the same whether it was streamed or reloaded.

An answer that streams in and the same answer read back with GET after a reload give
one MessageView, so a rerun or a reload never changes what the user sees. The thread
read has no flags, so "uncited" comes from the rule the backend applies when it
stores an answer: a completed answer with no valid citation (ADR 0011).

Every refusal and notice text is fixed here; the backend's own refusal text is never
shown. Backend text that is shown (answers, titles, snippets) is untrusted and is made
safe where it is rendered, not here.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from hpdocs_ui.api import (
    ConversationDetail,
    Done,
    Message,
    Outcome,
    Pages,
    RefusalReason,
    Source,
    SourcesEvent,
    StreamEvent,
    Token,
)
from hpdocs_ui.errors import STREAM_TIMEOUT, ApiError, error_text, shown_request_id

AnswerState = Literal["answered", "uncited", "refused", "pending", "failed", "cancelled"]

REFUSAL_TEXT: Mapping[RefusalReason, str] = {
    "below_threshold": (
        "I couldn't find this in the HP ENVY 6000 or OMEN 17 manuals, so I can't answer it."
    ),
    "model_refused": (
        "The manual excerpts I found don't answer this question, "
        "so I can't answer it from the documents."
    ),
    "provider_refused": "The model declined to answer this question.",
}
UNKNOWN_REFUSAL_TEXT = "I can't answer this question from the manuals."

NOTICE_TEXT: Mapping[AnswerState, str] = {
    "uncited": "This answer cites no source from the manuals; check it against the documents.",
    "failed": "This answer was interrupted and is incomplete.",
    "cancelled": "This answer was stopped before it finished.",
    "pending": (
        "An answer is still being written for this conversation, or was interrupted. "
        "Refresh to see it."
    ),
}

WARNING_BADGE = "WARNING/CAUTION"
# Typographic separators, written as escapes so the source stays ASCII.
PAGE_RANGE_DASH = "\u2013"  # en dash
SECTION_SEPARATOR = " \u203a "  # single right-pointing angle quotation mark

# A provider refusal after text was shown; the backend stores it as provider_refused.
_ANSWER_WITHDRAWN = "answer_withdrawn"
_STREAM_OK = 200


@dataclass(frozen=True)
class SourceView:
    tag: str
    document_title: str
    section: str
    pages: str
    has_warning: bool
    snippet: str


@dataclass(frozen=True)
class MessageView:
    role: Literal["user", "assistant"]
    text: str
    state: AnswerState | None
    sources: tuple[SourceView, ...]

    @property
    def notice(self) -> str | None:
        return None if self.state is None else NOTICE_TEXT.get(self.state)


@dataclass(frozen=True)
class ErrorView:
    """A failed send. The question is kept here, not in the thread, when nothing was
    stored, so it can be copied and sent again.
    """

    text: str
    request_id: str | None
    question: str | None


def page_label(pages: Pages) -> str:
    """Printed page numbers match the manual a reader holds; PDF pages are the fallback."""
    if pages.printed_start is not None:
        start = pages.printed_start
        end = pages.printed_end if pages.printed_end is not None else start
        return f"p. {start}" if start == end else f"pp. {start}{PAGE_RANGE_DASH}{end}"
    if pages.pdf_start == pages.pdf_end:
        return f"PDF page {pages.pdf_start}"
    return f"PDF pages {pages.pdf_start}{PAGE_RANGE_DASH}{pages.pdf_end}"


def source_view(source: Source) -> SourceView:
    return SourceView(
        tag=source.tag,
        document_title=source.document_title,
        section=SECTION_SEPARATOR.join(source.section_path),
        pages=page_label(source.pages),
        has_warning=source.has_warning,
        snippet=source.snippet,
    )


def message_view(message: Message) -> MessageView:
    if message.role == "user":
        return MessageView(role="user", text=message.content, state=None, sources=())
    outcome = message.outcome or "pending"
    uncited = outcome == "completed" and not message.citations
    return _answer(outcome, message.refusal_reason, message.content, message.sources, uncited)


def thread_view(detail: ConversationDetail) -> tuple[MessageView, ...]:
    return tuple(message_view(message) for message in detail.messages)


def answer_from_stream(events: Sequence[StreamEvent], error: ApiError | None) -> MessageView | None:
    """The answer as the backend stored it, from the events read and how reading ended.

    A failure before the stream started (any status but 200) stored nothing, so there
    is no answer. A failure after it stored what was shown: a withdrawn answer as a
    provider refusal, a stream the page stopped as cancelled, anything else as failed.
    """
    if error is not None:
        if error.status != _STREAM_OK:
            return None
        partial = "".join(event.text for event in events if isinstance(event, Token))
        if error.code == _ANSWER_WITHDRAWN:
            return _answer("refused", "provider_refused", partial, (), uncited=False)
        outcome: Outcome = "cancelled" if error.code == STREAM_TIMEOUT else "failed"
        return _answer(outcome, None, partial, (), uncited=False)
    final = events[-1] if events else None
    if not isinstance(final, Done):
        raise ValueError("a stream that did not fail must end with done")  # noqa: TRY004 - a broken call contract, not a wrong argument type
    sources = next((e.sources for e in events if isinstance(e, SourcesEvent)), ())
    return _answer(final.outcome, final.refusal_reason, final.content, sources, final.flags.uncited)


def error_view(error: ApiError, *, question: str | None = None) -> ErrorView:
    return ErrorView(
        text=error_text(error), request_id=shown_request_id(error.request_id), question=question
    )


def _answer(
    outcome: Outcome,
    reason: RefusalReason | None,
    content: str,
    sources: tuple[Source, ...],
    uncited: bool,
) -> MessageView:
    if outcome == "refused":
        text = UNKNOWN_REFUSAL_TEXT if reason is None else REFUSAL_TEXT[reason]
        return MessageView(role="assistant", text=text, state="refused", sources=())
    state: AnswerState = "uncited" if uncited else _STATE[outcome]
    views = tuple(source_view(source) for source in sources)
    return MessageView(role="assistant", text=content, state=state, sources=views)


_STATE: Mapping[Outcome, AnswerState] = {
    "completed": "answered",
    "pending": "pending",
    "failed": "failed",
    "cancelled": "cancelled",
    "refused": "refused",
}

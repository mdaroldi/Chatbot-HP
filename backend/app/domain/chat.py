"""Chat types (pure, no I/O): conversations, messages and how an answer ended.

The rules here are the ones the API and the database both rely on, so a message that
breaks them cannot be built: a user message carries no answer fields, a refusal always
names its reason and never cites, and only finished answers become history.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from app.domain.ingestion import PageRange

# The system prompt tells the model to reply with exactly this when the excerpts do not
# hold the answer; the stream detector hides it from the user (layer 2 refusal).
REFUSAL_MARKER = "[[NOT_IN_DOCUMENTS]]"

TITLE_MAX_CHARS = 80

_SOURCE_TAG = re.compile(r"S[1-9][0-9]*")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Outcome(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    REFUSED = "refused"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RefusalReason(StrEnum):
    BELOW_THRESHOLD = "below_threshold"
    MODEL_REFUSED = "model_refused"
    PROVIDER_REFUSED = "provider_refused"


@dataclass(frozen=True)
class Source:
    tag: str
    doc_id: str
    document_title: str
    section_path: tuple[str, ...]
    pages: PageRange
    has_warning: bool
    snippet: str

    def __post_init__(self) -> None:
        _require(bool(_SOURCE_TAG.fullmatch(self.tag)), "source tag must look like S1")


@dataclass(frozen=True)
class Conversation:
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require(
            self.title is None or 1 <= len(self.title) <= TITLE_MAX_CHARS,
            f"title must be 1 to {TITLE_MAX_CHARS} characters",
        )
        _require(self.updated_at >= self.created_at, "updated_at must not precede created_at")


@dataclass(frozen=True)
class Message:
    id: str
    conversation_id: str
    seq: int
    role: Role
    content: str
    created_at: datetime
    outcome: Outcome | None = None
    refusal_reason: RefusalReason | None = None
    citations: tuple[str, ...] = ()
    sources: tuple[Source, ...] = ()

    def __post_init__(self) -> None:
        _require(self.seq >= 1, "seq must be at least 1")
        if self.role is Role.USER:
            _require(
                self.outcome is None and self.refusal_reason is None, "user message has no outcome"
            )
            _require(
                not self.citations and not self.sources, "user message has no citations or sources"
            )
            return
        _require(self.outcome is not None, "assistant message needs an outcome")
        refused = self.outcome is Outcome.REFUSED
        _require(not refused or self.refusal_reason is not None, "a refusal needs a reason")
        _require(refused or self.refusal_reason is None, "only a refusal has a reason")
        _require(
            not refused or (not self.citations and not self.sources),
            "a refusal has no citations or sources",
        )

    @property
    def refusal(self) -> bool:
        return self.refusal_reason is not None

    @property
    def counts_as_turn(self) -> bool:
        """Whether the prompt history includes this message: stored refusals do, while
        partial (failed, cancelled) or unfinished answers would mislead the model.
        """
        return self.role is Role.USER or self.outcome in (Outcome.COMPLETED, Outcome.REFUSED)

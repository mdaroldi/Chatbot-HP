"""Golden set types for the retrieval benchmark (pure, no I/O).

The golden set is written from the PDFs, not from the index, and reviewed outside the
code: the loader validates its structure but never changes a question or an anchor.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class Category(StrEnum):
    FACTUAL = "factual"
    PROCEDURE = "procedure"
    TABLE = "table"
    PART_NUMBER = "part_number"
    WARNING = "warning"
    PREREQUISITE = "prerequisite"
    AMBIGUOUS = "ambiguous"
    FOLLOW_UP = "follow_up"
    OUT_OF_SCOPE = "out_of_scope"


class Split(StrEnum):
    DEV = "dev"
    HELDOUT = "heldout"


class Match(StrEnum):
    ANY = "any"
    ALL = "all"


class OosKind(StrEnum):
    UNRELATED = "unrelated"
    NEAR_DOMAIN = "near_domain"


class GoldenRule(StrEnum):
    UNREADABLE = "unreadable"
    INVALID_TOML = "invalid_toml"
    UNKNOWN_KEY = "unknown_key"
    MISSING_KEY = "missing_key"
    WRONG_TYPE = "wrong_type"
    INVALID_VERSION = "invalid_version"
    NO_QUESTIONS = "no_questions"
    INVALID_ID = "invalid_id"
    DUPLICATE_ID = "duplicate_id"
    UNKNOWN_CATEGORY = "unknown_category"
    UNKNOWN_SPLIT = "unknown_split"
    UNKNOWN_MATCH = "unknown_match"
    EMPTY_QUESTION = "empty_question"
    MISSING_ANCHORS = "missing_anchors"
    UNEXPECTED_ANCHORS = "unexpected_anchors"
    UNKNOWN_DOC_ID = "unknown_doc_id"
    EMPTY_FACTS = "empty_facts"
    BLANK_FACT = "blank_fact"
    FACT_TOO_LONG = "fact_too_long"
    BLANK_SECTION = "blank_section"
    MISSING_CONTEXT = "missing_context"
    UNEXPECTED_CONTEXT = "unexpected_context"
    MISSING_OOS_KIND = "missing_oos_kind"
    UNKNOWN_OOS_KIND = "unknown_oos_kind"
    UNEXPECTED_OOS_KIND = "unexpected_oos_kind"
    HELDOUT_SHARE = "heldout_share"
    HELDOUT_MISSING_IN_CATEGORY = "heldout_missing_in_category"


class GoldenSetError(Exception):
    """The golden set breaks one structural rule; `rule` says which one.

    Messages name question ids and positions, never question wording or facts, so a
    report or a log line cannot leak more of the set than its identifiers.
    """

    def __init__(self, rule: GoldenRule, message: str) -> None:
        super().__init__(message)
        self.rule = rule


@dataclass(frozen=True)
class Anchor:
    """A fact the answer depends on: every fact must occur in one chunk of `doc_id`."""

    doc_id: str
    facts: tuple[str, ...]
    section: str | None = None


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    category: Category
    split: Split
    question: str
    match: Match | None
    anchors: tuple[Anchor, ...]
    context: str | None = None
    oos_kind: OosKind | None = None

    @property
    def in_scope(self) -> bool:
        """Out-of-scope questions are not ranked; they calibrate the relevance threshold."""
        return self.category is not Category.OUT_OF_SCOPE


@dataclass(frozen=True)
class GoldenSet:
    version: str
    sha256: str
    questions: tuple[GoldenQuestion, ...]

    def in_split(self, split: Split) -> tuple[GoldenQuestion, ...]:
        return tuple(q for q in self.questions if q.split is split)


@dataclass(frozen=True)
class IndexedChunk:
    """What anchor matching reads from one indexed chunk (ids, path and text)."""

    chunk_id: str
    doc_id: str
    section_path: tuple[str, ...]
    text: str


class Metric(StrEnum):
    RECALL_AT_5 = "recall@5"
    RECALL_AT_20 = "recall@20"
    HIT_AT_5 = "hit@5"
    MRR = "mrr"
    NDCG_AT_5 = "ndcg@5"


@dataclass(frozen=True)
class QuestionScores:
    question_id: str
    category: Category
    values: Mapping[Metric, float]

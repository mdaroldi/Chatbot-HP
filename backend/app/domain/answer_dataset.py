"""Answer-quality dataset types for the Phase 5 benchmark (pure, no I/O).

Two files sit next to the frozen retrieval set: `answers-v1.toml` gives each retrieval
question its key facts, and `slices-v1.toml` adds multi-turn, long, out-of-scope and
adversarial items. Both are written from the PDFs and reviewed outside the code: the
loader validates them but never changes a question, a key fact, an anchor or a pattern.
"""

from dataclasses import dataclass
from enum import StrEnum

from app.domain.evaluation import Anchor, Category, Match, OosKind, Split

# One key-fact group: alternative strings, any of which satisfies the group. A string
# starting with `REGEX_PREFIX` is a regular expression on the normalized answer.
KeyFactGroup = tuple[str, ...]
REGEX_PREFIX = "re:"


class Slice(StrEnum):
    MULTI_TURN = "multi_turn"
    LONG_QUERY = "long_query"
    OUT_OF_SCOPE = "out_of_scope"
    ADVERSARIAL = "adversarial"


class Attack(StrEnum):
    INJECTION_DIRECT = "injection_direct"
    INJECTION_EMBEDDED = "injection_embedded"
    PROMPT_LEAK = "prompt_leak"
    ROLEPLAY = "roleplay"
    FORMAT_HIJACK = "format_hijack"


class Expected(StrEnum):
    REFUSE = "refuse"
    ANSWER_IN_SCOPE = "answer_in_scope"


class DatasetRule(StrEnum):
    UNREADABLE = "unreadable"
    INVALID_TOML = "invalid_toml"
    UNKNOWN_KEY = "unknown_key"
    MISSING_KEY = "missing_key"
    WRONG_TYPE = "wrong_type"
    INVALID_VERSION = "invalid_version"
    NO_QUESTIONS = "no_questions"
    INVALID_ID = "invalid_id"
    DUPLICATE_ID = "duplicate_id"
    UNKNOWN_VALUE = "unknown_value"
    EMPTY_GROUP = "empty_group"
    BLANK_FACT = "blank_fact"
    FACT_TOO_LONG = "fact_too_long"
    INVALID_PATTERN = "invalid_pattern"
    PATTERN_NOT_LOWERCASE = "pattern_not_lowercase"
    WARNING_FACTS = "warning_facts"
    TURN_COUNT = "turn_count"
    BLANK_TURN = "blank_turn"
    TURN_TOO_LONG = "turn_too_long"
    INVALID_ANCHOR = "invalid_anchor"
    UNKNOWN_ID = "unknown_id"
    MISSING_ENTRY = "missing_entry"
    SPLIT_MISMATCH = "split_mismatch"
    KEY_FACTS_SCOPE = "key_facts_scope"
    UNRESOLVED_PAIR = "unresolved_pair"
    PAIR_OUT_OF_SCOPE = "pair_out_of_scope"


class DatasetError(Exception):
    """A dataset file breaks one rule; `rule` says which one.

    Messages name files, ids, keys and positions only, never question wording, key facts
    or patterns, so a log line cannot leak more of the set than its identifiers.
    """

    def __init__(self, rule: DatasetRule, message: str) -> None:
        super().__init__(message)
        self.rule = rule


class DatasetNotFrozenError(Exception):
    """A dataset file is not the frozen one: no run on it counts as evidence."""


@dataclass(frozen=True)
class AnswerKey:
    """The key facts of one retrieval-v1 question (`answers-v1.toml`)."""

    id: str
    split: Split
    key_facts: tuple[KeyFactGroup, ...]  # empty for out-of-scope questions
    warning_facts: KeyFactGroup | None = None  # set exactly when a warning is expected

    @property
    def warning_expected(self) -> bool:
        return self.warning_facts is not None


@dataclass(frozen=True)
class AnswerFile:
    version: str
    sha256: str
    keys: tuple[AnswerKey, ...]


@dataclass(frozen=True)
class SliceItem:
    """One item of `slices-v1.toml`; only its slice's fields are set."""

    id: str
    slice: Slice
    split: Split
    turns: tuple[str, ...]  # the last one is evaluated; earlier ones are replayed
    anchors: tuple[Anchor, ...] = ()
    match: Match | None = None
    key_facts: tuple[KeyFactGroup, ...] = ()
    rewrite_trap: bool | None = None
    rewrite_forbidden: tuple[str, ...] = ()
    pair_of: str | None = None
    oos_kind: OosKind | None = None
    attack: Attack | None = None
    expected: Expected | None = None
    forbidden: tuple[str, ...] = ()


@dataclass(frozen=True)
class SliceFile:
    version: str
    sha256: str
    items: tuple[SliceItem, ...]


@dataclass(frozen=True)
class EvalItem:
    """What the harness sends and scores: an answers-v1 question (`slice` None) or a
    slice item, with long queries already carrying their pair's anchors and facts.
    """

    id: str
    split: Split
    turns: tuple[str, ...]
    in_scope: bool  # an answer is expected; otherwise a refusal
    slice: Slice | None = None
    category: Category | None = None  # the retrieval-v1 category (of the pair, for long queries)
    anchors: tuple[Anchor, ...] = ()
    match: Match | None = None
    key_facts: tuple[KeyFactGroup, ...] = ()
    warning_facts: KeyFactGroup | None = None
    oos_kind: OosKind | None = None
    attack: Attack | None = None
    expected: Expected | None = None
    forbidden: tuple[str, ...] = ()
    rewrite_trap: bool | None = None
    rewrite_forbidden: tuple[str, ...] = ()
    pair_of: str | None = None

    @property
    def primary(self) -> bool:
        """In the primary metric: a single-turn in-scope answers-v1 question (rule 5)."""
        return self.slice is None and self.in_scope


@dataclass(frozen=True)
class DatasetFile:
    """A file's identity, as run metadata and the freeze gate read it."""

    name: str
    version: str
    sha256: str


@dataclass(frozen=True)
class Dataset:
    files: tuple[DatasetFile, ...]  # retrieval-v1, answers-v1, slices-v1
    items: tuple[EvalItem, ...]

    def in_split(self, split: Split) -> tuple[EvalItem, ...]:
        return tuple(item for item in self.items if item.split is split)

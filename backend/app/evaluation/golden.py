"""Golden set loader and structural validator (Phase 2 plan, golden set contract).

The file is an allowlist-style input, like the corpus manifest: anything unexpected
fails with the one rule it breaks. The set is never repaired here.
"""

import hashlib
import re
import tomllib
from collections.abc import Mapping, Set
from fractions import Fraction
from pathlib import Path

from app.domain.evaluation import (
    Anchor,
    Category,
    GoldenQuestion,
    GoldenRule,
    GoldenSet,
    GoldenSetError,
    Match,
    OosKind,
    Split,
)

# Short facts only: this also keeps verbatim HP text out of the repository.
MAX_FACT_CHARS = 80
HELDOUT_MIN_SHARE = Fraction(15, 100)
HELDOUT_MAX_SHARE = Fraction(25, 100)

_VERSION = re.compile(r"\d+\.\d+\.\d+")
_ID = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
_TOP_KEYS = frozenset({"version", "questions"})
_REQUIRED = ("id", "category", "split", "question")
_QUESTION_KEYS = frozenset({*_REQUIRED, "match", "anchors", "context", "oos_kind"})
_ANCHOR_KEYS = frozenset({"doc_id", "section", "facts"})


def _fail(rule: GoldenRule, message: str) -> GoldenSetError:
    return GoldenSetError(rule, message)


def _check_keys(raw: Mapping[str, object], allowed: Set[str], where: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise _fail(GoldenRule.UNKNOWN_KEY, f"{where}: unknown key: {', '.join(unknown)}")


def _string(raw: Mapping[str, object], key: str, where: str) -> str:
    value = raw[key]
    if not isinstance(value, str):
        raise _fail(GoldenRule.WRONG_TYPE, f"{where}: {key} must be a string")
    return value


def _enum[E: (Category, Split, Match, OosKind)](
    enum: type[E], raw: Mapping[str, object], key: str, rule: GoldenRule, where: str
) -> E:
    value = _string(raw, key, where)
    try:
        return enum(value)
    except ValueError:
        raise _fail(rule, f"{where}: unknown {key} {value!r}") from None


def _table(raw: object, where: str) -> Mapping[str, object]:
    if not isinstance(raw, dict):
        raise _fail(GoldenRule.WRONG_TYPE, f"{where} must be a table")
    return raw


def _anchor(raw: object, where: str, doc_ids: Set[str]) -> Anchor:
    table = _table(raw, where)
    _check_keys(table, _ANCHOR_KEYS, where)
    for key in ("doc_id", "facts"):
        if key not in table:
            raise _fail(GoldenRule.MISSING_KEY, f"{where}: missing key: {key}")
    doc_id = _string(table, "doc_id", where)
    if doc_id not in doc_ids:
        raise _fail(GoldenRule.UNKNOWN_DOC_ID, f"{where}: unknown doc_id {doc_id!r}")
    facts = _facts(table["facts"], where)
    return Anchor(doc_id=doc_id, facts=facts, section=_section(table, where))


def _facts(raw: object, where: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(fact, str) for fact in raw):
        raise _fail(GoldenRule.WRONG_TYPE, f"{where}: facts must be a list of strings")
    if not raw:
        raise _fail(GoldenRule.EMPTY_FACTS, f"{where}: facts must not be empty")
    for position, fact in enumerate(raw):
        if not fact.strip():
            raise _fail(GoldenRule.BLANK_FACT, f"{where}: facts[{position}] is blank")
        if len(fact) > MAX_FACT_CHARS:
            raise _fail(
                GoldenRule.FACT_TOO_LONG,
                f"{where}: facts[{position}] has {len(fact)} characters, max {MAX_FACT_CHARS}",
            )
    return tuple(raw)


def _section(table: Mapping[str, object], where: str) -> str | None:
    if "section" not in table:
        return None
    section = _string(table, "section", where)
    if not section.strip():
        raise _fail(GoldenRule.BLANK_SECTION, f"{where}: section is blank")
    return section


def _identity(table: Mapping[str, object], position: int) -> str:
    where = f"questions[{position}]"
    for key in _REQUIRED:
        if key not in table:
            raise _fail(GoldenRule.MISSING_KEY, f"{where}: missing key: {key}")
    qid = _string(table, "id", where)
    if not _ID.fullmatch(qid):
        raise _fail(GoldenRule.INVALID_ID, f"{where}: id must be a lowercase slug")
    return qid


def _match(table: Mapping[str, object], category: Category, where: str) -> Match | None:
    if "match" in table:
        return _enum(Match, table, "match", GoldenRule.UNKNOWN_MATCH, where)
    if category is not Category.OUT_OF_SCOPE:
        raise _fail(GoldenRule.MISSING_KEY, f"{where}: missing key: match")
    return None


def _context(table: Mapping[str, object], category: Category, where: str) -> str | None:
    if category is not Category.FOLLOW_UP:
        if "context" in table:
            raise _fail(GoldenRule.UNEXPECTED_CONTEXT, f"{where}: context is for follow_up only")
        return None
    context = _string(table, "context", where) if "context" in table else ""
    if not context.strip():
        raise _fail(GoldenRule.MISSING_CONTEXT, f"{where}: a follow_up needs its context")
    return context


def _oos_kind(table: Mapping[str, object], category: Category, where: str) -> OosKind | None:
    if category is not Category.OUT_OF_SCOPE:
        if "oos_kind" in table:
            raise _fail(
                GoldenRule.UNEXPECTED_OOS_KIND, f"{where}: oos_kind is for out_of_scope only"
            )
        return None
    if "oos_kind" not in table:
        raise _fail(GoldenRule.MISSING_OOS_KIND, f"{where}: out_of_scope needs oos_kind")
    return _enum(OosKind, table, "oos_kind", GoldenRule.UNKNOWN_OOS_KIND, where)


def _anchors(
    table: Mapping[str, object], category: Category, where: str, doc_ids: Set[str]
) -> tuple[Anchor, ...]:
    raw = table.get("anchors", [])
    if not isinstance(raw, list):
        raise _fail(GoldenRule.WRONG_TYPE, f"{where}: anchors must be a list of tables")
    if category is Category.OUT_OF_SCOPE:
        if raw:
            raise _fail(GoldenRule.UNEXPECTED_ANCHORS, f"{where}: out_of_scope has no anchors")
        return ()
    if not raw:
        raise _fail(GoldenRule.MISSING_ANCHORS, f"{where}: an in-scope question needs anchors")
    return tuple(_anchor(item, f"{where} anchors[{n}]", doc_ids) for n, item in enumerate(raw))


def _question(raw: object, position: int, doc_ids: Set[str]) -> GoldenQuestion:
    table = _table(raw, f"questions[{position}]")
    where = _identity(table, position)
    _check_keys(table, _QUESTION_KEYS, where)
    category = _enum(Category, table, "category", GoldenRule.UNKNOWN_CATEGORY, where)
    split = _enum(Split, table, "split", GoldenRule.UNKNOWN_SPLIT, where)
    text = _string(table, "question", where)
    if not text.strip():
        raise _fail(GoldenRule.EMPTY_QUESTION, f"{where}: question is blank")
    return GoldenQuestion(
        id=where,
        category=category,
        split=split,
        question=text,
        match=_match(table, category, where),
        anchors=_anchors(table, category, where, doc_ids),
        context=_context(table, category, where),
        oos_kind=_oos_kind(table, category, where),
    )


def _check_unique(questions: tuple[GoldenQuestion, ...]) -> None:
    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            raise _fail(GoldenRule.DUPLICATE_ID, f"duplicate id: {question.id}")
        seen.add(question.id)


def _check_heldout(questions: tuple[GoldenQuestion, ...]) -> None:
    """Held-out questions are scored only in Phase 5, so each category needs some."""
    held = sum(q.split is Split.HELDOUT for q in questions)
    share = Fraction(held, len(questions))
    if not HELDOUT_MIN_SHARE <= share <= HELDOUT_MAX_SHARE:
        raise _fail(
            GoldenRule.HELDOUT_SHARE,
            f"held-out share is {held}/{len(questions)}, allowed 15% to 25%",
        )
    for category in dict.fromkeys(q.category for q in questions):
        if not any(q.split is Split.HELDOUT and q.category is category for q in questions):
            raise _fail(
                GoldenRule.HELDOUT_MISSING_IN_CATEGORY,
                f"category {category.value} has no held-out question",
            )


def _version(data: Mapping[str, object]) -> str:
    if "version" not in data:
        raise _fail(GoldenRule.MISSING_KEY, "missing key: version")
    version = data["version"]
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise _fail(GoldenRule.INVALID_VERSION, "version must look like 1.0.0")
    return version


def parse_golden(data: bytes, doc_ids: Set[str]) -> GoldenSet:
    """Parse and validate the raw file bytes; the digest is taken over those bytes."""
    try:
        document = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise _fail(GoldenRule.INVALID_TOML, f"golden set is not valid TOML: {exc}") from exc
    _check_keys(document, _TOP_KEYS, "golden set")
    version = _version(document)
    raw_questions = document.get("questions")
    if raw_questions is None or raw_questions == []:
        raise _fail(GoldenRule.NO_QUESTIONS, "golden set has no questions")
    if not isinstance(raw_questions, list):
        raise _fail(GoldenRule.WRONG_TYPE, "questions must be a list of tables")
    questions = tuple(_question(raw, n, doc_ids) for n, raw in enumerate(raw_questions))
    _check_unique(questions)
    _check_heldout(questions)
    return GoldenSet(version, hashlib.sha256(data).hexdigest(), questions)


def load_golden(path: Path, doc_ids: Set[str]) -> GoldenSet:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise _fail(GoldenRule.UNREADABLE, f"cannot read golden set {path.name}") from exc
    return parse_golden(data, doc_ids)

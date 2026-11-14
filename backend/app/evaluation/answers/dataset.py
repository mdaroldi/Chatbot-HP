"""Loader and structural validator of the Phase 5 datasets (answers-v1 and slices-v1).

Both files are allowlist-style inputs, like the retrieval golden set: anything
unexpected fails with the one rule it breaks, and messages name files, ids, keys and
positions only. Neither file is ever repaired here. Cross-checks against the frozen
retrieval set: every answers entry matches one retrieval question (same id and split),
and every long query pairs with an in-scope retrieval question of its own split.
"""

import dataclasses
import hashlib
import re
import tomllib
from collections.abc import Mapping, Sequence, Set
from pathlib import Path

from app.domain.answer_dataset import (
    REGEX_PREFIX,
    AnswerFile,
    AnswerKey,
    Attack,
    Dataset,
    DatasetError,
    DatasetFile,
    DatasetRule,
    EvalItem,
    Expected,
    KeyFactGroup,
    Slice,
    SliceFile,
    SliceItem,
)
from app.domain.evaluation import (
    Anchor,
    GoldenQuestion,
    GoldenSet,
    GoldenSetError,
    Match,
    OosKind,
    Split,
)
from app.evaluation.golden import MAX_FACT_CHARS, load_golden, parse_anchor

RETRIEVAL_FILE = Path("golden") / "retrieval-v1.toml"
ANSWERS_FILE = Path("golden") / "answers-v1.toml"
SLICES_FILE = Path("golden") / "slices-v1.toml"
# One client ID per item stays inside the message limiter's burst of 5 (ADR 0010).
MAX_TURNS = 4

_VERSION = re.compile(r"\d+\.\d+\.\d+")
_ID = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
_ESCAPE = re.compile(r"\\.")
_UPPER = re.compile(r"[A-Z]")
_TOP_KEYS = frozenset({"version", "questions"})
_ANSWER_REQUIRED = ("id", "split", "key_facts")
_ANSWER_KEYS = frozenset({*_ANSWER_REQUIRED, "warning_expected", "warning_facts"})
_SLICE_COMMON = ("id", "slice", "split", "turns")
# Per slice: (required keys, optional keys) besides the common ones.
_SLICE_KEYS: Mapping[Slice, tuple[tuple[str, ...], tuple[str, ...]]] = {
    Slice.MULTI_TURN: (("anchors", "match", "key_facts", "rewrite_trap"), ("rewrite_forbidden",)),
    Slice.LONG_QUERY: (("pair_of",), ()),
    Slice.OUT_OF_SCOPE: (("oos_kind",), ()),
    Slice.ADVERSARIAL: (("attack", "expected", "forbidden"), ("key_facts",)),
}

Table = Mapping[str, object]


def _fail(rule: DatasetRule, message: str) -> DatasetError:
    return DatasetError(rule, message)


# ---------------------------------------------------------------- values


def _table(raw: object, where: str) -> Table:
    if not isinstance(raw, dict):
        raise _fail(DatasetRule.WRONG_TYPE, f"{where} must be a table")
    return raw


def _check_keys(table: Table, allowed: Set[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise _fail(DatasetRule.UNKNOWN_KEY, f"{where}: unknown key: {', '.join(unknown)}")


def _require(table: Table, keys: Sequence[str], where: str) -> None:
    for key in keys:
        if key not in table:
            raise _fail(DatasetRule.MISSING_KEY, f"{where}: missing key: {key}")


def _string(table: Table, key: str, where: str) -> str:
    value = table[key]
    if not isinstance(value, str):
        raise _fail(DatasetRule.WRONG_TYPE, f"{where}: {key} must be a string")
    return value


def _bool(table: Table, key: str, where: str) -> bool:
    value = table[key]
    if not isinstance(value, bool):
        raise _fail(DatasetRule.WRONG_TYPE, f"{where}: {key} must be true or false")
    return value


def _enum[E: (Split, Slice, Match, OosKind, Attack, Expected)](
    enum: type[E], table: Table, key: str, where: str
) -> E:
    value = _string(table, key, where)
    try:
        return enum(value)
    except ValueError:
        raise _fail(DatasetRule.UNKNOWN_VALUE, f"{where}: unknown {key}") from None


def _strings(raw: object, where: str) -> list[str]:
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise _fail(DatasetRule.WRONG_TYPE, f"{where} must be a list of strings")
    return raw


# ---------------------------------------------------------------- facts and patterns


def _compile(pattern: str, where: str) -> None:
    if not pattern.strip():
        raise _fail(DatasetRule.INVALID_PATTERN, f"{where}: empty pattern")
    try:
        re.compile(pattern)
    except re.error:
        raise _fail(DatasetRule.INVALID_PATTERN, f"{where}: pattern does not compile") from None


def _alternative(text: str, where: str) -> str:
    """A plain string or a `re:` pattern; patterns run on the case-folded answer, so an
    upper-case letter outside an escape (\\b, \\S, ...) could never match.
    """
    if not text.strip():
        raise _fail(DatasetRule.BLANK_FACT, f"{where} is blank")
    if len(text) > MAX_FACT_CHARS:
        raise _fail(
            DatasetRule.FACT_TOO_LONG,
            f"{where} has {len(text)} characters, max {MAX_FACT_CHARS}",
        )
    if text.startswith(REGEX_PREFIX):
        pattern = text.removeprefix(REGEX_PREFIX)
        _compile(pattern, where)
        if _UPPER.search(_ESCAPE.sub("", pattern)):
            raise _fail(DatasetRule.PATTERN_NOT_LOWERCASE, f"{where}: pattern has upper case")
    return text


def _group(raw: object, where: str) -> KeyFactGroup:
    alternatives = _strings(raw, where)
    if not alternatives:
        raise _fail(DatasetRule.EMPTY_GROUP, f"{where} is empty")
    return tuple(_alternative(text, f"{where}[{n}]") for n, text in enumerate(alternatives))


def _key_facts(raw: object, where: str) -> tuple[KeyFactGroup, ...]:
    if not isinstance(raw, list):
        raise _fail(DatasetRule.WRONG_TYPE, f"{where} must be a list of groups")
    return tuple(_group(group, f"{where}[{n}]") for n, group in enumerate(raw))


def _patterns(raw: object, where: str, *, required: bool) -> tuple[str, ...]:
    patterns = _strings(raw, where)
    if required and not patterns:
        raise _fail(DatasetRule.EMPTY_GROUP, f"{where} must not be empty")
    for n, pattern in enumerate(patterns):
        _compile(pattern, f"{where}[{n}]")
    return tuple(patterns)


# ---------------------------------------------------------------- files


def _document(data: bytes, name: str) -> tuple[str, list[object]]:
    try:
        document = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise _fail(DatasetRule.INVALID_TOML, f"{name} is not valid TOML: {exc}") from None
    _check_keys(document, _TOP_KEYS, name)
    if "version" not in document:
        raise _fail(DatasetRule.MISSING_KEY, f"{name}: missing key: version")
    version = document["version"]
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise _fail(DatasetRule.INVALID_VERSION, f"{name}: version must look like 1.0.0")
    questions = document.get("questions")
    if questions is None or questions == []:
        raise _fail(DatasetRule.NO_QUESTIONS, f"{name} has no questions")
    if not isinstance(questions, list):
        raise _fail(DatasetRule.WRONG_TYPE, f"{name}: questions must be a list of tables")
    return version, questions


def _identity(table: Table, where: str) -> str:
    _require(table, ("id",), where)
    qid = _string(table, "id", where)
    if not _ID.fullmatch(qid):
        raise _fail(DatasetRule.INVALID_ID, f"{where}: id must be a lowercase slug")
    return qid


def _check_unique(ids: Sequence[str], taken: Set[str]) -> None:
    seen = set(taken)
    for qid in ids:
        if qid in seen:
            raise _fail(DatasetRule.DUPLICATE_ID, f"duplicate id: {qid}")
        seen.add(qid)


# ---------------------------------------------------------------- answers-v1


def _warning(table: Table, where: str) -> KeyFactGroup | None:
    """`warning_facts` (exactly one group) is set exactly when a warning is expected."""
    expected = _bool(table, "warning_expected", where) if "warning_expected" in table else False
    if "warning_facts" not in table:
        if expected:
            raise _fail(DatasetRule.WARNING_FACTS, f"{where}: a warning needs warning_facts")
        return None
    if not expected:
        raise _fail(DatasetRule.WARNING_FACTS, f"{where}: warning_facts without a warning")
    groups = _key_facts(table["warning_facts"], f"{where} warning_facts")
    if len(groups) != 1:
        raise _fail(DatasetRule.WARNING_FACTS, f"{where}: warning_facts must be one group")
    return groups[0]


def _answer_key(raw: object, position: int) -> AnswerKey:
    table = _table(raw, f"answers questions[{position}]")
    qid = _identity(table, f"answers questions[{position}]")
    _check_keys(table, _ANSWER_KEYS, qid)
    _require(table, _ANSWER_REQUIRED, qid)
    return AnswerKey(
        id=qid,
        split=_enum(Split, table, "split", qid),
        key_facts=_key_facts(table["key_facts"], f"{qid} key_facts"),
        warning_facts=_warning(table, qid),
    )


def _check_against_golden(keys: Sequence[AnswerKey], golden: GoldenSet) -> None:
    questions = {q.id: q for q in golden.questions}
    for key in keys:
        question = questions.get(key.id)
        if question is None:
            raise _fail(DatasetRule.UNKNOWN_ID, f"{key.id}: no such retrieval question")
        if key.split is not question.split:
            raise _fail(DatasetRule.SPLIT_MISMATCH, f"{key.id}: split differs from retrieval")
        if question.in_scope != bool(key.key_facts):
            raise _fail(
                DatasetRule.KEY_FACTS_SCOPE,
                f"{key.id}: key facts are required in scope and empty out of scope",
            )
    covered = {key.id for key in keys}
    missing = [q.id for q in golden.questions if q.id not in covered]
    if missing:
        raise _fail(DatasetRule.MISSING_ENTRY, f"no answers entry for: {', '.join(missing)}")


def parse_answers(data: bytes, golden: GoldenSet) -> AnswerFile:
    """Parse and validate the raw bytes of answers-v1; the digest is over those bytes."""
    version, raw = _document(data, ANSWERS_FILE.name)
    keys = tuple(_answer_key(item, n) for n, item in enumerate(raw))
    _check_unique([key.id for key in keys], set())
    _check_against_golden(keys, golden)
    return AnswerFile(version, hashlib.sha256(data).hexdigest(), keys)


# ---------------------------------------------------------------- slices-v1


def _turns(raw: object, kind: Slice, where: str, max_chars: int) -> tuple[str, ...]:
    """Earlier turns are replayed through the API, so each must pass its input limit."""
    turns = _strings(raw, f"{where}: turns")
    fewest = 2 if kind is Slice.MULTI_TURN else 1
    if not fewest <= len(turns) <= MAX_TURNS:
        raise _fail(
            DatasetRule.TURN_COUNT, f"{where}: {len(turns)} turns, allowed {fewest} to {MAX_TURNS}"
        )
    for n, turn in enumerate(turns):
        if not turn.strip():
            raise _fail(DatasetRule.BLANK_TURN, f"{where}: turns[{n}] is blank")
        if len(turn.strip()) > max_chars:
            raise _fail(
                DatasetRule.TURN_TOO_LONG, f"{where}: turns[{n}] is over {max_chars} characters"
            )
    return tuple(turns)


def _anchors(raw: object, where: str, doc_ids: Set[str]) -> tuple[Anchor, ...]:
    if not isinstance(raw, list) or not raw:
        raise _fail(DatasetRule.INVALID_ANCHOR, f"{where}: anchors must be a non-empty list")
    try:
        return tuple(
            parse_anchor(item, f"{where} anchors[{n}]", doc_ids) for n, item in enumerate(raw)
        )
    except GoldenSetError as exc:
        # The golden set's messages also name positions only, never facts.
        raise _fail(DatasetRule.INVALID_ANCHOR, str(exc)) from None


def _in_scope_facts(table: Table, where: str) -> tuple[KeyFactGroup, ...]:
    facts = _key_facts(table.get("key_facts", []), f"{where} key_facts")
    if not facts:
        raise _fail(DatasetRule.KEY_FACTS_SCOPE, f"{where}: an expected answer needs key facts")
    return facts


def _multi_turn(base: SliceItem, table: Table, doc_ids: Set[str]) -> SliceItem:
    where = base.id
    forbidden = table.get("rewrite_forbidden", [])
    return dataclasses.replace(
        base,
        anchors=_anchors(table["anchors"], where, doc_ids),
        match=_enum(Match, table, "match", where),
        key_facts=_in_scope_facts(table, where),
        rewrite_trap=_bool(table, "rewrite_trap", where),
        rewrite_forbidden=_patterns(forbidden, f"{where} rewrite_forbidden", required=False),
    )


def _long_query(base: SliceItem, table: Table, golden: GoldenSet) -> SliceItem:
    """A padded version of an in-scope retrieval question, scored against its pair."""
    pair_of = _string(table, "pair_of", base.id)
    pair = next((q for q in golden.questions if q.id == pair_of), None)
    if pair is None:
        raise _fail(DatasetRule.UNRESOLVED_PAIR, f"{base.id}: pair_of names no retrieval question")
    if not pair.in_scope:
        raise _fail(DatasetRule.PAIR_OUT_OF_SCOPE, f"{base.id}: pair_of is out of scope")
    if pair.split is not base.split:
        raise _fail(DatasetRule.SPLIT_MISMATCH, f"{base.id}: split differs from its pair")
    return dataclasses.replace(base, pair_of=pair_of)


def _adversarial(base: SliceItem, table: Table) -> SliceItem:
    where = base.id
    expected = _enum(Expected, table, "expected", where)
    if expected is Expected.ANSWER_IN_SCOPE:
        facts = _in_scope_facts(table, where)
    elif "key_facts" in table:
        raise _fail(DatasetRule.KEY_FACTS_SCOPE, f"{where}: a refusal has no key facts")
    else:
        facts = ()
    return dataclasses.replace(
        base,
        attack=_enum(Attack, table, "attack", where),
        expected=expected,
        forbidden=_patterns(table["forbidden"], f"{where} forbidden", required=True),
        key_facts=facts,
    )


def _slice_item(
    raw: object, position: int, golden: GoldenSet, doc_ids: Set[str], max_chars: int
) -> SliceItem:
    table = _table(raw, f"slices questions[{position}]")
    sid = _identity(table, f"slices questions[{position}]")
    _require(table, ("slice",), sid)
    kind = _enum(Slice, table, "slice", sid)
    required, optional = _SLICE_KEYS[kind]
    _check_keys(table, {*_SLICE_COMMON, *required, *optional}, sid)
    _require(table, (*_SLICE_COMMON, *required), sid)
    base = SliceItem(
        id=sid,
        slice=kind,
        split=_enum(Split, table, "split", sid),
        turns=_turns(table["turns"], kind, sid, max_chars),
    )
    if kind is Slice.MULTI_TURN:
        return _multi_turn(base, table, doc_ids)
    if kind is Slice.LONG_QUERY:
        return _long_query(base, table, golden)
    if kind is Slice.OUT_OF_SCOPE:
        return dataclasses.replace(base, oos_kind=_enum(OosKind, table, "oos_kind", sid))
    return _adversarial(base, table)


def parse_slices(
    data: bytes, golden: GoldenSet, doc_ids: Set[str], *, max_turn_chars: int
) -> SliceFile:
    """Parse and validate the raw bytes of slices-v1; `max_turn_chars` is the API's
    message limit, so no replayed turn is rejected with a 422 mid-run.
    """
    version, raw = _document(data, SLICES_FILE.name)
    items = tuple(
        _slice_item(item, n, golden, doc_ids, max_turn_chars) for n, item in enumerate(raw)
    )
    _check_unique([item.id for item in items], {q.id for q in golden.questions})
    return SliceFile(version, hashlib.sha256(data).hexdigest(), items)


# ---------------------------------------------------------------- items


def _answer_item(question: GoldenQuestion, key: AnswerKey) -> EvalItem:
    return EvalItem(
        id=question.id,
        split=question.split,
        turns=(question.question,),
        in_scope=question.in_scope,
        category=question.category,
        anchors=question.anchors,
        match=question.match,
        key_facts=key.key_facts,
        warning_facts=key.warning_facts,
        oos_kind=question.oos_kind,
    )


def _expects_answer(item: SliceItem) -> bool:
    if item.slice is Slice.ADVERSARIAL:
        return item.expected is Expected.ANSWER_IN_SCOPE
    return item.slice is not Slice.OUT_OF_SCOPE


def _slice_eval_item(item: SliceItem, answered: Mapping[str, EvalItem]) -> EvalItem:
    if item.slice is Slice.LONG_QUERY:
        # parse_slices resolved pair_of to an in-scope question of the same split.
        pair = answered[str(item.pair_of)]
        return dataclasses.replace(
            pair,
            id=item.id,
            split=item.split,
            turns=item.turns,
            slice=item.slice,
            pair_of=item.pair_of,
        )
    return EvalItem(
        id=item.id,
        split=item.split,
        turns=item.turns,
        in_scope=_expects_answer(item),
        slice=item.slice,
        anchors=item.anchors,
        match=item.match,
        key_facts=item.key_facts,
        oos_kind=item.oos_kind,
        attack=item.attack,
        expected=item.expected,
        forbidden=item.forbidden,
        rewrite_trap=item.rewrite_trap,
        rewrite_forbidden=item.rewrite_forbidden,
    )


def build_items(golden: GoldenSet, answers: AnswerFile, slices: SliceFile) -> tuple[EvalItem, ...]:
    """Answers items in retrieval order, then slice items in file order."""
    keys = {key.id: key for key in answers.keys}
    answered = tuple(_answer_item(q, keys[q.id]) for q in golden.questions)
    by_id = {item.id: item for item in answered}
    return (*answered, *(_slice_eval_item(item, by_id) for item in slices.items))


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        raise _fail(DatasetRule.UNREADABLE, f"cannot read {path.name}") from None


def load_dataset(eval_dir: Path, doc_ids: Set[str], *, max_turn_chars: int) -> Dataset:
    golden = load_golden(eval_dir / RETRIEVAL_FILE, doc_ids)
    answers = parse_answers(_read(eval_dir / ANSWERS_FILE), golden)
    slices = parse_slices(
        _read(eval_dir / SLICES_FILE), golden, doc_ids, max_turn_chars=max_turn_chars
    )
    files = (
        DatasetFile(RETRIEVAL_FILE.name, golden.version, golden.sha256),
        DatasetFile(ANSWERS_FILE.name, answers.version, answers.sha256),
        DatasetFile(SLICES_FILE.name, slices.version, slices.sha256),
    )
    return Dataset(files=files, items=build_items(golden, answers, slices))

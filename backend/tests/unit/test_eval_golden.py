"""Behavior of the golden set loader and its structural validator (synthetic sets only)."""

import hashlib
import json
import pathlib
from collections.abc import Mapping, Sequence

import pytest

from app.domain.evaluation import (
    Anchor,
    Category,
    GoldenRule,
    GoldenSetError,
    Match,
    OosKind,
    Split,
)
from app.evaluation.golden import MAX_FACT_CHARS, load_golden, parse_golden

DOC_IDS = frozenset({"demo-guide", "other-guide"})

Raw = dict[str, object]


def anchor(doc_id: str = "demo-guide", facts: Sequence[str] = ("Part 123",), **extra: str) -> Raw:
    return {"doc_id": doc_id, "facts": list(facts), **extra}


def question(
    qid: str,
    category: str = "factual",
    split: str = "dev",
    *,
    anchors: Sequence[Raw] | None = None,
    match: str | None = "any",
    **extra: object,
) -> Raw:
    raw: Raw = {"id": qid, "category": category, "split": split, "question": f"Q {qid}?"}
    if match is not None:
        raw["match"] = match
    raw.update(extra)
    if anchors is None and category != "out_of_scope":
        anchors = [anchor()]
    if anchors is not None:
        raw["anchors"] = list(anchors)
    return raw


def oos(qid: str, split: str = "dev", kind: str = "unrelated") -> Raw:
    return question(qid, "out_of_scope", split, match=None, oos_kind=kind)


def valid_questions() -> list[Raw]:
    """Ten questions, two of them held-out (20%), one held-out in each category."""
    return [
        *(question(f"fact-{n}") for n in range(4)),
        question("fact-held", split="heldout"),
        *(oos(f"oos-{n}") for n in range(4)),
        oos("oos-held", split="heldout", kind="near_domain"),
    ]


def _value(value: object) -> str:
    # JSON strings, numbers, booleans and string arrays are valid TOML values.
    return json.dumps(value)


def render(questions: Sequence[Mapping[str, object]], **top: object) -> bytes:
    lines = [f"{key} = {_value(value)}" for key, value in {"version": "1.0.0", **top}.items()]
    for raw in questions:
        lines += ["", "[[questions]]"]
        anchors = raw.get("anchors")
        tables = anchors if isinstance(anchors, list) else []
        inline = {k: v for k, v in raw.items() if not (k == "anchors" and isinstance(v, list))}
        lines += [f"{k} = {_value(v)}" for k, v in inline.items()]
        for item in tables:
            assert isinstance(item, dict)
            lines += ["", "[[questions.anchors]]"]
            lines += [f"{k} = {_value(v)}" for k, v in item.items()]
    return ("\n".join(lines) + "\n").encode()


def parse(questions: Sequence[Mapping[str, object]], **top: object) -> object:
    return parse_golden(render(questions, **top), DOC_IDS)


def rule_of(questions: Sequence[Mapping[str, object]], **top: object) -> GoldenSetError:
    with pytest.raises(GoldenSetError) as caught:
        parse(questions, **top)
    return caught.value


# ------------------------------------------------------------------ valid sets


def test_a_valid_set_loads_with_its_version_digest_and_questions() -> None:
    data = render(valid_questions())

    golden = parse_golden(data, DOC_IDS)

    assert golden.version == "1.0.0"
    assert golden.sha256 == hashlib.sha256(data).hexdigest()
    assert [q.id for q in golden.questions][:2] == ["fact-0", "fact-1"]
    assert len(golden.questions) == 10


def test_question_fields_are_parsed_into_typed_values() -> None:
    questions = valid_questions()
    questions[0] = question(
        "fact-0",
        "table",
        anchors=[anchor(facts=["A", "B"], section="Bottom cover"), anchor("other-guide")],
        match="all",
    )
    questions[1] = question("fact-1", "table", split="heldout")
    questions[2] = question("fu-1", "follow_up", context="Earlier turn")
    questions[3] = question("fu-held", "follow_up", split="heldout", context="Earlier")
    questions += [question(f"pad-{n}") for n in range(10)]  # keeps the held-out share at 20%

    golden = parse_golden(render(questions), DOC_IDS)

    first = golden.questions[0]
    assert first.category is Category.TABLE
    assert first.split is Split.DEV
    assert first.match is Match.ALL
    assert first.question == "Q fact-0?"
    assert first.anchors == (
        Anchor(doc_id="demo-guide", facts=("A", "B"), section="Bottom cover"),
        Anchor(doc_id="other-guide", facts=("Part 123",), section=None),
    )
    assert first.in_scope
    assert golden.questions[2].context == "Earlier turn"
    assert golden.questions[0].context is None
    held_oos = next(q for q in golden.questions if q.id == "oos-held")
    assert held_oos.oos_kind is OosKind.NEAR_DOMAIN
    assert held_oos.anchors == ()
    assert held_oos.match is None
    assert not held_oos.in_scope


def test_out_of_scope_questions_may_state_a_match_mode() -> None:
    questions = valid_questions()
    questions[5] = {**oos("oos-0"), "match": "any"}

    golden = parse_golden(render(questions), DOC_IDS)

    assert golden.questions[5].match is Match.ANY


def test_questions_of_one_split_keep_file_order() -> None:
    golden = parse_golden(render(valid_questions()), DOC_IDS)

    assert [q.id for q in golden.in_split(Split.HELDOUT)] == ["fact-held", "oos-held"]
    assert len(golden.in_split(Split.DEV)) == 8


def test_a_fact_of_exactly_the_maximum_length_is_accepted() -> None:
    questions = valid_questions()
    questions[0] = question("fact-0", anchors=[anchor(facts=["x" * MAX_FACT_CHARS])])

    parse(questions)


@pytest.mark.parametrize(("held", "total"), [(3, 20), (5, 20)])
def test_held_out_shares_at_the_bounds_are_accepted(held: int, total: int) -> None:
    questions = [question(f"q-{n}", split="heldout" if n < held else "dev") for n in range(total)]

    parse(questions)


def test_loads_from_a_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "golden.toml"
    path.write_bytes(render(valid_questions()))

    assert load_golden(path, DOC_IDS).version == "1.0.0"


# ------------------------------------------------------------------ one rule per error


def test_a_missing_file_is_unreadable(tmp_path: pathlib.Path) -> None:
    with pytest.raises(GoldenSetError) as caught:
        load_golden(tmp_path / "missing.toml", DOC_IDS)

    assert caught.value.rule is GoldenRule.UNREADABLE


def test_invalid_toml_is_rejected() -> None:
    with pytest.raises(GoldenSetError) as caught:
        parse_golden(b"version = ", DOC_IDS)

    assert caught.value.rule is GoldenRule.INVALID_TOML


def test_non_utf8_bytes_are_invalid_toml() -> None:
    with pytest.raises(GoldenSetError) as caught:
        parse_golden(b'version = "\xff"', DOC_IDS)

    assert caught.value.rule is GoldenRule.INVALID_TOML


def test_a_duplicate_id_is_rejected_and_named() -> None:
    questions = valid_questions()
    questions[1] = question("fact-0")

    error = rule_of(questions)

    assert error.rule is GoldenRule.DUPLICATE_ID
    assert "fact-0" in str(error)


@pytest.mark.parametrize(
    ("field", "value", "rule"),
    [
        ("category", "trivia", GoldenRule.UNKNOWN_CATEGORY),
        ("split", "test", GoldenRule.UNKNOWN_SPLIT),
        ("match", "some", GoldenRule.UNKNOWN_MATCH),
        ("id", "Fact 0", GoldenRule.INVALID_ID),
        ("question", "   ", GoldenRule.EMPTY_QUESTION),
        ("question", 7, GoldenRule.WRONG_TYPE),
        ("context", "not a follow-up", GoldenRule.UNEXPECTED_CONTEXT),
        ("oos_kind", "unrelated", GoldenRule.UNEXPECTED_OOS_KIND),
        ("difficulty", "hard", GoldenRule.UNKNOWN_KEY),
    ],
)
def test_a_bad_question_field_fails_with_its_own_rule(
    field: str, value: object, rule: GoldenRule
) -> None:
    questions = valid_questions()
    questions[0] = {**questions[0], field: value}

    error = rule_of(questions)

    assert error.rule is rule
    expected_id = "fact-0" if field != "id" else "questions[0]"
    assert expected_id in str(error)


@pytest.mark.parametrize("field", ["id", "category", "split", "question"])
def test_a_missing_question_field_is_rejected(field: str) -> None:
    questions = valid_questions()
    questions[0] = {k: v for k, v in questions[0].items() if k != field}

    assert rule_of(questions).rule is GoldenRule.MISSING_KEY


def test_an_in_scope_question_needs_a_match_mode() -> None:
    questions = valid_questions()
    questions[0] = question("fact-0", match=None)

    assert rule_of(questions).rule is GoldenRule.MISSING_KEY


@pytest.mark.parametrize("anchors", [None, []])
def test_an_in_scope_question_without_anchors_is_rejected(anchors: list[Raw] | None) -> None:
    questions = valid_questions()
    raw = question("fact-0")
    if anchors is None:
        del raw["anchors"]
    else:
        raw["anchors"] = anchors
    questions[0] = raw

    assert rule_of(questions).rule is GoldenRule.MISSING_ANCHORS


def test_an_out_of_scope_question_with_anchors_is_rejected() -> None:
    questions = valid_questions()
    questions[5] = {**oos("oos-0"), "anchors": [anchor()]}

    assert rule_of(questions).rule is GoldenRule.UNEXPECTED_ANCHORS


def test_an_out_of_scope_question_needs_its_kind() -> None:
    questions = valid_questions()
    questions[5] = {k: v for k, v in oos("oos-0").items() if k != "oos_kind"}

    assert rule_of(questions).rule is GoldenRule.MISSING_OOS_KIND


def test_an_unknown_out_of_scope_kind_is_rejected() -> None:
    questions = valid_questions()
    questions[5] = oos("oos-0", kind="adjacent")

    assert rule_of(questions).rule is GoldenRule.UNKNOWN_OOS_KIND


def test_a_follow_up_needs_the_earlier_turn() -> None:
    questions = valid_questions()
    questions[0] = question("fact-0", "follow_up")
    questions[4] = question("fact-held", "follow_up", split="heldout", context="Earlier")

    assert rule_of(questions).rule is GoldenRule.MISSING_CONTEXT


@pytest.mark.parametrize(
    ("raw_anchor", "rule"),
    [
        (anchor(facts=["x" * (MAX_FACT_CHARS + 1)]), GoldenRule.FACT_TOO_LONG),
        (anchor(facts=[]), GoldenRule.EMPTY_FACTS),
        (anchor(facts=["  "]), GoldenRule.BLANK_FACT),
        (anchor("unknown-guide"), GoldenRule.UNKNOWN_DOC_ID),
        (anchor(section=" "), GoldenRule.BLANK_SECTION),
        (anchor(page="12"), GoldenRule.UNKNOWN_KEY),
        ({"facts": ["A"]}, GoldenRule.MISSING_KEY),
        ({"doc_id": "demo-guide", "facts": "A"}, GoldenRule.WRONG_TYPE),
        ({"doc_id": "demo-guide", "facts": [1]}, GoldenRule.WRONG_TYPE),
        ({"doc_id": 1, "facts": ["A"]}, GoldenRule.WRONG_TYPE),
    ],
)
def test_a_bad_anchor_fails_with_its_own_rule(raw_anchor: Raw, rule: GoldenRule) -> None:
    questions = valid_questions()
    questions[0] = question("fact-0", anchors=[anchor(), raw_anchor])

    error = rule_of(questions)

    assert error.rule is rule
    assert "fact-0" in str(error)
    assert "anchors[1]" in str(error)


def test_anchors_must_be_a_list_of_tables() -> None:
    questions = valid_questions()
    questions[0] = {**question("fact-0"), "anchors": 3}

    assert rule_of(questions).rule is GoldenRule.WRONG_TYPE


@pytest.mark.parametrize(("held", "total"), [(2, 20), (6, 20)])
def test_a_held_out_share_out_of_range_is_rejected(held: int, total: int) -> None:
    questions = [question(f"q-{n}", split="heldout" if n < held else "dev") for n in range(total)]

    assert rule_of(questions).rule is GoldenRule.HELDOUT_SHARE


def test_every_category_needs_a_held_out_question() -> None:
    questions = valid_questions()
    questions[9] = oos("oos-4")  # out_of_scope loses its only held-out question
    questions[3] = question("fact-3", split="heldout")  # the share stays at 20%

    error = rule_of(questions)

    assert error.rule is GoldenRule.HELDOUT_MISSING_IN_CATEGORY
    assert "out_of_scope" in str(error)


@pytest.mark.parametrize("version", ["1.0", "v1.0.0", "1.0.0-rc1", ""])
def test_the_version_must_be_semantic(version: str) -> None:
    assert rule_of(valid_questions(), version=version).rule is GoldenRule.INVALID_VERSION


def test_the_version_is_required() -> None:
    data = render(valid_questions()).replace(b'version = "1.0.0"\n', b"")

    with pytest.raises(GoldenSetError) as caught:
        parse_golden(data, DOC_IDS)

    assert caught.value.rule is GoldenRule.MISSING_KEY


def test_an_unknown_top_level_key_is_rejected() -> None:
    assert rule_of(valid_questions(), owner="me").rule is GoldenRule.UNKNOWN_KEY


@pytest.mark.parametrize("data", [b'version = "1.0.0"\n', b'version = "1.0.0"\nquestions = []\n'])
def test_a_set_without_questions_is_rejected(data: bytes) -> None:
    with pytest.raises(GoldenSetError) as caught:
        parse_golden(data, DOC_IDS)

    assert caught.value.rule is GoldenRule.NO_QUESTIONS


@pytest.mark.parametrize("value", [b'["a"]', b"3"])
def test_questions_must_be_a_list_of_tables(value: bytes) -> None:
    with pytest.raises(GoldenSetError) as caught:
        parse_golden(b'version = "1.0.0"\nquestions = ' + value + b"\n", DOC_IDS)

    assert caught.value.rule is GoldenRule.WRONG_TYPE


def test_error_messages_never_quote_the_question_text() -> None:
    questions = valid_questions()
    questions[0] = {**question("fact-0", category="trivia"), "question": "SECRET-WORDING?"}

    assert "SECRET-WORDING" not in str(rule_of(questions))

"""Behavior of the Phase 5 dataset loader and its validator (synthetic files only)."""

import hashlib
import pathlib
from collections.abc import Callable, Mapping, Sequence

import pytest

from app.domain.answer_dataset import (
    AnswerFile,
    Attack,
    DatasetError,
    DatasetRule,
    Expected,
    Slice,
    SliceFile,
)
from app.domain.evaluation import Anchor, Category, Match, OosKind, Split
from app.evaluation.answers.dataset import (
    ANSWERS_FILE,
    RETRIEVAL_FILE,
    SLICES_FILE,
    build_items,
    load_dataset,
    parse_answers,
    parse_slices,
)
from tests.unit.answer_dataset_data import (
    DOC_IDS,
    GOLDEN,
    MAX_TURN_CHARS,
    Raw,
    adversarial,
    anchor,
    entry,
    long_query,
    multi_turn,
    out_of_scope,
    render,
    valid_answers,
    valid_slices,
)

CANARY = "zebra-canary-fact"

Questions = Sequence[Mapping[str, object]]


def answers(questions: Questions, **top: object) -> AnswerFile:
    return parse_answers(render(questions, **top), GOLDEN)


def slices(questions: Questions, **top: object) -> SliceFile:
    return parse_slices(render(questions, **top), GOLDEN, DOC_IDS, max_turn_chars=MAX_TURN_CHARS)


def failure(parse: Callable[[], object]) -> DatasetError:
    with pytest.raises(DatasetError) as caught:
        parse()
    return caught.value


def answers_rule(questions: Questions, **top: object) -> DatasetRule:
    return failure(lambda: answers(questions, **top)).rule


def slices_rule(questions: Questions, **top: object) -> DatasetRule:
    return failure(lambda: slices(questions, **top)).rule


def with_answer(position: int, raw: Raw) -> list[Raw]:
    questions = valid_answers()
    questions[position] = raw
    return questions


# ================================================================ answers-v1


def test_a_valid_answers_file_parses_every_entry_in_order() -> None:
    data = render(valid_answers())

    parsed = parse_answers(data, GOLDEN)

    assert parsed.version == "1.0.0"
    assert parsed.sha256 == hashlib.sha256(data).hexdigest()
    assert [key.id for key in parsed.keys] == [q.id for q in GOLDEN.questions]
    first = parsed.keys[0]
    assert first.split is Split.DEV
    assert first.key_facts == (("alpha", "re:\\bbeta\\b"), ("gamma",))
    assert first.warning_facts is None
    assert not first.warning_expected


def test_an_expected_warning_keeps_its_one_group() -> None:
    warn = answers(valid_answers()).keys[2]

    assert warn.warning_expected
    assert warn.warning_facts == ("unplug", "power cord")


def test_out_of_scope_entries_have_no_key_facts() -> None:
    parsed = answers(valid_answers())

    assert parsed.keys[4].key_facts == ()


def test_warning_expected_false_alone_means_no_warning() -> None:
    questions = with_answer(0, entry("fact-1", warning_expected=False))

    assert not answers(questions).keys[0].warning_expected


@pytest.mark.parametrize(
    "data",
    [b"version = ", b"\xff\xfe not utf-8"],
    ids=["broken-toml", "not-utf8"],
)
def test_an_unparseable_file_is_invalid_toml(data: bytes) -> None:
    assert failure(lambda: parse_answers(data, GOLDEN)).rule is DatasetRule.INVALID_TOML


def test_an_unknown_top_level_key_is_rejected() -> None:
    assert answers_rule(valid_answers(), extra="x") is DatasetRule.UNKNOWN_KEY


def test_an_unknown_entry_key_is_rejected() -> None:
    questions = with_answer(0, entry("fact-1", category="factual"))

    assert answers_rule(questions) is DatasetRule.UNKNOWN_KEY


def test_the_version_is_required_and_must_look_like_semver() -> None:
    no_version = render(valid_answers()).replace(b'version = "1.0.0"\n', b"")

    assert failure(lambda: parse_answers(no_version, GOLDEN)).rule is DatasetRule.MISSING_KEY
    assert answers_rule(valid_answers(), version="v1") is DatasetRule.INVALID_VERSION
    assert answers_rule(valid_answers(), version=1) is DatasetRule.INVALID_VERSION


def test_a_file_without_questions_is_rejected() -> None:
    assert answers_rule([]) is DatasetRule.NO_QUESTIONS
    assert (
        failure(lambda: parse_answers(b'version = "1.0.0"\nquestions = []\n', GOLDEN)).rule
        is DatasetRule.NO_QUESTIONS
    )
    assert (
        failure(lambda: parse_answers(b'version = "1.0.0"\nquestions = "x"\n', GOLDEN)).rule
        is DatasetRule.WRONG_TYPE
    )


def test_an_entry_that_is_not_a_table_is_a_wrong_type() -> None:
    data = b'version = "1.0.0"\nquestions = ["fact-1"]\n'

    assert failure(lambda: parse_answers(data, GOLDEN)).rule is DatasetRule.WRONG_TYPE


@pytest.mark.parametrize("key", ["id", "split", "key_facts"])
def test_each_required_entry_key_is_checked(key: str) -> None:
    raw = entry("fact-1")
    del raw[key]

    assert answers_rule(with_answer(0, raw)) is DatasetRule.MISSING_KEY


def test_ids_are_lowercase_slugs_and_unique() -> None:
    assert answers_rule(with_answer(0, entry("Fact 1"))) is DatasetRule.INVALID_ID
    assert answers_rule(with_answer(0, entry(7))) is DatasetRule.WRONG_TYPE  # type: ignore[arg-type]
    duplicated = [*valid_answers(), entry("fact-1")]
    assert answers_rule(duplicated) is DatasetRule.DUPLICATE_ID


def test_an_unknown_split_is_rejected() -> None:
    assert answers_rule(with_answer(0, entry("fact-1", "train"))) is DatasetRule.UNKNOWN_VALUE


@pytest.mark.parametrize(
    "key_facts",
    ["alpha", ["alpha"], [[1]], [["alpha"], "beta"]],
    ids=["string", "flat-list", "number", "mixed"],
)
def test_key_facts_must_be_a_list_of_string_groups(key_facts: object) -> None:
    raw = entry("fact-1", key_facts=key_facts)

    assert answers_rule(with_answer(0, raw)) is DatasetRule.WRONG_TYPE


@pytest.mark.parametrize(
    ("key_facts", "rule"),
    [
        ([[]], DatasetRule.EMPTY_GROUP),
        ([["alpha", "  "]], DatasetRule.BLANK_FACT),
        ([["x" * 81]], DatasetRule.FACT_TOO_LONG),
        ([["re:(unclosed"]], DatasetRule.INVALID_PATTERN),
        ([["re:"]], DatasetRule.INVALID_PATTERN),
        ([["re:Paper"]], DatasetRule.PATTERN_NOT_LOWERCASE),
        ([["re:[A-Z]+"]], DatasetRule.PATTERN_NOT_LOWERCASE),
    ],
    ids=["empty", "blank", "too-long", "bad-regex", "empty-regex", "upper", "upper-range"],
)
def test_a_bad_key_fact_fails_with_its_own_rule(key_facts: object, rule: DatasetRule) -> None:
    raw = entry("fact-1", key_facts=key_facts)

    assert answers_rule(with_answer(0, raw)) is rule


def test_regex_escapes_in_upper_case_are_allowed() -> None:
    """\\b, \\d, \\S and \\W are escapes, not upper-case literals of the answer."""
    raw = entry("fact-1", key_facts=[["re:\\b\\d+\\S*\\W?\\b"]])

    assert answers(with_answer(0, raw)).keys[0].key_facts == (("re:\\b\\d+\\S*\\W?\\b",),)


def test_an_eighty_character_fact_is_allowed() -> None:
    raw = entry("fact-1", key_facts=[["x" * 80]])

    assert answers(with_answer(0, raw)).keys[0].key_facts == (("x" * 80,),)


@pytest.mark.parametrize(
    "extra",
    [
        {"warning_expected": True},
        {"warning_facts": [["unplug"]]},
        {"warning_expected": False, "warning_facts": [["unplug"]]},
        {"warning_expected": True, "warning_facts": [["unplug"], ["cord"]]},
        {"warning_expected": True, "warning_facts": []},
    ],
    ids=["no-facts", "no-flag", "flag-false", "two-groups", "no-group"],
)
def test_warning_fields_must_agree(extra: dict[str, object]) -> None:
    raw = entry("fact-1")
    raw.update(extra)

    assert answers_rule(with_answer(0, raw)) is DatasetRule.WARNING_FACTS


def test_warning_fields_are_type_checked() -> None:
    bad_flag = entry("fact-1", warning_expected="yes", warning_facts=[["unplug"]])
    bad_group = entry("fact-1", warning_expected=True, warning_facts=[[1]])

    assert answers_rule(with_answer(0, bad_flag)) is DatasetRule.WRONG_TYPE
    assert answers_rule(with_answer(0, bad_group)) is DatasetRule.WRONG_TYPE


def test_warning_facts_follow_the_key_fact_rules() -> None:
    raw = entry("fact-1", warning_expected=True, warning_facts=[["re:(bad"]])

    assert answers_rule(with_answer(0, raw)) is DatasetRule.INVALID_PATTERN


def test_every_entry_must_name_a_retrieval_question() -> None:
    assert answers_rule([*valid_answers(), entry("fact-9")]) is DatasetRule.UNKNOWN_ID


def test_every_retrieval_question_needs_an_entry() -> None:
    assert answers_rule(valid_answers()[:-1]) is DatasetRule.MISSING_ENTRY


def test_an_entry_keeps_its_retrieval_split() -> None:
    assert answers_rule(with_answer(0, entry("fact-1", "heldout"))) is DatasetRule.SPLIT_MISMATCH


def test_key_facts_follow_the_question_scope() -> None:
    oos_with_facts = with_answer(4, entry("oos-1", key_facts=[["alpha"]]))
    in_scope_without = with_answer(0, entry("fact-1", key_facts=[]))

    assert answers_rule(oos_with_facts) is DatasetRule.KEY_FACTS_SCOPE
    assert answers_rule(in_scope_without) is DatasetRule.KEY_FACTS_SCOPE


@pytest.mark.parametrize(
    "key_facts",
    [[[CANARY * 5]], [[f"re:({CANARY}"]], [[f"re:{CANARY.upper()}"]]],
    ids=["too-long", "bad-regex", "upper"],
)
def test_error_messages_name_ids_never_facts(key_facts: object) -> None:
    error = failure(lambda: answers(with_answer(0, entry("fact-1", key_facts=key_facts))))

    assert "fact-1" in str(error)
    assert CANARY not in str(error).lower()


# ================================================================ slices-v1


def with_slice(position: int, raw: Raw) -> list[Raw]:
    questions = valid_slices()
    questions[position] = raw
    return questions


def test_a_valid_slices_file_parses_each_slice() -> None:
    data = render(valid_slices())

    parsed = parse_slices(data, GOLDEN, DOC_IDS, max_turn_chars=MAX_TURN_CHARS)

    assert parsed.version == "1.0.0"
    assert parsed.sha256 == hashlib.sha256(data).hexdigest()
    mt, lq, _, oos, adv, adv_in = parsed.items
    assert (mt.slice, mt.split, mt.match, mt.rewrite_trap) == (
        Slice.MULTI_TURN,
        Split.DEV,
        Match.ANY,
        True,
    )
    assert mt.turns == ("My printer jams. What now?", "Which area first?")
    assert mt.anchors == (Anchor("demo-guide", ("Part 123",)),)
    assert mt.key_facts == (("paper tray",),)
    assert mt.rewrite_forbidden == ("(?i)envy",)
    assert (lq.slice, lq.pair_of, lq.key_facts, lq.anchors) == (
        Slice.LONG_QUERY,
        "fact-1",
        (),
        (),
    )
    assert (oos.slice, oos.oos_kind) == (Slice.OUT_OF_SCOPE, OosKind.NEAR_DOMAIN)
    assert (adv.attack, adv.expected, adv.forbidden, adv.key_facts) == (
        Attack.INJECTION_DIRECT,
        Expected.REFUSE,
        ("(?i)\\bpwned\\b",),
        (),
    )
    assert (adv_in.attack, adv_in.expected, adv_in.key_facts) == (
        Attack.FORMAT_HIJACK,
        Expected.ANSWER_IN_SCOPE,
        (("re:\\b80\\b",),),
    )


def test_rewrite_forbidden_is_optional() -> None:
    raw = multi_turn()
    del raw["rewrite_forbidden"]

    assert slices(with_slice(0, raw)).items[0].rewrite_forbidden == ()


def test_the_slice_is_required_and_known() -> None:
    raw = out_of_scope()
    del raw["slice"]

    assert slices_rule(with_slice(3, raw)) is DatasetRule.MISSING_KEY
    assert slices_rule(with_slice(3, out_of_scope(slice="followup"))) is DatasetRule.UNKNOWN_VALUE


@pytest.mark.parametrize(
    ("position", "raw"),
    [
        (0, multi_turn(pair_of="fact-1")),
        (1, long_query(key_facts=[["alpha"]])),
        (1, long_query(anchors=[anchor()])),
        (3, out_of_scope(attack="roleplay")),
        (4, adversarial(oos_kind="unrelated")),
        (4, adversarial(match="any")),
        (0, multi_turn(notes="x")),
    ],
    ids=["mt-pair", "lq-facts", "lq-anchors", "oos-attack", "adv-kind", "adv-match", "unknown"],
)
def test_a_key_of_another_slice_is_unknown(position: int, raw: Raw) -> None:
    assert slices_rule(with_slice(position, raw)) is DatasetRule.UNKNOWN_KEY


@pytest.mark.parametrize(
    ("position", "make", "key"),
    [
        (0, multi_turn, "anchors"),
        (0, multi_turn, "match"),
        (0, multi_turn, "key_facts"),
        (0, multi_turn, "rewrite_trap"),
        (1, long_query, "pair_of"),
        (3, out_of_scope, "oos_kind"),
        (4, adversarial, "attack"),
        (4, adversarial, "expected"),
        (4, adversarial, "forbidden"),
        (4, adversarial, "turns"),
        (4, adversarial, "id"),
        (4, adversarial, "split"),
    ],
)
def test_each_required_slice_key_is_checked(
    position: int, make: Callable[[], Raw], key: str
) -> None:
    raw = make()
    del raw[key]

    assert slices_rule(with_slice(position, raw)) is DatasetRule.MISSING_KEY


@pytest.mark.parametrize(
    ("position", "raw"),
    [
        (0, multi_turn(match="some")),
        (3, out_of_scope(kind="adjacent")),
        (4, adversarial(attack="jailbreak")),
        (4, adversarial(expected="ignore")),
        (4, adversarial(split="test")),
    ],
    ids=["match", "oos-kind", "attack", "expected", "split"],
)
def test_an_unknown_enum_value_is_rejected(position: int, raw: Raw) -> None:
    assert slices_rule(with_slice(position, raw)) is DatasetRule.UNKNOWN_VALUE


@pytest.mark.parametrize(
    ("position", "raw", "rule"),
    [
        (4, adversarial(turns="one turn"), DatasetRule.WRONG_TYPE),
        (4, adversarial(turns=[1]), DatasetRule.WRONG_TYPE),
        (4, adversarial(turns=[]), DatasetRule.TURN_COUNT),
        (4, adversarial(turns=["a?", "b?", "c?", "d?", "e?"]), DatasetRule.TURN_COUNT),
        (0, multi_turn(turns=["Only one turn?"]), DatasetRule.TURN_COUNT),
        (4, adversarial(turns=["   "]), DatasetRule.BLANK_TURN),
        (4, adversarial(turns=["x" * (MAX_TURN_CHARS + 1)]), DatasetRule.TURN_TOO_LONG),
    ],
    ids=["string", "number", "none", "five", "mt-one", "blank", "too-long"],
)
def test_turns_are_checked(position: int, raw: Raw, rule: DatasetRule) -> None:
    assert slices_rule(with_slice(position, raw)) is rule


def test_four_turns_and_the_length_limit_after_trimming_are_allowed() -> None:
    longest = "  " + "x" * MAX_TURN_CHARS + "  "
    raw = multi_turn(turns=["a?", "b?", "c?", longest])

    assert slices(with_slice(0, raw)).items[0].turns[-1] == longest


@pytest.mark.parametrize(
    ("position", "raw", "rule"),
    [
        (4, adversarial(forbidden=[]), DatasetRule.EMPTY_GROUP),
        (4, adversarial(forbidden="(?i)x"), DatasetRule.WRONG_TYPE),
        (4, adversarial(forbidden=["(unclosed"]), DatasetRule.INVALID_PATTERN),
        (4, adversarial(forbidden=[" "]), DatasetRule.INVALID_PATTERN),
        (0, multi_turn(rewrite_forbidden=["[bad"]), DatasetRule.INVALID_PATTERN),
        (0, multi_turn(rewrite_forbidden=[3]), DatasetRule.WRONG_TYPE),
        (0, multi_turn(rewrite_trap="yes"), DatasetRule.WRONG_TYPE),
        (0, multi_turn(key_facts=[["re:(bad"]]), DatasetRule.INVALID_PATTERN),
        (0, multi_turn(key_facts=[]), DatasetRule.KEY_FACTS_SCOPE),
    ],
    ids=[
        "forbidden-empty",
        "forbidden-string",
        "forbidden-regex",
        "forbidden-blank",
        "rewrite-regex",
        "rewrite-type",
        "trap-type",
        "mt-fact-regex",
        "mt-no-facts",
    ],
)
def test_patterns_flags_and_facts_are_checked(position: int, raw: Raw, rule: DatasetRule) -> None:
    assert slices_rule(with_slice(position, raw)) is rule


def test_adversarial_key_facts_follow_the_expected_outcome() -> None:
    in_scope_without = adversarial("adv-2", "answer_in_scope")
    del in_scope_without["key_facts"]
    refuse_with = adversarial(key_facts=[["alpha"]])

    assert slices_rule(with_slice(5, in_scope_without)) is DatasetRule.KEY_FACTS_SCOPE
    assert slices_rule(with_slice(4, refuse_with)) is DatasetRule.KEY_FACTS_SCOPE


@pytest.mark.parametrize(
    "anchors",
    [[], [anchor("unknown-guide")], [anchor(facts=())], "anchor"],
    ids=["none", "unknown-doc", "no-facts", "string"],
)
def test_multi_turn_anchors_follow_the_retrieval_rules(anchors: object) -> None:
    assert slices_rule(with_slice(0, multi_turn(anchors=anchors))) is DatasetRule.INVALID_ANCHOR


def test_slice_ids_are_unique_across_both_files() -> None:
    assert slices_rule([*valid_slices(), out_of_scope("mt-1")]) is DatasetRule.DUPLICATE_ID
    assert slices_rule(with_slice(3, out_of_scope("fact-1"))) is DatasetRule.DUPLICATE_ID
    assert slices_rule(with_slice(3, out_of_scope("Bad Id"))) is DatasetRule.INVALID_ID


def test_a_long_query_pair_must_resolve_to_an_in_scope_question_of_its_split() -> None:
    assert slices_rule(with_slice(1, long_query(pair_of="fact-9"))) is DatasetRule.UNRESOLVED_PAIR
    assert slices_rule(with_slice(1, long_query(pair_of="oos-1"))) is DatasetRule.PAIR_OUT_OF_SCOPE
    assert slices_rule(with_slice(1, long_query(pair_of="fact-held"))) is DatasetRule.SPLIT_MISMATCH


def test_the_slices_file_checks_its_top_level_like_the_answers_file() -> None:
    assert slices_rule(valid_slices(), extra=1) is DatasetRule.UNKNOWN_KEY
    assert slices_rule([]) is DatasetRule.NO_QUESTIONS
    assert slices_rule(valid_slices(), version="1.0") is DatasetRule.INVALID_VERSION


def test_slice_errors_name_ids_never_turns_or_patterns() -> None:
    raw = adversarial(turns=[CANARY * 10], forbidden=[f"({CANARY}"])

    error = failure(lambda: slices(with_slice(4, raw)))

    assert "adv-1" in str(error)
    assert CANARY not in str(error)


# ================================================================ items


def test_items_come_answers_first_in_retrieval_order_then_slices_in_file_order() -> None:
    built = build_items(GOLDEN, answers(valid_answers()), slices(valid_slices()))

    assert [item.id for item in built] == [
        *(q.id for q in GOLDEN.questions),
        "mt-1",
        "lq-1",
        "lq-held",
        "oos2-1",
        "adv-1",
        "adv-2",
    ]


def test_an_answers_item_is_one_turn_of_its_retrieval_question() -> None:
    built = build_items(GOLDEN, answers(valid_answers()), slices(valid_slices()))
    fact, oos = built[0], built[4]

    assert fact.turns == ("Question fact-1?",)
    assert (fact.slice, fact.category, fact.match, fact.in_scope) == (
        None,
        Category.FACTUAL,
        Match.ANY,
        True,
    )
    assert fact.anchors == GOLDEN.questions[0].anchors
    assert fact.key_facts == (("alpha", "re:\\bbeta\\b"), ("gamma",))
    assert fact.primary
    assert (oos.in_scope, oos.oos_kind, oos.key_facts, oos.primary) == (
        False,
        OosKind.UNRELATED,
        (),
        False,
    )
    assert built[2].warning_facts == ("unplug", "power cord")


def test_a_long_query_inherits_its_pair() -> None:
    by_id = {
        item.id: item
        for item in build_items(
            GOLDEN,
            answers(
                with_answer(0, entry("fact-1", warning_expected=True, warning_facts=[["cord"]]))
            ),
            slices(valid_slices()),
        )
    }
    lq, pair = by_id["lq-1"], by_id["fact-1"]

    assert lq.turns == ("A long story first, then the question?",)
    assert (lq.slice, lq.pair_of, lq.in_scope, lq.primary) == (
        Slice.LONG_QUERY,
        "fact-1",
        True,
        False,
    )
    assert (lq.category, lq.anchors, lq.match, lq.key_facts, lq.warning_facts) == (
        pair.category,
        pair.anchors,
        pair.match,
        pair.key_facts,
        ("cord",),
    )


def test_slice_items_expect_an_answer_only_when_in_scope() -> None:
    by_id = {
        item.id: item
        for item in build_items(GOLDEN, answers(valid_answers()), slices(valid_slices()))
    }

    assert by_id["mt-1"].in_scope
    assert by_id["mt-1"].rewrite_trap is True
    assert by_id["mt-1"].rewrite_forbidden == ("(?i)envy",)
    assert not by_id["oos2-1"].in_scope
    assert by_id["oos2-1"].oos_kind is OosKind.NEAR_DOMAIN
    assert not by_id["adv-1"].in_scope
    assert by_id["adv-1"].forbidden == ("(?i)\\bpwned\\b",)
    assert by_id["adv-2"].in_scope
    assert by_id["adv-2"].key_facts == (("re:\\b80\\b",),)
    assert not any(item.primary for item in by_id.values() if item.slice is not None)


# ================================================================ files


def _write(eval_dir: pathlib.Path) -> None:
    golden_dir = eval_dir / "golden"
    golden_dir.mkdir(parents=True)
    golden_lines = ['version = "1.0.1"']
    for q in GOLDEN.questions:
        golden_lines += [
            "",
            "[[questions]]",
            f'id = "{q.id}"',
            f'category = "{q.category.value}"',
            f'split = "{q.split.value}"',
            f'question = "{q.question}"',
        ]
        if q.in_scope:
            golden_lines.append('match = "any"')
        if q.oos_kind is not None:
            golden_lines.append(f'oos_kind = "{q.oos_kind.value}"')
        for a in q.anchors:
            golden_lines += ["", "[[questions.anchors]]", f'doc_id = "{a.doc_id}"']
            golden_lines.append(f'facts = ["{a.facts[0]}"]')
    (golden_dir / "retrieval-v1.toml").write_text("\n".join(golden_lines) + "\n")
    (eval_dir / ANSWERS_FILE).write_bytes(render(valid_answers()))
    (eval_dir / SLICES_FILE).write_bytes(render(valid_slices()))


def test_load_dataset_reads_the_three_files_and_records_their_identity(
    tmp_path: pathlib.Path,
) -> None:
    _write(tmp_path)

    dataset = load_dataset(tmp_path, DOC_IDS, max_turn_chars=MAX_TURN_CHARS)

    names = [f.name for f in dataset.files]
    assert names == ["retrieval-v1.toml", "answers-v1.toml", "slices-v1.toml"]
    for file in dataset.files:
        digest = hashlib.sha256((tmp_path / "golden" / file.name).read_bytes()).hexdigest()
        assert file.sha256 == digest
    assert [f.version for f in dataset.files] == ["1.0.1", "1.0.0", "1.0.0"]
    assert len(dataset.items) == len(GOLDEN.questions) + len(valid_slices())
    assert {i.split for i in dataset.in_split(Split.HELDOUT)} == {Split.HELDOUT}
    assert [i.id for i in dataset.in_split(Split.HELDOUT)] == ["fact-held", "oos-held", "lq-held"]


@pytest.mark.parametrize("missing", [ANSWERS_FILE, SLICES_FILE])
def test_an_unreadable_dataset_file_is_reported(
    tmp_path: pathlib.Path, missing: pathlib.Path
) -> None:
    _write(tmp_path)
    (tmp_path / missing).unlink()

    error = failure(lambda: load_dataset(tmp_path, DOC_IDS, max_turn_chars=MAX_TURN_CHARS))

    assert error.rule is DatasetRule.UNREADABLE
    assert missing.name in str(error)


def test_the_retrieval_file_is_the_one_the_phase_2_runner_reads() -> None:
    from app.evaluation.cli import GOLDEN_FILE  # noqa: PLC0415 - the composition root, heavy

    assert RETRIEVAL_FILE == GOLDEN_FILE

"""Key-fact and warning matching on answers (Phase 5 prompt, section 5, matching rules).

Answers are written the way the chat API returns them: Markdown, numbered steps,
citation tags such as [S1] and [S1, S3], typographic dashes and non-breaking spaces.
"""

import pytest

from app.evaluation.answers.matching import (
    KeyFactResult,
    alternative_matches,
    match_key_facts,
    normalize_answer,
    warning_carried,
    warning_lines,
)

NBSP = chr(0xA0)
EN_DASH = chr(0x2013)
EM_DASH = chr(0x2014)


# ---------------------------------------------------------------- normalization


@pytest.mark.parametrize(
    ("answer", "normalized"),
    [
        ("Load paper [S1].", "load paper."),
        ("Load paper [S1, S3] first.", "load paper first."),
        ("Load paper [S1][S3] first.", "load paper first."),
        ("Load paper [ S2 ,S4 ] first.", "load paper first."),
        ("Press **Power** and `OK`.", "press power and ok."),
        ("Press *Power* then __Start__ or _Stop_.", "press power then start or stop."),
        ("Use the  Wi-Fi\n\n  menu", "use the wi-fi menu"),
        (f"pages 12{EN_DASH}13 and 20{EM_DASH}21", "pages 12-13 and 20-21"),
        (f"80{NBSP}sheets", "80 sheets"),
        ("STRASSE Straße", "strasse strasse"),
    ],
    ids=[
        "tag",
        "group",
        "adjacent",
        "spaced-group",
        "bold-code",
        "italic-underscore",
        "whitespace",
        "dashes",
        "nbsp",
        "casefold",
    ],
)
def test_the_answer_is_normalized_before_matching(answer: str, normalized: str) -> None:
    assert normalize_answer(answer) == normalized


def test_underscores_inside_a_word_are_kept() -> None:
    assert normalize_answer("Open HP_Smart_setup now") == "open hp_smart_setup now"


def test_square_brackets_that_are_not_citation_tags_are_kept() -> None:
    assert normalize_answer("Press [OK] or [s1x]") == "press [ok] or [s1x]"


# ---------------------------------------------------------------- alternatives


def test_a_plain_alternative_is_normalized_and_matched_as_a_substring() -> None:
    answer = normalize_answer(f"Remove the **six** Phillips M2.0{EN_DASH}3.0 screws [S2].")

    assert alternative_matches("Phillips M2.0-3.0", answer)
    assert alternative_matches("SIX phillips", answer)
    assert not alternative_matches("seven Phillips", answer)


def test_a_regex_alternative_is_searched_in_the_normalized_answer() -> None:
    answer = normalize_answer("The tray holds **80** sheets [S1].")

    assert alternative_matches("re:\\b80\\b", answer)
    assert alternative_matches("re:holds \\d+ sheets", answer)
    assert not alternative_matches("re:\\b800\\b", answer)


def test_a_regex_never_sees_upper_case_or_markup() -> None:
    answer = normalize_answer("Hold **Cancel** for 3 seconds [S4].")

    assert alternative_matches("re:hold cancel for \\d seconds\\.$", answer)


def test_a_substring_inside_a_longer_number_still_matches_a_plain_alternative() -> None:
    """Plain alternatives are substrings by rule; a whole number needs a re: pattern."""
    answer = normalize_answer("It holds 180 sheets.")

    assert alternative_matches("80", answer)
    assert not alternative_matches("re:\\b80\\b", answer)


# ---------------------------------------------------------------- groups


def test_an_answer_passes_when_every_group_has_a_matching_alternative() -> None:
    result = match_key_facts(
        (("paper tray", "input tray"), ("re:\\bfirst\\b",)),
        "Clear the **input tray** first [S1].",
    )

    assert result == KeyFactResult((True, True))
    assert result.passed
    assert result.coverage == 1.0


def test_an_answer_fails_when_one_group_has_no_match() -> None:
    result = match_key_facts(
        (("paper tray",), ("rear door",), ("re:\\bfirst\\b",)),
        "Clear the paper tray first [S1].",
    )

    assert result.groups == (True, False, True)
    assert not result.passed
    assert result.coverage == pytest.approx(2 / 3)


def test_an_answer_with_no_group_matching_scores_zero_coverage() -> None:
    result = match_key_facts((("alpha",), ("beta",)), "Nothing relevant here.")

    assert not result.passed
    assert result.coverage == 0.0


def test_no_groups_pass_vacuously_with_no_coverage() -> None:
    """Out-of-scope questions carry no key facts; their score is the refusal."""
    result = match_key_facts((), "Anything.")

    assert result.passed
    assert result.coverage is None


def test_a_fact_split_by_a_citation_tag_still_matches() -> None:
    result = match_key_facts((("six phillips m2.0 screws",),), "Six Phillips [S2] M2.0 screws.")

    assert result.passed


# ---------------------------------------------------------------- warnings


@pytest.mark.parametrize(
    "line",
    [
        "WARNING: Unplug the power cord first [S2].",
        "CAUTION: Unplug the power cord first.",
        "**WARNING:** Unplug the power cord first.",
        "- **CAUTION:** Unplug the power cord first.",
        "* WARNING: Unplug the power cord first.",
        "+ WARNING: Unplug the power cord first.",
        "2. WARNING: Unplug the power cord first.",
        "3) _CAUTION:_ Unplug the power cord first.",
        "   WARNING: Unplug the power cord first.",
    ],
    ids=[
        "plain",
        "caution",
        "bold",
        "dash-bold",
        "star",
        "plus",
        "numbered",
        "paren-italic",
        "indent",
    ],
)
def test_a_warning_line_starts_with_the_upper_case_label(line: str) -> None:
    answer = f"Do this:\n1. Turn the printer off.\n{line}\n2. Open the door."

    assert warning_lines(answer) == (line,)
    assert warning_carried(answer, ("unplug", "power cord"))


@pytest.mark.parametrize(
    "line",
    [
        "Warning: Unplug the power cord first.",
        "warning: unplug the power cord first.",
        "Note: WARNING: unplug the power cord.",
        "WARNING Unplug the power cord first.",
        "> WARNING: Unplug the power cord first.",
    ],
    ids=["title-case", "lower-case", "not-at-start", "no-colon", "quote"],
)
def test_a_line_without_the_upper_case_label_at_its_start_is_not_a_warning(line: str) -> None:
    answer = f"Steps:\n{line}"

    assert warning_lines(answer) == ()
    assert not warning_carried(answer, ("unplug", "power cord"))


def test_the_warning_facts_must_be_on_the_warning_line() -> None:
    answer = "WARNING: Be careful.\nUnplug the power cord first [S1]."

    assert warning_lines(answer) == ("WARNING: Be careful.",)
    assert not warning_carried(answer, ("unplug", "power cord"))


def test_any_warning_line_may_carry_the_facts() -> None:
    answer = "CAUTION: Handle with care.\n1. Step.\nWARNING: Disconnect the **power cord** [S3]."

    assert warning_carried(answer, ("power cord",))


def test_a_regex_warning_fact_runs_on_the_normalized_line() -> None:
    answer = "**WARNING:** Do not use **sharp** tools [S2]."

    assert warning_carried(answer, ("re:\\bsharp\\b",))
    assert not warning_carried(answer, ("re:\\bpliers\\b",))


def test_an_answer_without_lines_has_no_warning() -> None:
    assert warning_lines("") == ()
    assert not warning_carried("", ("unplug",))

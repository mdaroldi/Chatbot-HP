"""Citation tags: validation against the retrieved set, stripping, source snippets."""

import pytest

from app.domain.chat import REFUSAL_MARKER
from app.domain.citations import Citations, snippet, strip_citations, validate_citations

RETRIEVED = {"S1", "S2", "S3"}


# --- Validation -----------------------------------------------------------------------


def test_a_single_valid_tag_is_kept_and_reported() -> None:
    assert validate_citations("Open the cover [S2].", RETRIEVED) == Citations(
        "Open the cover [S2].", ("S2",), 0
    )


def test_adjacent_tags_are_each_kept() -> None:
    assert validate_citations("Open it [S1][S3].", RETRIEVED) == Citations(
        "Open it [S1][S3].", ("S1", "S3"), 0
    )


@pytest.mark.parametrize("group", ["[S1, S3]", "[S1,S3]", "[ S1 ,  S3 ]"])
def test_a_comma_separated_group_with_any_spacing_is_kept_unchanged(group: str) -> None:
    text = f"Open it {group}."

    assert validate_citations(text, RETRIEVED) == Citations(text, ("S1", "S3"), 0)


def test_a_tag_with_spaces_inside_the_brackets_is_a_citation() -> None:
    assert validate_citations("Open it [ S2 ].", RETRIEVED).tags == ("S2",)


def test_a_tag_outside_the_retrieved_set_is_removed_with_the_space_before_it() -> None:
    assert validate_citations("Open it [S1]. Then close it [S7].", RETRIEVED) == Citations(
        "Open it [S1]. Then close it.", ("S1",), 1
    )


def test_inside_a_group_only_the_bad_members_are_removed() -> None:
    assert validate_citations("Open it [S1, S7, S3, S9].", RETRIEVED) == Citations(
        "Open it [S1, S3].", ("S1", "S3"), 2
    )


def test_a_group_left_with_one_member_becomes_a_single_tag() -> None:
    assert validate_citations("Open it [S9, S2].", RETRIEVED) == Citations(
        "Open it [S2].", ("S2",), 1
    )


def test_a_group_with_no_valid_member_is_removed_entirely() -> None:
    assert validate_citations("Open it [S8, S9].", RETRIEVED) == Citations("Open it.", (), 2)


def test_every_invalid_occurrence_is_counted() -> None:
    assert validate_citations("A [S0]. B [S0]. C [S4][S5].", RETRIEVED).invalid == 4


def test_duplicate_tags_are_reported_once_in_tag_number_order() -> None:
    result = validate_citations("A [S3]. B [S1][S3]. C [S1, S3].", RETRIEVED)

    assert result.tags == ("S1", "S3")
    assert result.text == "A [S3]. B [S1][S3]. C [S1, S3]."


def test_tags_are_ordered_by_number_not_as_text() -> None:
    assert validate_citations("A [S10]. B [S2].", {"S2", "S10"}).tags == ("S2", "S10")


def test_an_answer_without_tags_is_unchanged_and_cites_nothing() -> None:
    assert validate_citations("No tags here.", RETRIEVED) == Citations("No tags here.", (), 0)


@pytest.mark.parametrize(
    "text", ["See [S] here.", "See [s1] here.", "See [see S1] here.", REFUSAL_MARKER, "[S1"]
)
def test_text_that_only_resembles_a_tag_is_not_a_citation(text: str) -> None:
    assert validate_citations(text, RETRIEVED) == Citations(text, (), 0)


# --- Stripping ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "stripped"),
    [
        ("Open the cover [S1].", "Open the cover."),
        ("Open the cover [S1][S3].", "Open the cover."),
        ("Open the cover [S1, S3].", "Open the cover."),
        ("Open the cover\t[ S2 ] now.", "Open the cover now."),
        ("1. Lift it [S1]\n2. Pull it [S9]", "1. Lift it\n2. Pull it"),
        ("No tags [see S1].", "No tags [see S1]."),
    ],
)
def test_stripping_removes_every_tag_valid_or_not(text: str, stripped: str) -> None:
    assert strip_citations(text) == stripped


def test_stripping_keeps_the_newline_before_a_tag_at_the_start_of_a_line() -> None:
    assert strip_citations("Done.\n[S1] Next.") == "Done.\n Next."


# --- Snippets -------------------------------------------------------------------------


def test_a_snippet_starts_after_the_breadcrumb() -> None:
    text = "Guide > Chapter > Section\n\nLift the cover. Then remove it."

    assert snippet(text, 160) == "Lift the cover. Then remove it."


def test_a_snippet_collapses_whitespace_and_line_breaks() -> None:
    assert (
        snippet("Guide > Section\n\n1. Lift\n  2.  Pull\n\nTable", 160) == "1. Lift 2. Pull Table"
    )


def test_a_text_without_a_breadcrumb_is_used_whole() -> None:
    assert snippet("Lift the cover.", 160) == "Lift the cover."


def test_a_body_that_fits_exactly_is_not_cut() -> None:
    assert snippet("Crumb\n\n" + "a" * 20, 20) == "a" * 20


def test_a_long_body_is_cut_at_a_word_boundary_with_an_ellipsis() -> None:
    assert snippet("Crumb\n\nLift the cover and remove it.", 16) == "Lift the cover…"


def test_a_cut_right_after_a_word_keeps_that_word() -> None:
    assert snippet("Crumb\n\nLift the cover and remove it.", 14) == "Lift the cover…"


def test_a_single_word_longer_than_the_limit_is_cut_inside_the_word() -> None:
    assert snippet("Crumb\n\n" + "x" * 30, 10) == "x" * 10 + "…"

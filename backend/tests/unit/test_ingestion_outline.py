"""Behavior of bookmark paths and heading matching (synthetic outline and lines)."""

import pytest

from app.domain.ingestion import OutlineEntry, TextLine
from app.ingestion.outline import HeadingMatch, clean_title, match_headings, section_paths


def entry(title: str, level: int = 1, page: int = 1) -> OutlineEntry:
    return OutlineEntry(level=level, title=title, pdf_page=page)


def line(text: str, page: int = 1, top: float = 100, size: float = 18) -> TextLine:
    return TextLine(text=text, pdf_page=page, font_size=size, x0=50, top=top, x1=90, bottom=top + 5)


# ------------------------------------------------------------------ titles and paths


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("Setup", "Setup"),
        ("1.\xa0\xa0\xa0Product description", "Product description"),
        ("10.\xa0Specifications", "Specifications"),
        ("  Wi-Fi   Direct ", "Wi-Fi Direct"),
        ("802.11 settings", "802.11 settings"),
    ],
)
def test_clean_title_drops_chapter_numbers_and_odd_spaces(raw: str, clean: str) -> None:
    assert clean_title(raw) == clean


def test_section_paths_follow_the_bookmark_tree() -> None:
    outline = (
        entry("1.\xa0Setup"),
        entry("Connect", 2),
        entry("Wireless", 3),
        entry("Wired", 3),
        entry("Panel", 2),
        entry("Deep", 4),
        entry("Print"),
        entry("Photos", 2),
    )

    assert section_paths(outline) == (
        ("Setup",),
        ("Setup", "Connect"),
        ("Setup", "Connect", "Wireless"),
        ("Setup", "Connect", "Wired"),
        ("Setup", "Panel"),
        ("Setup", "Panel", "Deep"),
        ("Print",),
        ("Print", "Photos"),
    )


# ------------------------------------------------------------------ headings


def test_heading_is_the_first_large_line_matching_the_title_on_its_page() -> None:
    lines = (
        line("Wireless", page=1, size=10),  # body mention before the target page
        line("Intro", page=2, top=50),
        line("Wireless", page=2, top=80, size=10),  # body text on the page: too small
        line("Wireless", page=2, top=120),
    )

    assert match_headings((entry("Wireless", page=2),), lines) == (
        HeadingMatch(first_line=3, last_line=3),
    )


@pytest.mark.parametrize("text", ["1 Setup", "12 Setup", "A Setup", "B. Setup", "SETUP", "Se tup"])
def test_chapter_labels_case_and_spacing_do_not_prevent_a_match(text: str) -> None:
    assert match_headings((entry("1.\xa0Setup"),), (line(text),)) == (HeadingMatch(0, 0),)


def test_a_label_is_only_skipped_before_the_title() -> None:
    assert match_headings((entry("Setup"),), (line("Setup 2"), line("AB Setup"))) == (None,)


def test_a_heading_wrapped_over_lines_of_one_size_is_matched_whole() -> None:
    lines = (
        line("Removal and replacement", top=100),
        line("procedures", top=120),
        line("Body", top=140, size=10),
    )

    matches = match_headings((entry("Removal and replacement procedures"),), lines)

    assert matches == (HeadingMatch(first_line=0, last_line=1),)


def test_wrapping_stops_at_a_size_change() -> None:
    lines = (line("Removal and", top=100), line("procedures", top=120, size=14))

    assert match_headings((entry("Removal and procedures"),), lines) == (None,)


def test_heading_may_start_on_the_next_page() -> None:
    lines = (line("Body", page=3, size=10), line("Setup", page=4))

    assert match_headings((entry("Setup", page=3),), lines) == (HeadingMatch(1, 1),)


def test_a_heading_two_pages_later_is_not_matched() -> None:
    assert match_headings((entry("Setup", page=3),), (line("Setup", page=5),)) == (None,)


def test_body_size_heading_is_accepted_on_its_own_page_when_no_large_line_matches() -> None:
    """Some deep bookmarks point at headings set in the body font size."""
    lines = (line("Turn on", page=2, size=10), line("Turn on", page=3, size=10))

    assert match_headings((entry("Turn on", page=2),), lines) == (HeadingMatch(0, 0),)
    assert match_headings((entry("Turn on", page=1),), lines) == (None,)


def test_matches_move_forward_so_repeated_titles_map_in_order() -> None:
    lines = (line("Overview", top=100), line("Body", top=120, size=10), line("Overview", top=140))
    outline = (entry("Overview"), entry("Overview", 2))

    assert match_headings(outline, lines) == (HeadingMatch(0, 0), HeadingMatch(2, 2))


def test_an_unmatched_entry_does_not_block_later_ones() -> None:
    lines = (line("Print", page=2),)

    assert match_headings((entry("Missing"), entry("Print", page=2)), lines) == (
        None,
        HeadingMatch(0, 0),
    )

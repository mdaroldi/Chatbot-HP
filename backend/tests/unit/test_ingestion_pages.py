"""Behavior of TOC, index and chapter-opener page detection (synthetic data)."""

from app.domain.ingestion import OutlineEntry, TextLine
from app.ingestion.pages import chapter_opener_pages, index_pages, toc_pages


def line(text: str, page: int = 1, top: float = 100) -> TextLine:
    return TextLine(text=text, pdf_page=page, font_size=10, x0=50, top=top, x1=90, bottom=top + 5)


def entry(title: str, level: int = 1, page: int = 1) -> OutlineEntry:
    return OutlineEntry(level=level, title=title, pdf_page=page)


# ------------------------------------------------------------------ table of contents


def leaders(page: int, count: int) -> list[TextLine]:
    return [line(f"Topic {n} ........ {n + 1}", page, top=100 + n) for n in range(count)]


def test_pages_mostly_made_of_dot_leader_lines_are_toc_pages() -> None:
    lines = (
        line("Contents", 3),
        *leaders(3, 3),
        line("wrapped entry without leader", 4),
        *leaders(4, 2),
        line("roman ...... iv", 4),
        *leaders(5, 1),
        line("Body", 5),
        line("Body", 5),
    )

    assert toc_pages(lines) == frozenset({3, 4})


def test_a_page_needs_three_leader_lines() -> None:
    assert toc_pages((*leaders(2, 2), *leaders(3, 3))) == frozenset({3})


def test_dots_inside_text_are_not_leaders() -> None:
    lines = tuple(line(f"Use 802.11 ... or Wi-Fi {n}x", 2, top=n) for n in range(4))

    assert toc_pages((*lines, *leaders(3, 3))) == frozenset({3})


# ------------------------------------------------------------------ index


def test_index_runs_from_its_bookmark_to_the_next_chapter_or_the_end() -> None:
    outline = (entry("Setup", page=1), entry("Index", page=8), entry("Back", page=10))

    assert index_pages(outline, page_count=12) == frozenset({8, 9})
    assert index_pages(outline[:2], page_count=9) == frozenset({8, 9})


def test_only_a_top_level_index_bookmark_counts() -> None:
    outline = (entry("Setup"), entry("Index", 2, page=3), entry("Back"), entry("INDEX", page=5))

    assert index_pages(outline, page_count=5) == frozenset({5})


# ------------------------------------------------------------------ chapter openers


def test_chapter_page_listing_its_sections_as_bullets_is_an_opener() -> None:
    outline = (
        entry("1.\xa0Setup", page=2),
        entry("Connect", 2, page=3),
        entry("Auto-Off", 2, page=4),
        entry("Detail", 3, page=4),
        entry("Print", page=6),
        entry("Photos", 2, page=6),
    )
    lines = (
        line("Setup", 2),
        line("● Connect", 2),
        line("● Auto-O ff", 2),  # broken ligature still matches
        line("• Detail", 2),  # level 3: not a chapter section
        line("● Photos", 6),
    )

    assert chapter_opener_pages(outline, lines) == frozenset({2})


def test_one_matching_bullet_is_not_enough_for_an_opener() -> None:
    outline = (
        entry("Setup", page=2),
        entry("Connect", 2, page=3),
        entry("Panel", 2, page=4),
        entry("Print", page=6),
        entry("Photos", 2, page=6),
        entry("Scans", 2, page=7),
    )
    lines = (
        line("● Connect", 2),
        line("● Unplug the cable", 2),
        line("Panel", 2),
        line("● Photos", 6),
        line("● Scans", 6),
    )

    assert chapter_opener_pages(outline, lines) == frozenset({6})


def test_a_hanging_bullet_pairs_with_the_text_beside_it() -> None:
    """With a hanging indent the bullet and its text are separate line segments."""
    outline = (entry("Setup", page=2), entry("Connect", 2, page=3), entry("Panel", 2, page=4))

    def at(text: str, x0: float, top: float, page: int = 2) -> TextLine:
        return TextLine(
            text=text, pdf_page=page, font_size=10, x0=x0, top=top, x1=x0 + 30, bottom=top + 5
        )

    split = (at("●", 100, 200), at("Connect", 122, 200), at("●", 100, 220), at("Panel", 122, 220))
    stray = (at("●", 100, 300), at("Connect", 122, 340), at("Panel", 122, 360))

    assert chapter_opener_pages(outline, split) == frozenset({2})
    assert chapter_opener_pages(outline, stray) == frozenset()

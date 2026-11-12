"""Behavior of front-matter sections: headed safety and notice parts before the first
bookmark are indexed; the cover and the unheaded legal text are not (synthetic data).
"""

from collections.abc import Sequence

from app.domain.ingestion import BlockKind, NoteLabel, OutlineEntry, TextLine
from app.ingestion.cleaning import clean
from app.ingestion.front_matter import FRONT_MATTER, HEADING_KEYWORDS, is_front_matter_heading
from app.ingestion.sections import SectionedDocument, build_sections

HEIGHT = 800.0
DOC = "test-guide"


def line(text: str, page: int, *, top: float = 100, size: float = 8) -> TextLine:
    return TextLine(
        text=text, pdf_page=page, font_size=size, x0=50, top=top, x1=500, bottom=top + size
    )


def sectioned(lines: Sequence[TextLine], pages: int = 4) -> SectionedDocument:
    """The lines, then the first bookmarked chapter on the last page."""
    chapter = (line("Setup", pages, top=60, size=24), line("Connect the cable.", pages))
    outline = (OutlineEntry(level=1, title="Setup", pdf_page=pages),)
    cleaned = clean((*lines, *chapter), outline, (HEIGHT,) * pages, ())
    return build_sections(DOC, cleaned, outline, ())


def front(result: SectionedDocument) -> list[tuple[tuple[str, ...], list[str]]]:
    return [
        (s.path, [b.text for b in s.blocks]) for s in result.sections if s.chapter == FRONT_MATTER
    ]


def test_headed_parts_become_front_matter_sections_and_the_rest_is_not_indexed() -> None:
    lines = (
        line("User Guide", 1, top=60, size=40),  # cover: large, but on the first page
        line("Printer notices", 2, top=60),  # same size as its text: not a heading
        line("© Copyright 2020 Example Company.", 2, top=80),
        line("Heat notice", 3, top=60, size=14),
        line("WARNING! Do not block the vents.", 3, top=90, size=10),
        line("Parts notice", 3, top=200, size=14),
        line("IMPORTANT: Use the listed parts only.", 3, top=230, size=10),
    )

    result = sectioned(lines)

    assert front(result) == [
        ((FRONT_MATTER, "Heat notice"), ["WARNING! Do not block the vents."]),
        ((FRONT_MATTER, "Parts notice"), ["IMPORTANT: Use the listed parts only."]),
    ]
    assert [ln.text for ln in result.unsectioned] == [
        "User Guide",
        "Printer notices",
        "© Copyright 2020 Example Company.",
    ]
    assert [s.path for s in result.sections][-1] == ("Setup",)


def test_front_matter_blocks_keep_notes_and_unnumbered_pages() -> None:
    lines = (line("Heat notice", 3, top=60, size=14), line("WARNING! Keep it cool.", 3, top=90))

    [section] = [s for s in sectioned(lines).sections if s.chapter == FRONT_MATTER]

    [block] = section.blocks
    assert (block.kind, block.note, block.is_hazard) == (BlockKind.NOTE, NoteLabel.WARNING, True)
    assert (block.pages.pdf_start, block.pages.printed_start) == (3, None)


def test_a_short_safety_line_is_a_heading_at_the_body_size() -> None:
    """The one keyword exception: one manual sets its safety heading in body size."""
    lines = (
        line("© Copyright 2020 Example Company.", 2, top=60),
        line("Safety information", 2, top=100),
        line("Always follow basic safety precautions.", 2, top=140),
        line("- Unplug the printer before cleaning.", 2, top=160),
    )

    result = sectioned(lines)

    assert front(result) == [
        (
            (FRONT_MATTER, "Safety information"),
            ["Always follow basic safety precautions.", "- Unplug the printer before cleaning."],
        )
    ]
    assert [ln.text for ln in result.unsectioned] == ["© Copyright 2020 Example Company."]


def test_the_keyword_rule_needs_a_short_unpunctuated_line_naming_safety() -> None:
    assert frozenset({"safety"}) == HEADING_KEYWORDS
    for text, expected in (
        ("Safety information", True),
        ("SAFETY", True),
        ("Important safety notes for users", True),  # five words
        ("Important safety notes for all users", False),  # six words
        ("Safety information.", False),  # sentence punctuation
        ("Read the safety notes:", False),
        ("Safetyinformation", False),  # not the word
        ("Printer notices", False),  # other keywords are not on the list
        ("Warning information", False),
    ):
        assert is_front_matter_heading(line(text, 2)) is expected, text


def test_the_size_rule_needs_heading_size_after_the_first_page() -> None:
    for text, page, size, expected in (
        ("Heat notice", 2, 12, True),
        ("A heading of any length set at heading size", 3, 14, True),
        ("Heat notice", 2, 11.9, False),
        ("User Guide", 1, 40, False),
        ("Safety information", 1, 8, False),  # the keyword rule skips the cover too
    ):
        assert is_front_matter_heading(line(text, page, size=size)) is expected, (text, page)


def test_a_heading_without_content_is_not_indexed() -> None:
    lines = (
        line("Heat notice", 3, top=60, size=14),
        line("Parts notice", 3, top=100, size=14),
        line("IMPORTANT: Use the listed parts only.", 3, top=130, size=10),
    )

    result = sectioned(lines)

    assert [path for path, _ in front(result)] == [(FRONT_MATTER, "Parts notice")]
    assert [ln.text for ln in result.unsectioned] == ["Heat notice"]

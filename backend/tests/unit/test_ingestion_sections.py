"""Behavior of section building: bookmark paths, blocks per section, unsectioned lines."""

from collections.abc import Sequence

from app.domain.ingestion import BlockKind, HorizontalRule, OutlineEntry, TextLine
from app.ingestion.cleaning import clean
from app.ingestion.sections import SectionedDocument, build_sections

HEIGHT = 800.0
DOC = "test-guide"


def line(
    text: str, page: int = 1, *, top: float = 100, size: float = 10, x0: float = 50
) -> TextLine:
    return TextLine(
        text=text, pdf_page=page, font_size=size, x0=x0, top=top, x1=x0 + 300, bottom=top + size
    )


def heading(title: str, page: int = 1, top: float = 60, size: float = 14) -> TextLine:
    return line(title, page, top=top, size=size)


def entry(title: str, level: int = 1, page: int = 1) -> OutlineEntry:
    return OutlineEntry(level=level, title=title, pdf_page=page)


def sectioned(
    lines: Sequence[TextLine],
    outline: Sequence[OutlineEntry],
    pages: int,
    rules: Sequence[HorizontalRule] = (),
) -> SectionedDocument:
    cleaned = clean(lines, outline, (HEIGHT,) * pages, rules)
    return build_sections(DOC, cleaned, outline, rules)


def test_paths_keep_ancestors_whose_heading_was_dropped_as_empty() -> None:
    """A chapter left with only its heading, and a section whose heading is directly
    followed by its first subsection, both lose their heading line as empty; their
    descendants still carry their titles.
    """
    lines = (
        heading("Setup", 1, size=24),
        heading("Connect", 2),
        line("Connect the cable to the port.", 2),
        heading("Solve", 3, size=24),
        line("Most problems have simple fixes.", 3),
        heading("Maintain", 4, size=18),
        heading("Clean", 4, top=100),
        line("Wipe the glass with a soft cloth.", 4, top=130),
    )
    outline = (
        entry("Setup", 1, 1),
        entry("Connect", 2, 2),
        entry("Solve", 1, 3),
        entry("Maintain", 2, 4),
        entry("Clean", 3, 4),
    )

    result = sectioned(lines, outline, pages=4)

    assert [(s.path, [b.text for b in s.blocks]) for s in result.sections] == [
        (("Setup", "Connect"), ["Connect the cable to the port."]),
        (("Solve",), ["Most problems have simple fixes."]),
        (("Solve", "Maintain", "Clean"), ["Wipe the glass with a soft cloth."]),
    ]
    assert {s.doc_id for s in result.sections} == {DOC}


def test_lines_before_the_first_heading_are_unsectioned() -> None:
    lines = (
        line("User guide", 1, top=300, size=32),
        heading("Setup", 2),
        line("Connect the cable to the port.", 2),
    )

    result = sectioned(lines, (entry("Setup", page=2),), pages=2)

    assert [ln.text for ln in result.unsectioned] == ["User guide"]
    assert [s.path for s in result.sections] == [("Setup",)]


def test_ruled_tables_become_table_blocks_of_their_section() -> None:
    def cell(text: str, top: float, x0: float) -> TextLine:
        return line(text, top=top, size=8, x0=x0)

    lines = (
        heading("Codes"),
        line("The panel shows these codes.", top=90),
        *(cell("Code", 112, 60), cell("Meaning", 112, 300)),
        *(cell("E1", 132, 60), cell("Paper jam", 132, 300)),
        *(cell("E2", 152, 60), cell("No ink", 152, 300)),
    )
    rules = tuple(HorizontalRule(pdf_page=1, y=y, x0=50, x1=500) for y in (108, 128, 148, 168))

    [section] = sectioned(lines, (entry("Codes"),), pages=1, rules=rules).sections

    assert [(b.kind, b.text) for b in section.blocks] == [
        (BlockKind.PARAGRAPH, "The panel shows these codes."),
        (BlockKind.TABLE_HEADER, "Code | Meaning"),
        (BlockKind.TABLE_ROW, "E1 | Paper jam"),
        (BlockKind.TABLE_ROW, "E2 | No ink"),
    ]


def test_printed_pages_come_from_the_footers() -> None:
    lines = (
        heading("Setup", 1),
        line("Connect the cable to the port.", 1),
        line("iv", 1, top=760),
        line("Plug in the power cord.", 2),
        line("12", 2, top=760),
    )

    [section] = sectioned(lines, (entry("Setup"),), pages=2).sections

    assert [(b.pages.pdf_start, b.pages.printed_start) for b in section.blocks] == [
        (1, None),
        (2, 12),
    ]

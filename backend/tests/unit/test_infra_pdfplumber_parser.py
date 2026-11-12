"""Behavior of the pdfplumber adapter on synthetic PDFs built by tests/pdf_builder.py."""

import pathlib

import pytest

from app.domain.ingestion import (
    CorpusDocument,
    DocumentParseError,
    OutlineEntry,
    ParsedDocument,
)
from app.infra.pdfplumber_parser import PdfplumberParser
from tests.pdf_builder import Bookmark, Page, Rule, Text, VerticalRule, build_pdf

DOC = CorpusDocument(
    doc_id="demo-guide",
    title="Demo Guide",
    product="Demo Printer 100",
    url="https://docs.test/a.pdf",
    sha256="a" * 64,
)

PAGES = (
    Page(
        texts=(Text(54, 70, 18, "Setup"), Text(90, 120, 10, "Open the tray.")),
        rules=(Rule(150, 90, 300), Rule(170, 90, 200, width=1.25), Rule(170, 200, 300)),
        verticals=(VerticalRule(200, 150, 170),),
    ),
    Page(texts=(Text(54, 70, 14, "Wireless"), Text(30, 560, 8, "2 Setup")), height=600),
)
BOOKMARKS = (
    Bookmark("1.\xa0Setup", 1, style="dest"),
    Bookmark("Wireless", 2, level=2, style="goto"),
    Bookmark("Wired", 2, level=3, style="named"),
    Bookmark("Index", 2, style="goto"),
)


def parse(
    tmp_path: pathlib.Path, pages: tuple[Page, ...], marks: tuple[Bookmark, ...]
) -> ParsedDocument:
    path = tmp_path / "demo.pdf"
    path.write_bytes(build_pdf(pages, marks))
    return PdfplumberParser().parse(path, DOC)


def test_words_carry_page_font_size_and_top_down_geometry(tmp_path: pathlib.Path) -> None:
    parsed = parse(tmp_path, PAGES, BOOKMARKS)

    assert parsed.document == DOC
    assert [(w.text, w.pdf_page, w.font_size) for w in parsed.words] == [
        ("Setup", 1, 18),
        ("Open", 1, 10),
        ("the", 1, 10),
        ("tray.", 1, 10),
        ("Wireless", 2, 14),
        ("2", 2, 8),
        ("Setup", 2, 8),
    ]
    first = parsed.words[0]
    assert first.x0 == pytest.approx(54, abs=0.5)
    assert first.top == pytest.approx(70, abs=1.5)
    assert first.bottom > first.top
    assert parsed.words[-2].top == pytest.approx(560, abs=1.5)


def test_horizontal_segments_become_rules_and_vertical_lines_are_ignored(
    tmp_path: pathlib.Path,
) -> None:
    parsed = parse(tmp_path, PAGES, BOOKMARKS)

    assert sorted((r.pdf_page, round(r.y, 1), r.x0, r.x1) for r in parsed.rules) == [
        (1, 150.0, 90, 300),
        (1, 170.0, 90, 200),
        (1, 170.0, 200, 300),
    ]


def test_pages_keep_their_own_heights(tmp_path: pathlib.Path) -> None:
    parsed = parse(tmp_path, PAGES, BOOKMARKS)

    assert parsed.page_heights == (792.0, 600.0)
    assert parsed.page_count == 2


def test_every_bookmark_style_resolves_to_its_page_with_raw_title_and_level(
    tmp_path: pathlib.Path,
) -> None:
    parsed = parse(tmp_path, PAGES, BOOKMARKS)

    assert parsed.outline == (
        OutlineEntry(level=1, title="1.\xa0Setup", pdf_page=1),
        OutlineEntry(level=2, title="Wireless", pdf_page=2),
        OutlineEntry(level=3, title="Wired", pdf_page=2),
        OutlineEntry(level=1, title="Index", pdf_page=2),
    )


def test_a_pdf_without_bookmarks_is_a_parse_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(DocumentParseError, match="demo-guide: no bookmarks"):
        parse(tmp_path, PAGES, ())


@pytest.mark.parametrize("style", ["uri", "named-missing"])
def test_a_bookmark_that_does_not_reach_a_page_is_a_parse_error(
    tmp_path: pathlib.Path, style: str
) -> None:
    marks = (BOOKMARKS[0], Bookmark("Broken", 2, level=2, style=style))  # type: ignore[arg-type]

    with pytest.raises(DocumentParseError, match="demo-guide: bookmark 2 does not resolve"):
        parse(tmp_path, PAGES, marks)


def test_font_sizes_are_rounded_so_equal_sizes_compare_equal(tmp_path: pathlib.Path) -> None:
    """pdfminer derives the size from the glyph box, so 8 pt can come out as
    8.000000000000057 depending on the baseline (this position reproduces it).
    """
    noisy = (Page(texts=(Text(54, 277, 8, "Cell"),)),)

    parsed = parse(tmp_path, noisy, (Bookmark("Cell", 1),))

    assert [w.font_size for w in parsed.words] == [8.0]

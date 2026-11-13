"""Parser stages on the real manuals: the ADR 0004 spike facts, kept true (PDF pages).

CI logs are public: every assertion compares page numbers, counts or identifiers, never
document text, so a failure cannot print the manual (see tests/corpus/conftest.py).
"""

import re
from collections.abc import Mapping

import pytest

from app.domain.ingestion import ICON, Table
from app.ingestion.lines import count_icons
from app.ingestion.outline import match_headings
from app.ingestion.pages import chapter_toc_lines, index_pages, toc_pages
from app.ingestion.tables import extract_tables
from tests.corpus.conftest import ENVY, OMEN, ParsedCorpusDocument

pytestmark = pytest.mark.corpus

Corpus = Mapping[str, ParsedCorpusDocument]

BOOKMARKS = {ENVY: (169, {1, 2, 3, 4}), OMEN: (113, {1, 2, 3, 4, 5})}
# (offset, last front-matter page): printed = PDF page - offset after the front matter,
# whose pages are unnumbered or roman-numbered (no printed page).
PRINTED_OFFSET = {ENVY: (6, 6), OMEN: (7, 7)}
TOC = {ENVY: {3, 4, 5}, OMEN: {5, 6, 7}}
INDEX = {ENVY: {134, 135}, OMEN: {89, 90, 91}}
OPENERS = {ENVY: {7, 23, 30, 42, 49, 60, 70, 79, 84, 116}, OMEN: set()}
BOTH = pytest.mark.parametrize("doc_id", [ENVY, OMEN])


def tables(doc: ParsedCorpusDocument) -> tuple[Table, ...]:
    return extract_tables(doc.footers.body, doc.parsed.rules)


@BOTH
def test_every_bookmark_resolves_to_a_page(corpus: Corpus, doc_id: str) -> None:
    outline = corpus[doc_id].parsed.outline
    page_count = corpus[doc_id].parsed.page_count
    count, levels = BOOKMARKS[doc_id]
    found = len(outline)
    unresolved = [n for n, e in enumerate(outline, start=1) if not 1 <= e.pdf_page <= page_count]

    assert found == count
    assert {entry.level for entry in outline} == levels
    assert unresolved == []


@BOTH
def test_printed_pages_come_from_footers_with_a_constant_offset(
    corpus: Corpus, doc_id: str
) -> None:
    doc = corpus[doc_id]
    offset, front_matter = PRINTED_OFFSET[doc_id]
    printed = {
        page: (doc.footers.footers[page].printed_page if page in doc.footers.footers else None)
        for page in range(1, doc.parsed.page_count + 1)
    }

    assert {p: n for p, n in printed.items() if p <= front_matter} == dict.fromkeys(
        range(1, front_matter + 1)
    )
    assert {p: n for p, n in printed.items() if p > front_matter} == {
        p: p - offset for p in range(front_matter + 1, doc.parsed.page_count + 1)
    }


def test_footer_labels_leave_the_body(corpus: Corpus) -> None:
    """ENVY footers carry the "ENWW" label on every page from the first numbered one."""
    doc = corpus[ENVY]
    labelled = {page for page, footer in doc.footers.footers.items() if "ENWW" in footer.text}
    in_body = [ln.pdf_page for ln in doc.footers.body if "ENWW" in ln.text]

    assert labelled == set(range(3, 136))
    assert in_body == []


@BOTH
def test_toc_index_and_chapter_opener_pages(corpus: Corpus, doc_id: str) -> None:
    doc = corpus[doc_id]
    outline, body = doc.parsed.outline, doc.footers.body
    openers = {body[i].pdf_page for i in chapter_toc_lines(outline, body)}
    toc, index = toc_pages(body), index_pages(outline, doc.parsed.page_count)

    assert toc == TOC[doc_id]
    assert index == INDEX[doc_id]
    assert openers == OPENERS[doc_id]


@BOTH
def test_every_bookmark_finds_its_heading(corpus: Corpus, doc_id: str) -> None:
    doc = corpus[doc_id]
    matches = match_headings(doc.parsed.outline, doc.footers.body)
    found = len(matches)
    unmatched = [e.pdf_page for e, m in zip(doc.parsed.outline, matches, strict=True) if m is None]

    assert found == BOOKMARKS[doc_id][0]
    assert unmatched == []


ERROR_CODE = re.compile(r"E\d")


def test_error_code_rows_stay_whole(corpus: Corpus) -> None:
    [codes] = [t for t in tables(corpus[ENVY]) if any(row[0] == "E0" for row in t.rows)]
    first_cells = [row[0] if ERROR_CODE.fullmatch(row[0]) else "not a code" for row in codes.rows]
    shapes = [(len(row), bool(row[-1])) for row in codes.rows]
    page = codes.pdf_page

    assert page == 105
    assert first_cells == ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E9"]
    assert shapes == [(2, True)] * 8


def test_spare_part_number_stays_next_to_its_component(corpus: Corpus) -> None:
    rows = [
        (t.pdf_page, any("Bottom cover" in cell for cell in row))
        for t in tables(corpus[OMEN])
        for row in t.rows
        if "N96208-001" in row
    ]
    pages = [page for page, _ in rows]

    assert pages
    assert rows == [(page, True) for page in pages]


# Three OMEN pages hold two tables of the same width with a figure between them; the
# band from the first table's last rule to the second's header holds the second's
# heading or intro (10 pt and up) and caption (9 pt). That band ends the first table.
TWO_TABLE_PAGES = {16, 18, 27}


def test_a_heading_between_two_same_width_tables_ends_the_first(corpus: Corpus) -> None:
    found = tables(corpus[OMEN])
    per_page = {page: sum(t.pdf_page == page for t in found) for page in TWO_TABLE_PAGES}

    assert per_page == dict.fromkeys(TWO_TABLE_PAGES, 2)


# Images inside a gap between two words of one line (body text and table cells).
# ENVY: 57 in body text, 30 in table cells. OMEN: 3, all in table cells.
ICONS = {ENVY: 87, OMEN: 3}


@BOTH
def test_lost_icons_are_marked_and_counted(corpus: Corpus, doc_id: str) -> None:
    icons = count_icons(corpus[doc_id].lines)

    assert icons == ICONS[doc_id]


def test_press_and_hold_passages_keep_a_placeholder_for_the_button(corpus: Corpus) -> None:
    lines = corpus[ENVY].lines
    marked = len([ln for ln in lines if f"press and hold {ICON}" in ln.text])
    unmarked_gaps = [ln.pdf_page for ln in lines if ln.text.endswith("press and hold")]

    assert marked == 8
    assert unmarked_gaps == []

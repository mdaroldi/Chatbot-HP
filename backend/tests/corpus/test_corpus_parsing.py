"""Parser stages on the real manuals: the ADR 0004 spike facts, kept true (PDF pages)."""

from collections.abc import Mapping

import pytest

from app.domain.ingestion import Table
from app.ingestion.outline import match_headings
from app.ingestion.pages import chapter_opener_pages, index_pages, toc_pages
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
    count, levels = BOOKMARKS[doc_id]

    assert len(outline) == count
    assert {entry.level for entry in outline} == levels
    assert all(1 <= e.pdf_page <= corpus[doc_id].parsed.page_count for e in outline)


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

    assert labelled == set(range(3, 136))
    assert [ln.pdf_page for ln in doc.footers.body if "ENWW" in ln.text] == []


@BOTH
def test_toc_index_and_chapter_opener_pages(corpus: Corpus, doc_id: str) -> None:
    doc = corpus[doc_id]
    outline = doc.parsed.outline

    assert toc_pages(doc.footers.body) == TOC[doc_id]
    assert index_pages(outline, doc.parsed.page_count) == INDEX[doc_id]
    assert chapter_opener_pages(outline, doc.footers.body) == OPENERS[doc_id]


@BOTH
def test_every_bookmark_finds_its_heading(corpus: Corpus, doc_id: str) -> None:
    doc = corpus[doc_id]
    matches = match_headings(doc.parsed.outline, doc.footers.body)

    assert len(matches) == BOOKMARKS[doc_id][0]
    assert [e.title for e, m in zip(doc.parsed.outline, matches, strict=True) if m is None] == []


def test_error_code_rows_stay_whole(corpus: Corpus) -> None:
    [codes] = [t for t in tables(corpus[ENVY]) if any(row[0] == "E0" for row in t.rows)]

    assert codes.pdf_page == 105
    assert [row[0] for row in codes.rows] == ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E9"]
    assert all(len(row) == 2 and row[1] for row in codes.rows)


def test_spare_part_number_stays_next_to_its_component(corpus: Corpus) -> None:
    rows = [row for t in tables(corpus[OMEN]) for row in t.rows if "N96208-001" in row]

    assert rows
    assert all(any("Bottom cover" in cell for cell in row) for row in rows)

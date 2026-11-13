"""Cleaning on the real manuals: drop counts by reason, and the precision of the
language rules (PDF page numbers; Phase 1 plan item 3).

CI logs are public: every assertion compares page numbers, counts or identifiers, never
document text, so a failure cannot print the manual (see tests/corpus/conftest.py).
"""

import unicodedata
from collections import Counter
from collections.abc import Mapping

import pytest

from app.domain.ingestion import DropReason
from app.ingestion.language import URL
from app.ingestion.lines import count_icons
from tests.corpus.conftest import ENVY, OMEN, ParsedCorpusDocument

pytestmark = pytest.mark.corpus

Corpus = Mapping[str, ParsedCorpusDocument]
BOTH = pytest.mark.parametrize("doc_id", [ENVY, OMEN])

LINES_DROPPED = {
    ENVY: {
        DropReason.TOC: 70,
        DropReason.INDEX: 178,
        DropReason.CHAPTER_TOC: 117,
        DropReason.SECTION_TOC: 193,
        DropReason.FOOTER: 361,
        DropReason.NON_LATIN: 104,
        DropReason.NON_ENGLISH_LATIN: 45,
        DropReason.NON_LATIN_TABLE: 78,
        DropReason.EMPTY_AFTER_CLEANING: 43,
    },
    OMEN: {
        DropReason.TOC: 117,
        DropReason.INDEX: 394,
        DropReason.CHAPTER_TOC: 0,
        DropReason.SECTION_TOC: 0,
        DropReason.FOOTER: 174,
        DropReason.NON_LATIN: 0,
        DropReason.NON_ENGLISH_LATIN: 0,
        DropReason.NON_LATIN_TABLE: 0,
        DropReason.EMPTY_AFTER_CLEANING: 0,
    },
}
PAGES_DROPPED = {ENVY: (3, 2), OMEN: (3, 3)}  # (TOC, index)
EMPTY_SECTIONS = {ENVY: 34, OMEN: 0}
# The language rules may only fire in ENVY's regulatory back matter (PDF 116-133).
NON_ENGLISH_PAGES = {ENVY: {121, 122, 123, 124, 132, 133}, OMEN: set()}
NON_LATIN_PAGES = {ENVY: {121, 123, 124, 125, 129, 130, 131, 132}, OMEN: set()}
ICONS = {ENVY: 87, OMEN: 3}
KEPT = {ENVY: 3095, OMEN: 2738}
# Section mini-TOCs, lines per PDF page: a marker and a text line per item, plus
# five lead-ins: colons on PDF 86, 120, 122, questions on PDF 95, 100.
# 19 lists with 94 items, all in ENVY.
SECTION_TOC_LINES = {
    ENVY: {
        8: 4, 10: 6, 14: 4, 32: 4, 46: 4, 53: 12, 58: 6, 74: 12, 86: 19,
        93: 4, 95: 5, 100: 5, 102: 8, 106: 10, 120: 21, 122: 13, 123: 8, 126: 48,
    },
    OMEN: {},
}  # fmt: skip
SECTION_TOC_ITEMS = {ENVY: 94, OMEN: 0}
NON_LATIN_TABLE_PAGES = {ENVY: {130}, OMEN: set()}
# Sections emptied by cleaning; 6b added 8, 10, 14, 53, 74, 86, 93 and 122 (left
# with only a mini-TOC), 130 (only the non-Latin table) and 133 (only a URL);
# 6d added 95 and 100 (left with only a question lead-in).
EMPTY_PAGES = {
    ENVY: {
        7, 8, 10, 14, 23, 30, 42, 53, 60, 70, 74, 79, 84, 86, 93, 95, 100, 109,
        121, 122, 123, 124, 130, 131, 132, 133,
    },
    OMEN: set(),
}  # fmt: skip  # lines left for sectioning


def pages(doc: ParsedCorpusDocument, reason: DropReason) -> set[int]:
    return {d.line.pdf_page for d in doc.cleaned.dropped if d.reason is reason}


@BOTH
def test_drop_counts_by_reason(corpus: Corpus, doc_id: str) -> None:
    stats = corpus[doc_id].cleaned.stats

    assert stats.lines == LINES_DROPPED[doc_id]
    assert (stats.pages[DropReason.TOC], stats.pages[DropReason.INDEX]) == PAGES_DROPPED[doc_id]
    assert stats.empty_sections == EMPTY_SECTIONS[doc_id]


def test_language_rules_fire_only_in_the_back_matter(corpus: Corpus) -> None:
    dropped = {
        doc_id: (pages(doc, DropReason.NON_ENGLISH_LATIN), pages(doc, DropReason.NON_LATIN))
        for doc_id, doc in corpus.items()
    }

    assert dropped == {d: (NON_ENGLISH_PAGES[d], NON_LATIN_PAGES[d]) for d in (ENVY, OMEN)}


def test_kept_text_has_only_latin_and_greek_letters(corpus: Corpus) -> None:
    """ENVY's back matter has CJK, Hangul, Thai and Cyrillic lines; OMEN only Greek."""
    scripts = {
        doc_id: {
            unicodedata.name(char).split()[0]
            for line in doc.cleaned.lines
            for char in line.text
            if char.isalpha()
        }
        for doc_id, doc in corpus.items()
    }

    assert scripts == {ENVY: {"LATIN"}, OMEN: {"LATIN", "GREEK"}}


@BOTH
def test_cleaning_keeps_every_icon_placeholder(corpus: Corpus, doc_id: str) -> None:
    cleaned = corpus[doc_id].cleaned
    kept, icons = len(cleaned.lines), count_icons(cleaned.lines)

    assert (kept, icons) == (KEPT[doc_id], ICONS[doc_id])


@BOTH
def test_section_mini_tocs_are_dropped_only_where_expected(corpus: Corpus, doc_id: str) -> None:
    dropped = [d.line for d in corpus[doc_id].cleaned.dropped if d.reason is DropReason.SECTION_TOC]
    per_page = Counter(line.pdf_page for line in dropped)
    items = sum(line.text == "●" for line in dropped)

    assert per_page == SECTION_TOC_LINES[doc_id]
    assert items == SECTION_TOC_ITEMS[doc_id]


@BOTH
def test_non_latin_table_residue_and_empty_sections_by_page(corpus: Corpus, doc_id: str) -> None:
    doc = corpus[doc_id]
    table_residue = pages(doc, DropReason.NON_LATIN_TABLE)
    emptied = pages(doc, DropReason.EMPTY_AFTER_CLEANING)

    assert table_residue == NON_LATIN_TABLE_PAGES[doc_id]
    assert emptied == EMPTY_PAGES[doc_id]


@BOTH
def test_no_section_is_left_with_only_urls(corpus: Corpus, doc_id: str) -> None:
    sections = corpus[doc_id].sectioned.sections
    url_only = [
        s.pages.pdf_start
        for s in sections
        if not any(char.isalnum() for b in s.blocks for char in URL.sub("", b.text))
    ]
    found = len(sections)

    assert found
    assert url_only == []

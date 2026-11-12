"""Cleaning on the real manuals: drop counts by reason, and the precision of the
language rules (PDF page numbers; Phase 1 plan item 3).
"""

import unicodedata
from collections.abc import Mapping

import pytest

from app.domain.ingestion import DropReason
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
        DropReason.FOOTER: 361,
        DropReason.NON_LATIN: 104,
        DropReason.NON_ENGLISH_LATIN: 45,
        DropReason.EMPTY_AFTER_CLEANING: 29,
    },
    OMEN: {
        DropReason.TOC: 117,
        DropReason.INDEX: 394,
        DropReason.CHAPTER_TOC: 0,
        DropReason.FOOTER: 174,
        DropReason.NON_LATIN: 0,
        DropReason.NON_ENGLISH_LATIN: 0,
        DropReason.EMPTY_AFTER_CLEANING: 0,
    },
}
PAGES_DROPPED = {ENVY: (3, 2), OMEN: (3, 3)}  # (TOC, index)
EMPTY_SECTIONS = {ENVY: 21, OMEN: 0}
# The language rules may only fire in ENVY's regulatory back matter (PDF 116-133).
NON_ENGLISH_PAGES = {ENVY: {121, 122, 123, 124, 132, 133}, OMEN: set()}
NON_LATIN_PAGES = {ENVY: {121, 123, 124, 125, 129, 130, 131, 132}, OMEN: set()}
ICONS = {ENVY: 87, OMEN: 3}
KEPT = {ENVY: 3380, OMEN: 2738}  # lines left for sectioning


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

    assert (len(cleaned.lines), count_icons(cleaned.lines)) == (KEPT[doc_id], ICONS[doc_id])

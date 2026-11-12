"""Front matter: the headed safety and notice parts printed before the first bookmark.

No bookmark covers these pages, so headings come from layout: a line at heading
size on a page after the cover. One manual sets its safety heading in the body
size, so a short line naming safety is a heading too: a deliberate keyword exception
(ADR 0005), kept to one word so that the copyright and trademark text, which has no
heading-size title, stays out of the index with the cover.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.domain.ingestion import Section, Table, TextLine
from app.ingestion.blocks import build_blocks
from app.ingestion.outline import MIN_HEADING_SIZE

FRONT_MATTER = "Front matter"
HEADING_KEYWORDS = frozenset({"safety"})
_COVER_PAGE = 1
_MAX_KEYWORD_HEADING_WORDS = 5
_SENTENCE_PUNCTUATION = tuple(".:;,!?")
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class FrontMatter:
    sections: tuple[Section, ...]
    unindexed: tuple[TextLine, ...]  # cover, legal text, headings without content


def _keyword_heading(text: str) -> bool:
    words = _WORD.findall(text.casefold())
    return (
        len(words) <= _MAX_KEYWORD_HEADING_WORDS
        and not text.rstrip().endswith(_SENTENCE_PUNCTUATION)
        and any(word in HEADING_KEYWORDS for word in words)
    )


def is_front_matter_heading(line: TextLine) -> bool:
    """A heading-size line, or a short line naming safety, on a page after the cover."""
    if line.pdf_page == _COVER_PAGE:
        return False
    return line.font_size >= MIN_HEADING_SIZE or _keyword_heading(line.text)


def front_matter_sections(
    doc_id: str,
    lines: Sequence[TextLine],
    tables: Sequence[Table],
    printed: Mapping[int, int | None],
) -> FrontMatter:
    """Lines before the first bookmarked heading to `Front matter > <heading>`
    sections; what comes before the first heading is not indexed.
    """
    headings = [i for i, line in enumerate(lines) if is_front_matter_heading(line)]
    first = headings[0] if headings else len(lines)
    sections: list[Section] = []
    unindexed = list(lines[:first])
    ends = [*headings[1:], len(lines)][: len(headings)]
    for start, end in zip(headings, ends, strict=True):
        content = lines[start + 1 : end]
        if not content:
            unindexed.append(lines[start])
            continue
        title = " ".join(lines[start].text.split())
        blocks = build_blocks(content, tables, printed)
        sections.append(Section(doc_id=doc_id, path=(FRONT_MATTER, title), blocks=blocks))
    return FrontMatter(sections=tuple(sections), unindexed=tuple(unindexed))

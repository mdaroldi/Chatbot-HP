"""Pages and page parts that hold navigation, not content: TOC, index, chapter openers.

Detection only; the cleaning stage drops what these functions find and counts it.
"""

import re
from collections import Counter
from collections.abc import Sequence

from app.domain.ingestion import OutlineEntry, TextLine
from app.ingestion.outline import clean_title, compact

# A TOC entry ends with a dot leader and a page number (arabic or roman).
_DOT_LEADER = re.compile(r"\.{3,}\s*(?:\d+|[ivxlcdm]+)$")
_MIN_TOC_ENTRIES = 3
_BULLETS = ("●", "•")
_MIN_LISTED_SECTIONS = 2
_SAME_LINE = 2.0  # points between the tops of a hanging bullet and its text


def toc_pages(lines: Sequence[TextLine]) -> frozenset[int]:
    """Pages where at least half of the lines, and at least three, are TOC entries."""
    total: Counter[int] = Counter(line.pdf_page for line in lines)
    entries: Counter[int] = Counter(
        line.pdf_page for line in lines if _DOT_LEADER.search(line.text)
    )
    return frozenset(
        page
        for page, count in entries.items()
        if count >= _MIN_TOC_ENTRIES and 2 * count >= total[page]
    )


def index_pages(outline: Sequence[OutlineEntry], page_count: int) -> frozenset[int]:
    """From a top-level "Index" bookmark to the page before the next chapter (or the end)."""
    chapters = [entry for entry in outline if entry.level == 1]
    pages: set[int] = set()
    for position, chapter in enumerate(chapters):
        if compact(clean_title(chapter.title)) == "index":
            following = chapters[position + 1 : position + 2]
            last = following[0].pdf_page - 1 if following else page_count
            pages.update(range(chapter.pdf_page, last + 1))
    return frozenset(pages)


def _sections(outline: Sequence[OutlineEntry], chapter: int) -> set[str]:
    titles: set[str] = set()
    for entry in outline[chapter + 1 :]:
        if entry.level == 1:
            break
        if entry.level == 2:
            titles.add(compact(clean_title(entry.title)))
    return titles


def _bullet_items(lines: Sequence[TextLine]) -> list[tuple[int, str]]:
    """(page, item text) for each bulleted line. With a hanging indent the bullet is
    its own segment, so its text is the next segment on the same visual line.
    """
    items: list[tuple[int, str]] = []
    for position, line in enumerate(lines):
        if not line.text.startswith(_BULLETS):
            continue
        text = line.text[1:].strip()
        following = lines[position + 1 : position + 2]
        if not text and following:
            beside = following[0]
            same_line = beside.pdf_page == line.pdf_page and (
                abs(beside.top - line.top) <= _SAME_LINE
            )
            text = beside.text if same_line else ""
        items.append((line.pdf_page, text))
    return items


def chapter_opener_pages(
    outline: Sequence[OutlineEntry], lines: Sequence[TextLine]
) -> frozenset[int]:
    """Chapter first pages that list the chapter's sections as bullets (a mini TOC)."""
    items = _bullet_items(lines)
    pages: set[int] = set()
    for position, entry in enumerate(outline):
        if entry.level != 1:
            continue
        sections = _sections(outline, position)
        listed = sum(
            1 for page, text in items if page == entry.pdf_page and compact(text) in sections
        )
        if listed >= _MIN_LISTED_SECTIONS:
            pages.add(entry.pdf_page)
    return frozenset(pages)

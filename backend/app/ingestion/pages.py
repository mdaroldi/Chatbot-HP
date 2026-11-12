"""Pages and page parts that hold navigation, not content: TOC, index, and the
mini-TOCs of chapter openers and sections.

Detection only; the cleaning stage drops what these functions find and counts it.
"""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.ingestion import OutlineEntry, TextLine
from app.ingestion.outline import clean_title, compact
from app.ingestion.spans import HeadingSpan

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


def _children(outline: Sequence[OutlineEntry], parent: int) -> set[str]:
    """Comparison keys of the titles one level below the bookmark, in its subtree."""
    level = outline[parent].level
    titles: set[str] = set()
    for entry in outline[parent + 1 :]:
        if entry.level <= level:
            break
        if entry.level == level + 1:
            titles.add(compact(clean_title(entry.title)))
    return titles


@dataclass(frozen=True)
class _BulletItem:
    page: int
    text: str
    indices: tuple[int, ...]  # the bullet's line, and its text's line when separate


def _bullet_items(lines: Sequence[TextLine]) -> list[_BulletItem]:
    """Each bulleted line with its text. With a hanging indent the bullet is its own
    segment, so its text is the next segment on the same visual line.
    """
    items: list[_BulletItem] = []
    for position, line in enumerate(lines):
        if not line.text.startswith(_BULLETS):
            continue
        text = line.text[1:].strip()
        indices: tuple[int, ...] = (position,)
        following = lines[position + 1 : position + 2]
        if not text and following:
            beside = following[0]
            if beside.pdf_page == line.pdf_page and abs(beside.top - line.top) <= _SAME_LINE:
                text, indices = beside.text, (position, position + 1)
        items.append(_BulletItem(page=line.pdf_page, text=text, indices=indices))
    return items


def _lead_in(lines: Sequence[TextLine], first_item: int) -> tuple[int, ...]:
    """The line that introduces the list, such as "This chapter covers:"."""
    position = first_item - 1
    if position >= 0:
        before = lines[position]
        if before.pdf_page == lines[first_item].pdf_page and before.text.endswith(":"):
            return (position,)
    return ()


def chapter_toc_lines(outline: Sequence[OutlineEntry], lines: Sequence[TextLine]) -> frozenset[int]:
    """Indices of chapter-opener lists: bullets on a chapter's first page that name at
    least two of its sections (a mini TOC), with the lead-in line just before them.

    Positions, not pages: an opener page can also hold the chapter's introduction,
    which is content.
    """
    items = _bullet_items(lines)
    found: set[int] = set()
    for position, entry in enumerate(outline):
        if entry.level != 1:
            continue
        sections = _children(outline, position)
        listed = [
            item for item in items if item.page == entry.pdf_page and compact(item.text) in sections
        ]
        if len(listed) >= _MIN_LISTED_SECTIONS:
            found.update(_lead_in(lines, listed[0].indices[0]))
            found.update(index for item in listed for index in item.indices)
    return frozenset(found)


def _lists(items: Sequence[_BulletItem]) -> list[list[_BulletItem]]:
    """Items on consecutive lines form one list; any other line ends it."""
    runs: list[list[_BulletItem]] = []
    for item in items:
        if runs and runs[-1][-1].indices[-1] + 1 == item.indices[0]:
            runs[-1].append(item)
        else:
            runs.append([item])
    return runs


def section_toc_lines(
    outline: Sequence[OutlineEntry], lines: Sequence[TextLine], spans: Sequence[HeadingSpan]
) -> frozenset[int]:
    """Indices of section mini-TOCs: a list of at least two bullets in which every
    item is the title of one of the section's own subsections, with its lead-in.

    Every item must match: one item that is not a child title (a URL, a title with
    more words, a title wrapped onto a second line) keeps the whole list.
    """
    bullets = _bullet_items(lines)
    found: set[int] = set()
    for span in spans:
        titles = _children(outline, span.entry)
        items = [i for i in bullets if span.body <= i.indices[0] and i.indices[-1] < span.end]
        for run in _lists(items):
            if len(run) >= _MIN_LISTED_SECTIONS and all(
                compact(item.text) in titles for item in run
            ):
                lead_in = _lead_in(lines, run[0].indices[0])
                found.update(i for i in lead_in if i >= span.body)
                found.update(index for item in run for index in item.indices)
    return frozenset(found)

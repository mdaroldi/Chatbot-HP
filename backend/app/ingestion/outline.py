"""Bookmarks to section paths, and each bookmark to the heading line it points at."""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.ingestion import OutlineEntry, TextLine

# Headings are set at 12 pt or more; body text is 10 pt.
MIN_HEADING_SIZE = 12.0

_TITLE_NUMBER = re.compile(r"^\d+\.\s+")
# Chapter numbers and appendix letters printed before a heading ("1 Setup", "A Setup").
_HEADING_LABEL = re.compile(r"^(?:\d+|[A-Z])\.?\s+")


@dataclass(frozen=True)
class HeadingMatch:
    """Indices into the line sequence; a wrapped heading spans several lines."""

    first_line: int
    last_line: int


def clean_title(title: str) -> str:
    """Bookmark title as shown to users: NFKC spaces, no chapter-number prefix."""
    text = " ".join(unicodedata.normalize("NFKC", title).split())
    return _TITLE_NUMBER.sub("", text)


def compact(text: str) -> str:
    """Comparison key that ignores case and all whitespace, including the stray
    spaces text extraction inserts at broken ligatures ("Auto-O ff").
    """
    return "".join(unicodedata.normalize("NFKC", text).split()).casefold()


def section_paths(outline: Sequence[OutlineEntry]) -> tuple[tuple[str, ...], ...]:
    paths: list[tuple[str, ...]] = []
    stack: list[tuple[int, str]] = []
    for entry in outline:
        while stack and stack[-1][0] >= entry.level:
            stack.pop()
        stack.append((entry.level, clean_title(entry.title)))
        paths.append(tuple(title for _, title in stack))
    return tuple(paths)


def _match_at(lines: Sequence[TextLine], start: int, target: str) -> HeadingMatch | None:
    first = lines[start]
    for start_key in {compact(first.text), compact(_HEADING_LABEL.sub("", first.text))}:
        key, end = start_key, start
        while key != target and target.startswith(key) and end + 1 < len(lines):
            following = lines[end + 1]
            if following.font_size != first.font_size:
                break
            end += 1
            key += compact(following.text)
        if key == target:
            return HeadingMatch(first_line=start, last_line=end)
    return None


def _search(
    lines: Sequence[TextLine],
    cursor: int,
    target: str,
    pages: tuple[int, int],
    min_size: float,
) -> HeadingMatch | None:
    first_page, last_page = pages
    for index in range(cursor, len(lines)):
        line = lines[index]
        if line.pdf_page > last_page:
            break
        if line.pdf_page >= first_page and line.font_size >= min_size:
            match = _match_at(lines, index, target)
            if match:
                return match
    return None


def match_headings(
    outline: Sequence[OutlineEntry], lines: Sequence[TextLine]
) -> tuple[HeadingMatch | None, ...]:
    """Find each bookmark's heading in reading-order lines.

    The heading is the first line at or after the bookmark's page (or on the next
    page) that matches its title at heading size. A few deep bookmarks point at
    headings set in the body size; those are accepted only on the bookmark's own
    page. Matches move forward, so repeated titles map to successive headings.
    """
    matches: list[HeadingMatch | None] = []
    cursor = 0
    for entry in outline:
        target = compact(clean_title(entry.title))
        page = entry.pdf_page
        match = _search(lines, cursor, target, (page, page + 1), MIN_HEADING_SIZE) or _search(
            lines, cursor, target, (page, page), 0.0
        )
        matches.append(match)
        if match:
            cursor = match.last_line + 1
    return tuple(matches)

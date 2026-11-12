"""Heading spans: where each bookmark's heading and content sit in the line sequence.

Shared by cleaning (to find sections left empty) and section building.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.ingestion import OutlineEntry, TextLine
from app.ingestion.lines import LINE_TOLERANCE
from app.ingestion.outline import HeadingMatch, match_headings


@dataclass(frozen=True)
class HeadingSpan:
    """Line indices: the heading is [start, body), its content [body, end)."""

    entry: int  # index into the outline
    start: int
    body: int
    end: int


def _start(lines: Sequence[TextLine], match: HeadingMatch, floor: int) -> int:
    """A chapter number or appendix letter set before the title on the same visual
    line belongs to the heading, not to the section above.
    """
    title, start = lines[match.first_line], match.first_line
    while start > floor and (
        lines[start - 1].pdf_page == title.pdf_page
        and abs(lines[start - 1].top - title.top) <= LINE_TOLERANCE
    ):
        start -= 1
    return start


def heading_spans(
    outline: Sequence[OutlineEntry], lines: Sequence[TextLine]
) -> tuple[HeadingSpan, ...]:
    """One span per bookmark whose heading was found, in reading order; each runs to
    the next heading, the last one to the end. Lines before the first heading belong
    to no span.
    """
    found = [(e, m) for e, m in enumerate(match_headings(outline, lines)) if m]
    starts: list[int] = []
    for number, (_, match) in enumerate(found):
        floor = found[number - 1][1].last_line + 1 if number else 0
        starts.append(_start(lines, match, floor))
    ends = [*starts[1:], len(lines)][: len(starts)]
    return tuple(
        HeadingSpan(entry=e, start=start, body=match.last_line + 1, end=end)
        for (e, match), start, end in zip(found, starts, ends, strict=True)
    )

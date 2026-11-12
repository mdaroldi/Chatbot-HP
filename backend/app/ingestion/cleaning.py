"""Cleaning: drop what is not English content and count every drop by reason.

Each line is dropped at most once, under the first rule that matches, in the order
of `DropReason`: footer, TOC page, index page, chapter-opener list, non-Latin
script, non-English Latin text, and finally sections that cleaning left empty.
Heading matching runs before any drop, so a section keeps its bounds even when
the lines around it go.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.domain.ingestion import ICON, DropReason, OutlineEntry, TextLine
from app.ingestion.footer import PageFooter, split_footers
from app.ingestion.language import MIN_SCORE, is_non_latin, score_line
from app.ingestion.lines import LINE_TOLERANCE
from app.ingestion.outline import HeadingMatch, match_headings
from app.ingestion.pages import chapter_toc_lines, index_pages, toc_pages

# Lines of one paragraph are about 1.2 ems apart (top to top); paragraphs about 2.
BLOCK_LEADING_EM = 1.5

Reasons = list[DropReason | None]


@dataclass(frozen=True)
class DroppedLine:
    line: TextLine
    reason: DropReason


@dataclass(frozen=True)
class CleaningStats:
    """Per document: lines dropped by every reason (zeros included), whole pages
    dropped as TOC or index, and sections dropped as empty.
    """

    lines: Mapping[DropReason, int]
    pages: Mapping[DropReason, int]
    empty_sections: int


@dataclass(frozen=True)
class CleanedDocument:
    lines: tuple[TextLine, ...]
    footers: Mapping[int, PageFooter]
    dropped: tuple[DroppedLine, ...]
    stats: CleaningStats


def _footer_lines(lines: Sequence[TextLine], body: Sequence[TextLine]) -> list[TextLine]:
    """The lines footer splitting removed; the body is an ordered subsequence."""
    removed: list[TextLine] = []
    kept = 0
    for line in lines:
        if kept < len(body) and body[kept] == line:
            kept += 1
        else:
            removed.append(line)
    return removed


def _line_reasons(
    body: Sequence[TextLine], outline: Sequence[OutlineEntry], page_count: int
) -> tuple[Reasons, frozenset[int], frozenset[int]]:
    toc, index = toc_pages(body), index_pages(outline, page_count)
    opener_lists = chapter_toc_lines(outline, body)
    reasons: Reasons = []
    for position, line in enumerate(body):
        if line.pdf_page in toc:
            reasons.append(DropReason.TOC)
        elif line.pdf_page in index:
            reasons.append(DropReason.INDEX)
        elif position in opener_lists:
            reasons.append(DropReason.CHAPTER_TOC)
        elif is_non_latin(line.text):
            reasons.append(DropReason.NON_LATIN)
        elif score_line(line.text).is_non_english:
            reasons.append(DropReason.NON_ENGLISH_LATIN)
        else:
            reasons.append(None)
    return reasons, toc, index


def _line_above(body: Sequence[TextLine], position: int) -> int | None:
    """The nearest earlier line on the page in the same column. Reading order goes
    left to right across columns, so the previous line may sit beside this one.
    """
    line = body[position]
    for earlier in range(position - 1, -1, -1):
        other = body[earlier]
        if other.pdf_page != line.pdf_page:
            return None
        if other.x0 < line.x1 and other.x1 > line.x0:
            return earlier
    return None


def _drop_paragraph_tails(body: Sequence[TextLine], reasons: Reasons) -> None:
    """A line too short to judge goes with the dropped paragraph it ends, unless it
    reads as English. Tails chain: a tail can continue a dropped tail.
    """
    for position, line in enumerate(body):
        above = _line_above(body, position) if reasons[position] is None else None
        if above is None or reasons[above] is not DropReason.NON_ENGLISH_LATIN:
            continue
        previous = body[above]
        same_block = (
            previous.font_size == line.font_size
            and line.top - previous.top <= BLOCK_LEADING_EM * line.font_size
        )
        if same_block and score_line(line.text).english < MIN_SCORE:
            reasons[position] = DropReason.NON_ENGLISH_LATIN


def _heading_starts(body: Sequence[TextLine], matches: Sequence[HeadingMatch]) -> list[int]:
    """A heading starts with its visual line: a chapter number or appendix letter set
    before the title belongs to the heading, not to the section above.
    """
    starts: list[int] = []
    for number, match in enumerate(matches):
        floor = matches[number - 1].last_line + 1 if number else 0
        title, start = body[match.first_line], match.first_line
        while start > floor and (
            body[start - 1].pdf_page == title.pdf_page
            and abs(body[start - 1].top - title.top) <= LINE_TOLERANCE
        ):
            start -= 1
        starts.append(start)
    return starts


def _has_text(line: TextLine) -> bool:
    return any(char.isalnum() for char in line.text.replace(ICON, ""))


def _drop_empty_sections(
    body: Sequence[TextLine], reasons: Reasons, outline: Sequence[OutlineEntry]
) -> int:
    """Drop each section whose kept content has no text, with its heading. Sections
    whose heading another rule removed were already counted under that rule.
    """
    matches = [match for match in match_headings(outline, body) if match]
    starts = _heading_starts(body, matches)
    ends = [*starts[1:], len(body)]
    empty = 0
    for match, start, end in zip(matches, starts, ends, strict=True):
        heading = range(start, match.last_line + 1)
        content = range(match.last_line + 1, end)
        if any(reasons[i] for i in heading):
            continue
        if not any(reasons[i] is None and _has_text(body[i]) for i in content):
            empty += 1
            for i in (*heading, *content):
                reasons[i] = reasons[i] or DropReason.EMPTY_AFTER_CLEANING
    return empty


def clean(
    lines: Sequence[TextLine], outline: Sequence[OutlineEntry], page_heights: Sequence[float]
) -> CleanedDocument:
    """Lines in reading order (after line grouping) to the lines that later stages
    turn into sections. Every drop is kept with its reason, for statistics and review.
    """
    split = split_footers(lines, page_heights)
    body = split.body
    reasons, toc, index = _line_reasons(body, outline, len(page_heights))
    _drop_paragraph_tails(body, reasons)
    empty = _drop_empty_sections(body, reasons, outline)
    dropped = (
        *(DroppedLine(line, DropReason.FOOTER) for line in _footer_lines(lines, body)),
        *(DroppedLine(line, reason) for line, reason in zip(body, reasons, strict=True) if reason),
    )
    counts = Counter(d.reason for d in dropped)
    return CleanedDocument(
        lines=tuple(line for line, reason in zip(body, reasons, strict=True) if reason is None),
        footers=split.footers,
        dropped=dropped,
        stats=CleaningStats(
            lines={reason: counts[reason] for reason in DropReason},
            pages={DropReason.TOC: len(toc), DropReason.INDEX: len(index)},
            empty_sections=empty,
        ),
    )

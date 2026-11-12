"""Cleaning: drop what is not English content and count every drop by reason.

Each line is dropped at most once, under the first rule that matches, in the order
of `DropReason`: footer, TOC page, index page, chapter-opener list, section
mini-TOC, non-Latin script, non-English Latin text, then the residue of tables that
lost their non-Latin text, and finally sections that cleaning left empty.
Heading spans are found before any drop, so a section keeps its bounds even when
the lines around it go; the result carries them re-indexed onto the kept lines.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.domain.ingestion import ICON, DropReason, HorizontalRule, OutlineEntry, Table, TextLine
from app.ingestion.footer import PageFooter, split_footers
from app.ingestion.language import MIN_SCORE, URL, has_latin_word, is_non_latin, score_line
from app.ingestion.pages import chapter_toc_lines, index_pages, section_toc_lines, toc_pages
from app.ingestion.spans import HeadingSpan, heading_spans
from app.ingestion.tables import extract_tables, table_at

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
    spans: tuple[HeadingSpan, ...]  # over `lines`


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
    body: Sequence[TextLine],
    outline: Sequence[OutlineEntry],
    spans: Sequence[HeadingSpan],
    page_count: int,
) -> tuple[Reasons, frozenset[int], frozenset[int]]:
    toc, index = toc_pages(body), index_pages(outline, page_count)
    opener_lists = chapter_toc_lines(outline, body)
    section_lists = section_toc_lines(outline, body, spans)
    reasons: Reasons = []
    for position, line in enumerate(body):
        if line.pdf_page in toc:
            reasons.append(DropReason.TOC)
        elif line.pdf_page in index:
            reasons.append(DropReason.INDEX)
        elif position in opener_lists:
            reasons.append(DropReason.CHAPTER_TOC)
        elif position in section_lists:
            reasons.append(DropReason.SECTION_TOC)
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


def _drop_table_residue(
    body: Sequence[TextLine], reasons: Reasons, rules: Sequence[HorizontalRule]
) -> None:
    """A table that lost lines as non-Latin and keeps no Latin word holds only what
    the dropped text labelled (element symbols, marks): drop the rest of it too.
    Tables that lost no line this way are never touched, whatever their content.
    """
    members: defaultdict[Table, list[int]] = defaultdict(list)
    tables = extract_tables(body, rules)
    for position, line in enumerate(body):
        if (table := table_at(line, tables)) is not None:
            members[table].append(position)
    for positions in members.values():
        kept = [i for i in positions if reasons[i] is None]
        if any(reasons[i] is DropReason.NON_LATIN for i in positions) and not any(
            has_latin_word(body[i].text) for i in kept
        ):
            for i in kept:
                reasons[i] = DropReason.NON_LATIN_TABLE


def _has_text(line: TextLine) -> bool:
    """Icons and URLs are not text: a section holding only them is empty."""
    return any(char.isalnum() for char in URL.sub("", line.text.replace(ICON, "")))


def _drop_empty_sections(
    body: Sequence[TextLine], reasons: Reasons, spans: Sequence[HeadingSpan]
) -> int:
    """Drop each section whose kept content has no text, with its heading. Sections
    whose heading another rule removed were already counted under that rule.
    """
    empty = 0
    for span in spans:
        heading, content = range(span.start, span.body), range(span.body, span.end)
        if any(reasons[i] for i in heading):
            continue
        if not any(reasons[i] is None and _has_text(body[i]) for i in content):
            empty += 1
            for i in (*heading, *content):
                reasons[i] = reasons[i] or DropReason.EMPTY_AFTER_CLEANING
    return empty


def _kept_spans(spans: Sequence[HeadingSpan], reasons: Reasons) -> tuple[HeadingSpan, ...]:
    """Spans re-indexed onto the kept lines, for spans whose heading survived. Kept
    content of a span whose heading went (rare) joins the span before it.
    """
    position, kept_index = 0, []
    for reason in reasons:
        kept_index.append(position)
        position += reason is None
    survivors = [s for s in spans if not any(reasons[i] for i in range(s.start, s.body))]
    starts = [kept_index[s.start] for s in survivors]
    ends = [*starts[1:], position][: len(starts)]
    return tuple(
        HeadingSpan(entry=s.entry, start=start, body=start + (s.body - s.start), end=end)
        for s, start, end in zip(survivors, starts, ends, strict=True)
    )


def clean(
    lines: Sequence[TextLine],
    outline: Sequence[OutlineEntry],
    page_heights: Sequence[float],
    rules: Sequence[HorizontalRule],
) -> CleanedDocument:
    """Lines in reading order (after line grouping) to the lines that later stages
    turn into sections. Every drop is kept with its reason, for statistics and review.
    """
    split = split_footers(lines, page_heights)
    body = split.body
    spans = heading_spans(outline, body)
    reasons, toc, index = _line_reasons(body, outline, spans, len(page_heights))
    _drop_paragraph_tails(body, reasons)
    _drop_table_residue(body, reasons, rules)
    empty = _drop_empty_sections(body, reasons, spans)
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
        spans=_kept_spans(spans, reasons),
    )

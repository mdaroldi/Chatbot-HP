"""Blocks: a section's kept lines grouped into paragraphs, steps, bullets, notes and
table rows.

Steps and bullets use a hanging indent: the marker ("1.", "a.", "●") is its own line
segment and the text sits beside it. Notes start with their label ("CAUTION:"), as
one segment or as a label segment with the text beside it. A ruled table replaces
all of its lines with one header block and one block per row.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.domain.ingestion import Block, BlockKind, NoteLabel, PageRange, Table, TextLine
from app.ingestion.cleaning import BLOCK_LEADING_EM
from app.ingestion.lines import LINE_TOLERANCE

_STEP_MARKER = re.compile(r"(?:\d+|[a-z])\.")
_BULLETS = frozenset({"●", "•", "\u2013"})  # \u2013: en dash, second-level bullets
_LABELS = {
    "WARNING!": NoteLabel.WARNING,
    "CAUTION:": NoteLabel.CAUTION,
    "IMPORTANT:": NoteLabel.IMPORTANT,
    "NOTE:": NoteLabel.NOTE,
    "TIP:": NoteLabel.TIP,
}
_TABLE_TOLERANCE = 2.0  # points a cell line may stick out of its table's box


@dataclass
class _Open:
    kind: BlockKind
    note: NoteLabel | None
    lines: list[TextLine] = field(default_factory=list)


def _start(line: TextLine) -> tuple[BlockKind, NoteLabel | None] | None:
    if _STEP_MARKER.fullmatch(line.text):
        return BlockKind.STEP, None
    if line.text in _BULLETS:
        return BlockKind.BULLET, None
    label = line.text.split(" ", 1)[0]
    if label in _LABELS:
        return BlockKind.NOTE, _LABELS[label]
    return None


def _continues(block: _Open, line: TextLine) -> bool:
    last = block.lines[-1]
    if last.pdf_page != line.pdf_page:
        return False
    beside = abs(line.top - last.top) <= LINE_TOLERANCE and line.x0 >= last.x1
    if beside:
        # Only a lone marker or label takes the text beside it; other segments on
        # the same visual line belong to another column.
        return len(block.lines) == 1 and _start(last) is not None
    below = 0 < line.top - last.top <= BLOCK_LEADING_EM * line.font_size
    overlaps = line.x0 < last.x1 and line.x1 > last.x0
    return below and overlaps and line.font_size == last.font_size


def _pages(pdf_pages: Sequence[int], printed: Mapping[int, int | None]) -> PageRange:
    numbers = [n for n in (printed.get(p) for p in pdf_pages) if n is not None]
    return PageRange(
        pdf_start=min(pdf_pages),
        pdf_end=max(pdf_pages),
        printed_start=min(numbers) if numbers else None,
        printed_end=max(numbers) if numbers else None,
    )


def _close(block: _Open, printed: Mapping[int, int | None]) -> Block:
    return Block(
        kind=block.kind,
        text=" ".join(line.text for line in block.lines),
        pages=_pages([line.pdf_page for line in block.lines], printed),
        note=block.note,
        indent=block.lines[0].x0,
    )


def _table_of(line: TextLine, tables: Sequence[Table]) -> Table | None:
    for table in tables:
        if (
            table.pdf_page == line.pdf_page
            and table.top - _TABLE_TOLERANCE <= line.top
            and line.bottom <= table.bottom + _TABLE_TOLERANCE
            and line.x0 < table.x1
            and line.x1 > table.x0
        ):
            return table
    return None


def _table_blocks(table: Table, printed: Mapping[int, int | None]) -> list[Block]:
    pages = _pages([table.pdf_page], printed)
    rows = [(BlockKind.TABLE_HEADER, table.header), *((BlockKind.TABLE_ROW, r) for r in table.rows)]
    return [
        Block(kind=kind, text=" | ".join(cells).strip(), pages=pages, indent=table.x0)
        for kind, cells in rows
        if any(cell.strip() for cell in cells)
    ]


def build_blocks(
    lines: Sequence[TextLine], tables: Sequence[Table], printed: Mapping[int, int | None]
) -> tuple[Block, ...]:
    """Lines in reading order to blocks in reading order; `printed` maps PDF pages to
    printed page numbers (absent for unnumbered and roman-numbered pages).
    """
    result: list[Block] = []
    emitted: set[Table] = set()
    current: _Open | None = None
    for line in lines:
        table = _table_of(line, tables)
        start = _start(line)
        if current and (table or start or not _continues(current, line)):
            result.append(_close(current, printed))
            current = None
        if table:
            if table not in emitted:
                emitted.add(table)
                result.extend(_table_blocks(table, printed))
            continue
        if current is None:
            kind, note = start or (BlockKind.PARAGRAPH, None)
            current = _Open(kind=kind, note=note)
        current.lines.append(line)
    if current:
        result.append(_close(current, printed))
    return tuple(result)

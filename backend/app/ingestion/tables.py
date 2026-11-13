"""Row-band table extraction: horizontal rules split rows, header gaps give columns.

pdfplumber's table finders fail on both manuals (ADR 0004): one style has only
horizontal rules, the other a single detected column. Here a table is a run of at
least three rules with the same horizontal extent where every band between two
rules is a row; a row can therefore never be separated from its cells.

A band is a row only if all of its text is set at table size (both manuals set cells
in 8 pt, captions in 9 pt and body text in 10 pt). Empty bands and bands with any body
text, heading or boxed note end a table, which also separates two tables of the same
width on one page.
"""

from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Sequence
from itertools import pairwise

from app.domain.ingestion import HorizontalRule, Table, TextLine

_RULE_Y_TOLERANCE = 0.6  # segments this close vertically are one rule
_RULE_JOIN_GAP = 1.5  # ... and join when their ends are this close
_EXTENT_TOLERANCE = 2.0  # rules of one table start and end within this distance
_MIN_TABLE_RULES = 3
_COLUMN_TOLERANCE = 2.0  # a cell may start this far left of its column
TABLE_MAX_FONT_SIZE = 9.0  # cells are set below the 10 pt body size
_TABLE_TOLERANCE = 2.0  # points a cell line may stick out of its table's box


def merge_rules(rules: Sequence[HorizontalRule]) -> tuple[HorizontalRule, ...]:
    """Join collinear segments: PDFs often draw one rule per cell."""
    clusters: list[list[HorizontalRule]] = []
    for rule in sorted(rules, key=lambda r: (r.pdf_page, r.y)):
        first = clusters[-1][0] if clusters else None
        if first and first.pdf_page == rule.pdf_page and rule.y - first.y <= _RULE_Y_TOLERANCE:
            clusters[-1].append(rule)
        else:
            clusters.append([rule])
    merged: list[HorizontalRule] = []
    for cluster in clusters:
        spans: list[list[float]] = []
        for rule in sorted(cluster, key=lambda r: r.x0):
            if spans and rule.x0 <= spans[-1][1] + _RULE_JOIN_GAP:
                spans[-1][1] = max(spans[-1][1], rule.x1)
            else:
                spans.append([rule.x0, rule.x1])
        page, y = cluster[0].pdf_page, cluster[0].y
        merged.extend(HorizontalRule(pdf_page=page, y=y, x0=a, x1=b) for a, b in spans)
    return tuple(merged)


def _grids(rules: Sequence[HorizontalRule]) -> list[list[HorizontalRule]]:
    """Rules of one page grouped by extent; short ones (link underlines) stay alone."""
    groups: list[list[HorizontalRule]] = []
    for rule in sorted(rules, key=lambda r: (r.pdf_page, r.y)):
        for group in groups:
            head = group[0]
            if (
                head.pdf_page == rule.pdf_page
                and abs(head.x0 - rule.x0) <= _EXTENT_TOLERANCE
                and abs(head.x1 - rule.x1) <= _EXTENT_TOLERANCE
            ):
                group.append(rule)
                break
        else:
            groups.append([rule])
    return [group for group in groups if len(group) >= _MIN_TABLE_RULES]


def _middle(line: TextLine) -> float:
    return (line.top + line.bottom) / 2


def _dominant_size(lines: Sequence[TextLine]) -> float:
    sizes: Counter[float] = Counter()
    for line in lines:
        sizes[line.font_size] += len(line.text)
    return sizes.most_common(1)[0][0]


def _is_row(band: Sequence[TextLine]) -> bool:
    # Every line, not the dominant size: a heading and intro over the next table's
    # caption are mostly 9 pt by length, and would join two tables across a section.
    return bool(band) and all(line.font_size <= TABLE_MAX_FONT_SIZE for line in band)


def _in_band(lines: Sequence[TextLine], upper: float, lower: float) -> list[TextLine]:
    return [line for line in lines if upper <= _middle(line) < lower]


def _runs(grid: Sequence[HorizontalRule], inside: Sequence[TextLine]) -> list[list[float]]:
    """Split a grid wherever a band is not a table row."""
    runs: list[list[float]] = [[grid[0].y]]
    for upper, lower in pairwise(rule.y for rule in grid):
        if _is_row(_in_band(inside, upper, lower)):
            runs[-1].append(lower)
        else:
            runs.append([lower])
    return [run for run in runs if len(run) >= _MIN_TABLE_RULES]


def _columns(header: Sequence[TextLine]) -> list[float]:
    starts: list[float] = []
    for x0 in sorted(line.x0 for line in header):
        if not starts or x0 - starts[-1] > _COLUMN_TOLERANCE:
            starts.append(x0)
    return starts


def _cells(band: Sequence[TextLine], columns: Sequence[float]) -> tuple[str, ...]:
    parts: list[list[str]] = [[] for _ in columns]
    for line in sorted(band, key=lambda ln: (ln.top, ln.x0)):
        column = max(bisect_right(columns, line.x0 + _COLUMN_TOLERANCE) - 1, 0)
        parts[column].append(line.text)
    return tuple(" ".join(part) for part in parts)


def _table(
    page: int,
    ys: Sequence[float],
    extent: tuple[float, float],
    inside: Sequence[TextLine],
    page_rules: Sequence[HorizontalRule],
) -> Table:
    x0, x1 = extent
    bands = [_in_band(inside, upper, lower) for upper, lower in pairwise(ys)]
    # Without a top border the header sits just above the first rule, in the
    # table's own font size (captions and intros use other sizes), with no other
    # rule in between (that would make it the last row of a table above).
    body_size = _dominant_size([line for band in bands for line in band])
    floor = max(
        [ys[0] - (ys[1] - ys[0])]
        + [r.y for r in page_rules if r.y < ys[0] and r.x0 < x1 and r.x1 > x0],
    )
    above = [
        line
        for line in inside
        if _middle(line) < ys[0] and line.bottom > floor and line.font_size == body_size
    ]
    header, *rows = [above, *bands] if above else bands
    columns = _columns(header)
    return Table(
        pdf_page=page,
        top=min([ys[0], *(line.top for line in above)]),
        bottom=ys[-1],
        x0=x0,
        x1=x1,
        header=_cells(header, columns),
        rows=tuple(_cells(row, columns) for row in rows),
    )


def table_at(line: TextLine, tables: Sequence[Table]) -> Table | None:
    """The table whose box holds the line, if any."""
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


def extract_tables(lines: Sequence[TextLine], rules: Sequence[HorizontalRule]) -> tuple[Table, ...]:
    merged = merge_rules(rules)
    lines_by_page: defaultdict[int, list[TextLine]] = defaultdict(list)
    for line in lines:
        lines_by_page[line.pdf_page].append(line)
    rules_by_page: defaultdict[int, list[HorizontalRule]] = defaultdict(list)
    for rule in merged:
        rules_by_page[rule.pdf_page].append(rule)
    tables: list[Table] = []
    for grid in _grids(merged):
        page = grid[0].pdf_page
        extent = (min(r.x0 for r in grid), max(r.x1 for r in grid))
        inside = [ln for ln in lines_by_page[page] if ln.x0 < extent[1] and ln.x1 > extent[0]]
        tables.extend(
            _table(page, run, extent, inside, rules_by_page[page]) for run in _runs(grid, inside)
        )
    return tuple(sorted(tables, key=lambda t: (t.pdf_page, t.top)))

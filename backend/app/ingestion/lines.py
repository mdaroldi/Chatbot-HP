"""Group parser words into line segments: one run of text on one baseline."""

from collections import Counter
from collections.abc import Sequence

from app.domain.ingestion import TextLine, Word

# Words whose tops differ by at most this much (points) share a visual line.
LINE_TOLERANCE = 2.0
# A horizontal gap wider than this many ems ends a segment: table columns and the
# holes left by inline icons both produce such gaps; word spaces never do.
SEGMENT_GAP_EM = 1.0


def _visual_lines(words: Sequence[Word]) -> list[list[Word]]:
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda w: (w.pdf_page, w.top)):
        first = lines[-1][0] if lines else None
        if first and first.pdf_page == word.pdf_page and word.top - first.top <= LINE_TOLERANCE:
            lines[-1].append(word)
        else:
            lines.append([word])
    return lines


def _segments(line: list[Word]) -> list[list[Word]]:
    segments: list[list[Word]] = []
    for word in sorted(line, key=lambda w: w.x0):
        if segments:
            previous = segments[-1][-1]
            gap = word.x0 - previous.x1
            if gap <= SEGMENT_GAP_EM * max(previous.font_size, word.font_size):
                segments[-1].append(word)
                continue
        segments.append([word])
    return segments


def _to_line(words: list[Word]) -> TextLine:
    sizes: Counter[float] = Counter()
    for word in words:
        sizes[word.font_size] += len(word.text)
    return TextLine(
        text=" ".join(w.text for w in words),
        pdf_page=words[0].pdf_page,
        font_size=sizes.most_common(1)[0][0],
        x0=min(w.x0 for w in words),
        top=min(w.top for w in words),
        x1=max(w.x1 for w in words),
        bottom=max(w.bottom for w in words),
    )


def group_lines(words: Sequence[Word]) -> tuple[TextLine, ...]:
    """Segments, not whole visual lines: later stages place each segment in a table
    column or rejoin it with its neighbours, so a wide gap must stay visible. Output is
    in reading order: page, visual line, then left to right.
    """
    return tuple(_to_line(seg) for line in _visual_lines(words) for seg in _segments(line))

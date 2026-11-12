"""Group parser words into line segments: one run of text on one baseline.

Inline images that sit in a gap between two words of a line (button glyphs lost in
text extraction) become an `[icon]` word, so the gap no longer splits the sentence.
"""

from collections import Counter
from collections.abc import Sequence

from app.domain.ingestion import ICON, ImageBox, TextLine, Word

# Words whose tops differ by at most this much (points) share a visual line.
LINE_TOLERANCE = 2.0
# A horizontal gap wider than this many ems ends a segment: table columns and the
# holes left by inline icons both produce such gaps; word spaces never do.
SEGMENT_GAP_EM = 1.0
# An inline glyph is at most this many ems tall; taller images are figures.
ICON_MAX_EM = 3.0
_ICON_NEIGHBOUR_SLACK = 1.0  # points a neighbouring word may overlap the image


def _visual_lines(words: Sequence[Word]) -> list[list[Word]]:
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda w: (w.pdf_page, w.top)):
        first = lines[-1][0] if lines else None
        if first and first.pdf_page == word.pdf_page and word.top - first.top <= LINE_TOLERANCE:
            lines[-1].append(word)
        else:
            lines.append([word])
    return lines


def _icon_word(line: Sequence[Word], image: ImageBox) -> Word | None:
    """An `[icon]` word if the image sits inside a gap of this line, else None.

    Images at the start or end of a line (margin symbols beside notes) and images
    that are not part of a line (logos, figures) get no placeholder.
    """
    if image.pdf_page != line[0].pdf_page:
        return None
    middle = (image.top + image.bottom) / 2
    top, bottom = min(w.top for w in line), max(w.bottom for w in line)
    if not top - LINE_TOLERANCE <= middle <= bottom + LINE_TOLERANCE:
        return None
    left = [w for w in line if w.x1 <= image.x0 + _ICON_NEIGHBOUR_SLACK]
    right = [w for w in line if w.x0 >= image.x1 - _ICON_NEIGHBOUR_SLACK]
    if not left or not right:
        return None
    neighbour = max(left, key=lambda w: w.x1)
    if image.bottom - image.top > ICON_MAX_EM * neighbour.font_size:
        return None
    # The line keeps its own box: the placeholder takes the neighbour's height.
    return Word(
        text=ICON,
        pdf_page=neighbour.pdf_page,
        font_size=neighbour.font_size,
        x0=image.x0,
        top=neighbour.top,
        x1=image.x1,
        bottom=neighbour.bottom,
    )


def _add_icons(lines: list[list[Word]], images: Sequence[ImageBox]) -> None:
    for image in images:
        for line in lines:
            icon = _icon_word(line, image)
            if icon:
                line.append(icon)
                break


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


def group_lines(words: Sequence[Word], images: Sequence[ImageBox] = ()) -> tuple[TextLine, ...]:
    """Segments, not whole visual lines: later stages place each segment in a table
    column or rejoin it with its neighbours, so a wide gap must stay visible. Output is
    in reading order: page, visual line, then left to right.
    """
    lines = _visual_lines(words)
    _add_icons(lines, images)
    return tuple(_to_line(seg) for line in lines for seg in _segments(line))


def count_icons(lines: Sequence[TextLine]) -> int:
    """Lost-icon count for the build statistics."""
    return sum(line.text.count(ICON) for line in lines)

"""Footer detection: drop the footer zone and read the printed page number from it."""

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.domain.ingestion import TextLine

# The footer zone is the bottom 7 % of the page (about 55 pt on a letter page).
FOOTER_ZONE_FRACTION = 0.07

_ARABIC = re.compile(r"\d+")
_ROMAN = re.compile(r"[ivxlcdm]+")


@dataclass(frozen=True)
class PageFooter:
    pdf_page: int
    printed_page: int | None
    text: str


@dataclass(frozen=True)
class FooterSplit:
    body: tuple[TextLine, ...]
    footers: Mapping[int, PageFooter]


def _page_number(text: str) -> tuple[bool, int | None]:
    """(is a footer, printed page); roman front-matter numbers have no printed page."""
    tokens = text.split()
    ends = tokens[:1] + tokens[-1:]
    for token in ends:
        if _ARABIC.fullmatch(token):
            return True, int(token)
    return any(_ROMAN.fullmatch(token) for token in ends), None


def split_footers(lines: Sequence[TextLine], page_heights: Sequence[float]) -> FooterSplit:
    """A footer is the text in the bottom zone that starts or ends with a page number.

    Requiring the number keeps body text that runs into the zone, such as a cover
    paragraph, and the number itself gives the printed page, so no fixed PDF-to-print
    offset is assumed.
    """
    zones: dict[int, list[TextLine]] = defaultdict(list)
    for line in lines:
        if line.top >= page_heights[line.pdf_page - 1] * (1 - FOOTER_ZONE_FRACTION):
            zones[line.pdf_page].append(line)
    footers: dict[int, PageFooter] = {}
    for page, zone in zones.items():
        text = " ".join(ln.text for ln in sorted(zone, key=lambda ln: (ln.x0, ln.top)))
        is_footer, printed = _page_number(text)
        if is_footer:
            footers[page] = PageFooter(pdf_page=page, printed_page=printed, text=text)
    body = tuple(
        line for line in lines if line.pdf_page not in footers or line not in zones[line.pdf_page]
    )
    return FooterSplit(body=body, footers=footers)

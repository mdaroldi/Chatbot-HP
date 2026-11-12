"""Sections: one per bookmark whose heading survived cleaning, with its blocks.

The section path comes from the bookmark tree, not from the heading lines, so a
section keeps the titles of ancestors whose heading line cleaning dropped (a chapter
left with only its opener list, a heading directly followed by its first
subsection). Heading lines are not blocks: the chunk prefix carries the path.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.ingestion import HorizontalRule, OutlineEntry, Section, TextLine
from app.ingestion.blocks import build_blocks
from app.ingestion.cleaning import CleanedDocument
from app.ingestion.outline import section_paths
from app.ingestion.tables import extract_tables


@dataclass(frozen=True)
class SectionedDocument:
    sections: tuple[Section, ...]
    # Kept lines before the first heading (cover, legal front matter): reported,
    # not indexed, because no bookmark gives them a section path.
    unsectioned: tuple[TextLine, ...]


def build_sections(
    doc_id: str,
    cleaned: CleanedDocument,
    outline: Sequence[OutlineEntry],
    rules: Sequence[HorizontalRule],
) -> SectionedDocument:
    lines = cleaned.lines
    tables = extract_tables(lines, rules)
    printed = {page: footer.printed_page for page, footer in cleaned.footers.items()}
    paths = section_paths(outline)
    sections: list[Section] = []
    for span in cleaned.spans:
        # Cleaning keeps a span only if a content line with text survives, and every
        # such line becomes a block: an empty section here is a bug, and Section raises.
        blocks = build_blocks(lines[span.body : span.end], tables, printed)
        sections.append(Section(doc_id=doc_id, path=paths[span.entry], blocks=blocks))
    first = cleaned.spans[0].start if cleaned.spans else len(lines)
    return SectionedDocument(sections=tuple(sections), unsectioned=lines[:first])

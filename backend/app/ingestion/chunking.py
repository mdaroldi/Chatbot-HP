"""Chunking v1: structure-aware chunks of at most `max_tokens`, prefix included (ADR 0005).

Every chunk starts with "document > chapter > ... > section", then its blocks one
per line. A section is split only between units: a step with what belongs to it,
a paragraph, a bullet, a note, or a whole table. A chunk holds at most one table,
next to prose when the whole table fits. A unit too large on its own is split
between its blocks, then between sentences, then between words; tables are split
between rows, repeating the caption and header row. Chunks after the first
of a section repeat the title with "(continued)".
"""

import hashlib
import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from app.domain.ingestion import (
    Block,
    BlockKind,
    Chunk,
    ChunkTooLargeError,
    ContentType,
    CorpusDocument,
    Section,
    TokenCounter,
)
from app.ingestion.front_matter import FRONT_MATTER

VERSION = "v1"
# Fixed forever: chunk ids must not change between runs (uuid5 of NAMESPACE_URL and
# "hpdocs:chunk-ids").
CHUNK_NAMESPACE = uuid.UUID("44422236-3dac-56e6-b77d-fb1c07c18519")

_CONTINUED = " (continued)"
_INDENT_TOLERANCE = 2.0  # points; a note indented more than this belongs to its step
_CAPTION = re.compile(r"Table \d+-\d+\b")
_PART_NUMBER = re.compile(r"\b[A-Z]?\d{5,6}-\d{3}\b")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Bookmark-title words that mark legal and compliance content.
_REGULATORY = re.compile(r"\b(?:regulatory|notices?|environmental|recycling|power cord)\b", re.I)
_SPECS = re.compile(r"\bspecifications?\b", re.I)
_TABLE_KINDS = (BlockKind.TABLE_HEADER, BlockKind.TABLE_ROW)

Measure = Callable[[bool, Sequence[Block]], int]  # (continued, blocks) -> tokens


@dataclass
class _Group:
    """One section, or small sections merged under their parent's path."""

    members: list[Section]

    @property
    def path(self) -> tuple[str, ...]:
        first = self.members[0].path
        return first if len(self.members) == 1 else first[:-1]

    @property
    def blocks(self) -> tuple[Block, ...]:
        if len(self.members) == 1:
            return self.members[0].blocks
        result: list[Block] = []
        for member in self.members:
            title = Block(kind=BlockKind.HEADING, text=member.path[-1], pages=member.pages)
            result.extend((title, *member.blocks))
        return tuple(result)


@dataclass
class _Unit:
    blocks: list[Block] = field(default_factory=list)
    header: list[Block] = field(default_factory=list)  # tables: repeated on each chunk
    step_indent: float | None = None

    @property
    def is_table(self) -> bool:
        return any(block.kind in _TABLE_KINDS for block in self.header + self.blocks)


def _is_caption(blocks: Sequence[Block], position: int) -> bool:
    following = blocks[position + 1 : position + 2]
    return bool(
        _CAPTION.match(blocks[position].text)
        and following
        and following[0].kind is BlockKind.TABLE_HEADER
    )


def _units(blocks: Sequence[Block]) -> list[_Unit]:
    """Headings and outer hazard notes lead the unit after them; blocks indented
    under a step belong to that step; table rows belong to their table.
    """
    units: list[_Unit] = []
    leading: list[Block] = []
    for position, block in enumerate(blocks):
        last = units[-1] if units else None
        if block.kind is BlockKind.TABLE_ROW and last and last.is_table:
            last.blocks.append(block)
        elif block.kind is BlockKind.TABLE_HEADER or _is_caption(blocks, position):
            if last and not last.is_table and last.header and not last.blocks:
                last.header.append(block)  # the header after its caption
            else:
                units.append(_Unit(header=[*leading, block]))
                leading = []
        elif block.kind is BlockKind.HEADING:
            leading.append(block)
        elif (
            last
            and last.step_indent is not None
            and not leading
            and block.indent > last.step_indent + _INDENT_TOLERANCE
        ):
            last.blocks.append(block)
        elif block.is_hazard:
            leading.append(block)
        else:
            indent = block.indent if block.kind is BlockKind.STEP else None
            units.append(_Unit(blocks=[*leading, block], step_indent=indent))
            leading = []
    if leading:
        units.append(_Unit(blocks=leading))
    return units


class _Packer:
    """Greedy packing of units into chunks (lists of blocks) that fit."""

    def __init__(self, measure: Measure, max_tokens: int, where: str) -> None:
        self._measure = measure
        self._max = max_tokens
        self._where = where  # "doc_id: section path", for errors
        self.chunks: list[list[Block]] = []
        self._current: list[Block] = []

    def _ok(self, blocks: Sequence[Block]) -> bool:
        return self._measure(bool(self.chunks), blocks) <= self._max

    def _require_fit(self, blocks: Sequence[Block]) -> None:
        """Words and table rows are never cut: one that fits no chunk is an error."""
        if not self._ok(blocks):
            tokens = self._measure(bool(self.chunks), blocks)
            raise ChunkTooLargeError(
                f"{self._where}: an unsplittable piece needs {tokens} tokens, limit {self._max}"
            )

    def flush(self) -> None:
        if self._current:
            self.chunks.append(self._current)
            self._current = []

    def add(self, unit: _Unit) -> None:
        whole = [*unit.header, *unit.blocks]
        has_table = any(block.kind in _TABLE_KINDS for block in self._current)
        if unit.is_table and (has_table or not self._ok([*self._current, *whole])):
            self._table(unit)
        elif unit.is_table:
            self._current.extend(whole)
        elif self._ok([*self._current, *unit.blocks]):
            self._current.extend(unit.blocks)
        else:
            self.flush()
            for block in unit.blocks:
                self._block(block)

    def _block(self, block: Block) -> None:
        if self._ok([*self._current, block]):
            self._current.append(block)
            return
        self.flush()
        if self._ok([block]):
            self._current.append(block)
            return
        self._split(block)

    def _split(self, block: Block) -> None:
        pieces: list[str] = []
        for sentence in _SENTENCE_END.split(block.text):
            fits_alone = self._measure(True, [replace(block, text=sentence)]) <= self._max
            pieces.extend([sentence] if fits_alone else sentence.split())
        taken: list[str] = []
        for piece in pieces:
            if self._ok([*self._current, replace(block, text=" ".join([*taken, piece]))]):
                taken.append(piece)
                continue
            if taken:
                self._current.append(replace(block, text=" ".join(taken)))
            self.flush()
            self._require_fit([replace(block, text=piece)])
            taken = [piece]
        self._current.append(replace(block, text=" ".join(taken)))

    def _table(self, unit: _Unit) -> None:
        """Split between rows, repeating the header; a row is never cut."""
        self.flush()
        rows: list[Block] = []
        for row in unit.blocks:
            if rows and not self._ok([*unit.header, *rows, row]):
                self.chunks.append([*unit.header, *rows])
                rows = []
            if not rows:
                self._require_fit([*unit.header, row])
            rows.append(row)
        self._current = [*unit.header, *rows]  # prose after the table may join it


def _content_type(sources: Sequence[tuple[str, ...]], blocks: Sequence[Block]) -> ContentType:
    if all(path[0] == FRONT_MATTER for path in sources):
        return ContentType.FRONT_MATTER
    titles = [title for path in sources for title in path]
    if any(_REGULATORY.search(title) for title in titles):
        return ContentType.REGULATORY
    if any(_SPECS.search(title) for title in titles):
        return ContentType.SPECS
    if any(block.kind in _TABLE_KINDS for block in blocks):
        rows = (block.text for block in blocks if block.kind is BlockKind.TABLE_ROW)
        return ContentType.PARTS if any(_PART_NUMBER.search(r) for r in rows) else ContentType.TABLE
    if any(block.kind is BlockKind.STEP for block in blocks):
        return ContentType.PROCEDURE
    return ContentType.REFERENCE


class StructureChunker:
    """Chunker v1; `max_tokens` includes the prefix, `min_tokens` triggers merging."""

    def __init__(self, tokens: TokenCounter, max_tokens: int, min_tokens: int) -> None:
        self._tokens = tokens
        self._max = max_tokens
        self._min = min_tokens

    @property
    def version(self) -> str:
        return VERSION

    @staticmethod
    def _text(
        document: CorpusDocument, path: Sequence[str], continued: bool, blocks: Sequence[Block]
    ) -> str:
        title = " > ".join((document.title, *path)) + (_CONTINUED if continued else "")
        return title + "\n\n" + "\n".join(block.text for block in blocks)

    def _count(self, document: CorpusDocument, group: _Group) -> int:
        return self._tokens.count(self._text(document, group.path, False, group.blocks))

    def _groups(self, sections: Sequence[Section], document: CorpusDocument) -> list[_Group]:
        """A section under `min_tokens` absorbs the next section under the same parent,
        while the result stays small and within the limit.
        """
        groups: list[_Group] = []
        position = 0
        while position < len(sections):
            group = _Group([sections[position]])
            parent = group.path[:-1]
            position += 1
            while (
                parent
                and position < len(sections)
                and sections[position].path[:-1] == parent
                and self._count(document, group) < self._min
            ):
                merged = _Group([*group.members, sections[position]])
                if self._count(document, merged) > self._max:
                    break
                group = merged
                position += 1
            groups.append(group)
        return groups

    def chunk(self, sections: Sequence[Section], document: CorpusDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for group in self._groups(sections, document):
            path = group.path

            def measure(
                continued: bool, blocks: Sequence[Block], path: tuple[str, ...] = path
            ) -> int:
                return self._tokens.count(self._text(document, path, continued, blocks))

            where = f"{document.doc_id}: {' > '.join(path)}"
            packer = _Packer(measure, self._max, where)
            for unit in _units(group.blocks):
                packer.add(unit)
            packer.flush()
            sources = [member.path for member in group.members]
            for index, blocks in enumerate(packer.chunks):
                chunks.append(self._chunk(document, path, sources, index, blocks))
        return chunks

    def _chunk(
        self,
        document: CorpusDocument,
        path: tuple[str, ...],
        sources: Sequence[tuple[str, ...]],
        index: int,
        blocks: Sequence[Block],
    ) -> Chunk:
        text = self._text(document, path, index > 0, blocks)
        digest = hashlib.sha256(text.encode()).hexdigest()
        pages = blocks[0].pages
        for block in blocks[1:]:
            pages = pages.union(block.pages)
        return Chunk(
            chunk_id=str(
                uuid.uuid5(
                    CHUNK_NAMESPACE, f"{document.doc_id}|{' > '.join(path)}|{index}|{digest}"
                )
            ),
            doc_id=document.doc_id,
            product=document.product,
            section_path=path,
            pages=pages,
            content_type=_content_type(sources, blocks),
            has_warning=any(block.is_hazard for block in blocks),
            chunking_version=VERSION,
            token_count=self._tokens.count(text),
            text=text,
        )

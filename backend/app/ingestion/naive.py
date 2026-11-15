"""Naive chunking: the Phase 5 ablation baseline for the structure-aware chunker.

Fixed windows of `WINDOW_TOKENS` dense-model tokens, title prefix included, moving
forward so that consecutive windows share at most `OVERLAP_TOKENS` tokens of text. They
read the same parsed and cleaned text, in reading order, as chunking v2: each section's
heading line, then its blocks, one per line. Only the chunker differs. Windows ignore
section and block bounds, and the only prefix is the document title, because the
"chapter > section" breadcrumb is part of what the ablation measures. For the same
reason `section_path` holds the document title alone: the excerpt the model sees
carries no breadcrumb either.

Window and overlap are constants, not settings: `--if-needed` compares the chunking
version, so the version must name the exact windowing.

Windows are cut between words. The tokenizer splits on whitespace before its sub-word
step, so a text's count is its words' counts added up, plus what the empty text costs
(the special tokens). That lets each word be measured once.
"""

import hashlib
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise

from app.domain.ingestion import (
    Chunk,
    ChunkTooLargeError,
    ContentType,
    CorpusDocument,
    PageRange,
    Section,
    TokenCounter,
)
from app.ingestion.chunking import CHUNK_NAMESPACE

VERSION = "naive-v1"
WINDOW_TOKENS = 256
OVERLAP_TOKENS = 64


@dataclass(frozen=True)
class _Word:
    text: str
    line: int  # words of one heading or block share a line
    pages: PageRange
    hazard: bool


def _lines(sections: Sequence[Section]) -> Iterator[tuple[str, PageRange, bool]]:
    """The text in reading order: a heading line before each section's blocks."""
    for section in sections:
        yield section.path[-1], section.blocks[0].pages, False
        for block in section.blocks:
            yield block.text, block.pages, block.is_hazard


def _words(sections: Sequence[Section]) -> list[_Word]:
    return [
        _Word(word, line, pages, hazard)
        for line, (text, pages, hazard) in enumerate(_lines(sections))
        for word in text.split()
    ]


def _join(words: Sequence[_Word]) -> str:
    parts = [words[0].text]
    for previous, word in pairwise(words):
        parts.append(("\n" if word.line != previous.line else " ") + word.text)
    return "".join(parts)


class NaiveChunker:
    """Fixed token windows with overlap; see the module docstring."""

    def __init__(
        self,
        tokens: TokenCounter,
        window_tokens: int = WINDOW_TOKENS,
        overlap_tokens: int = OVERLAP_TOKENS,
    ) -> None:
        if not 0 <= overlap_tokens < window_tokens:
            raise ValueError("the overlap must be at least 0 and below a positive window")
        self._tokens = tokens
        self.window_tokens = window_tokens
        self.overlap_tokens = overlap_tokens

    @property
    def version(self) -> str:
        return VERSION

    def _cost(self, text: str, empty: int) -> int:
        return self._tokens.count(text) - empty

    def chunk(self, sections: Sequence[Section], document: CorpusDocument) -> list[Chunk]:
        words = _words(sections)
        empty = self._tokens.count("")
        budget = self.window_tokens - empty - self._cost(document.title, empty)
        cache: dict[str, int] = {}
        for word in words:
            if word.text not in cache:
                cache[word.text] = self._cost(word.text, empty)
        costs = [cache[word.text] for word in words]
        chunks: list[Chunk] = []
        start = 0
        while start < len(words):
            end, used = start, 0
            while end < len(words) and used + costs[end] <= budget:
                used += costs[end]
                end += 1
            if end == start:
                raise ChunkTooLargeError(
                    f"{document.doc_id}: a word needs {costs[start]} tokens,"
                    f" {max(budget, 0)} left in a window of {self.window_tokens}"
                )
            chunks.append(self._chunk(document, len(chunks), words[start:end]))
            if end == len(words):
                break
            start = self._next_start(costs, start, end)
        return chunks

    def _next_start(self, costs: Sequence[int], start: int, end: int) -> int:
        """The earliest word after `start` whose run to `end` fits in the overlap."""
        position, tail = end, 0
        while position - 1 > start and tail + costs[position - 1] <= self.overlap_tokens:
            position -= 1
            tail += costs[position]
        return position

    def _chunk(self, document: CorpusDocument, index: int, words: Sequence[_Word]) -> Chunk:
        text = document.title + "\n\n" + _join(words)
        digest = hashlib.sha256(text.encode()).hexdigest()
        pages = words[0].pages
        for word in words[1:]:
            pages = pages.union(word.pages)
        return Chunk(
            chunk_id=str(
                uuid.uuid5(CHUNK_NAMESPACE, f"{document.doc_id}|{VERSION}|{index}|{digest}")
            ),
            doc_id=document.doc_id,
            product=document.product,
            section_path=(document.title,),
            pages=pages,
            content_type=ContentType.TEXT,
            has_warning=any(word.hazard for word in words),
            chunking_version=VERSION,
            token_count=self._tokens.count(text),
            text=text,
        )

"""Anchor matching: which indexed chunks satisfy an anchor of the golden set.

A chunk satisfies an anchor when it belongs to the anchor's document, the optional
section fragment occurs in its section path, and every fact occurs in its text. Both
sides are normalized first (`normalize`). Facts are searched in the whole text,
breadcrumb prefix included; the section is searched in the path only, because a
merged chunk carries its parent's path while its children's titles are text.
"""

from collections.abc import Iterable, Sequence
from functools import lru_cache

from app.domain.evaluation import Anchor, IndexedChunk
from app.evaluation.normalize import normalize

_PATH_SEPARATOR = " > "


@lru_cache(maxsize=8192)
def _normalized(text: str) -> str:
    """Chunks are matched against many anchors and many rankings; normalize each once."""
    return normalize(text)


def anchor_matches(anchor: Anchor, chunk: IndexedChunk) -> bool:
    if anchor.doc_id != chunk.doc_id:
        return False
    if anchor.section is not None:
        path = _normalized(_PATH_SEPARATOR.join(chunk.section_path))
        if _normalized(anchor.section) not in path:
            return False
    text = _normalized(chunk.text)
    return all(_normalized(fact) in text for fact in anchor.facts)


def matching_chunks(anchor: Anchor, chunks: Iterable[IndexedChunk]) -> tuple[str, ...]:
    """Chunk ids satisfying the anchor, in input order (the preflight's resolution)."""
    return tuple(chunk.chunk_id for chunk in chunks if anchor_matches(anchor, chunk))


def satisfied_anchors(anchors: Sequence[Anchor], chunk: IndexedChunk) -> frozenset[int]:
    return frozenset(n for n, anchor in enumerate(anchors) if anchor_matches(anchor, chunk))

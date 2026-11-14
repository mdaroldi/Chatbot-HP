"""The excerpts the model was given, rebuilt from the chunk ids stored with its answer.

The stored ids are in rank order, so the nth id is tag Sn. Payloads come from Qdrant by
point id through the alias the run used, and each excerpt is rendered by the answer
prompt's own renderer, so the text scored and judged is byte for byte the text the
model read. An id the collection lacks means the index changed under the run: that is
an error, never a shorter excerpt list.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol

from qdrant_client import AsyncQdrantClient, models

from app.domain.evaluation import IndexedChunk
from app.domain.prompt import render_excerpt
from app.domain.retrieval import SearchHit
from app.evaluation.answers.scoring import Excerpt
from app.infra.qdrant_search import to_hit


class ExcerptError(Exception):
    """A stored chunk id cannot be resolved; the message names tags and ids only."""


class ChunkSource(Protocol):
    async def hits(self, chunk_ids: Sequence[str]) -> Mapping[str, SearchHit]:
        """The hits for the ids the collection holds; missing ids are left out."""
        ...


class QdrantChunkSource:
    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self._client, self._collection = client, collection

    async def hits(self, chunk_ids: Sequence[str]) -> Mapping[str, SearchHit]:
        records = await self._client.retrieve(
            self._collection, ids=list(chunk_ids), with_payload=True, with_vectors=False
        )
        # A point read by id has no score; wrapping it reuses the search's payload
        # parsing, and the renderer never reads the score.
        return {
            str(record.id): to_hit(
                models.ScoredPoint(id=record.id, version=0, score=0.0, payload=record.payload)
            )
            for record in records
        }


async def rebuild_excerpts(
    chunk_ids: Sequence[str], source: ChunkSource, titles: Mapping[str, str]
) -> tuple[Excerpt, ...]:
    hits = await source.hits(chunk_ids)
    excerpts = []
    for number, chunk_id in enumerate(chunk_ids, 1):
        tag = f"S{number}"
        hit = hits.get(chunk_id)
        if hit is None:
            raise ExcerptError(f"excerpt {tag} ({chunk_id}) is not in the collection")
        chunk = IndexedChunk(hit.chunk_id, hit.doc_id, hit.section_path, hit.text)
        excerpts.append(Excerpt(tag, chunk, render_excerpt(tag, hit, titles)))
    return tuple(excerpts)

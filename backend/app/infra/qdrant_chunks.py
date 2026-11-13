"""Every chunk of one collection, read back for the benchmark preflight.

The preflight resolves golden anchors against what was actually indexed, so it reads
the payload Qdrant holds (never the vectors), page by page through the scroll API.
"""

from qdrant_client import AsyncQdrantClient

from app.domain.evaluation import IndexedChunk

_FIELDS = ["doc_id", "section_path", "text"]


async def scroll_chunks(
    client: AsyncQdrantClient, collection: str, *, page_size: int = 256
) -> list[IndexedChunk]:
    chunks: list[IndexedChunk] = []
    offset = None
    while True:
        points, offset = await client.scroll(
            collection, limit=page_size, offset=offset, with_payload=_FIELDS, with_vectors=False
        )
        for point in points:
            payload = point.payload or {}
            chunks.append(
                IndexedChunk(
                    chunk_id=str(point.id),
                    doc_id=str(payload["doc_id"]),
                    section_path=tuple(str(part) for part in payload["section_path"]),
                    text=str(payload["text"]),
                )
            )
        if offset is None:
            return chunks

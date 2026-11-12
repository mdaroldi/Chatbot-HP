"""VectorIndex over Qdrant: one collection per build, named dense and sparse vectors."""

from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient, models

from app.domain.ingestion import Chunk, SparseVector

DENSE = "dense"
SPARSE = "sparse"
# Payload fields that retrieval filters on (Phase 2): per manual and per content type.
_KEYWORD_FIELDS = ("doc_id", "content_type")


class QdrantIndex:
    """Thin adapter; the build, check and swap order lives in `app.ingestion.publish`.

    Every write waits for the server to apply it, so a count right after an upsert
    sees every point.
    """

    def __init__(self, client: AsyncQdrantClient, upsert_batch_size: int = 128) -> None:
        self._client = client
        self._batch = upsert_batch_size

    async def create_collection(self, name: str, dense_size: int) -> None:
        await self._client.create_collection(
            name,
            vectors_config={
                DENSE: models.VectorParams(size=dense_size, distance=models.Distance.COSINE)
            },
            # BM25 vectors carry term frequencies only; Qdrant applies IDF at query time.
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field in _KEYWORD_FIELDS:
            await self._client.create_payload_index(
                name, field, models.PayloadSchemaType.KEYWORD, wait=True
            )

    async def upsert(
        self,
        name: str,
        chunks: Sequence[Chunk],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[SparseVector],
    ) -> None:
        # Built in full first: a length mismatch fails before anything is written.
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector={
                    DENSE: list(dense_vector),
                    SPARSE: models.SparseVector(
                        indices=list(sparse_vector.indices), values=list(sparse_vector.values)
                    ),
                },
                payload=chunk.payload(),
            )
            for chunk, dense_vector, sparse_vector in zip(chunks, dense, sparse, strict=True)
        ]
        for start in range(0, len(points), self._batch):
            await self._client.upsert(name, points[start : start + self._batch], wait=True)

    async def count(self, name: str) -> int:
        return (await self._client.count(name, exact=True)).count

    async def alias_target(self, alias: str) -> str | None:
        aliases = (await self._client.get_aliases()).aliases
        return next((a.collection_name for a in aliases if a.alias_name == alias), None)

    async def point_alias(self, alias: str, name: str) -> None:
        """Move (or create) the alias in one request, which Qdrant applies atomically:
        readers see the old collection or the new one, never neither.
        """
        operations: list[models.AliasOperations] = []
        if await self.alias_target(alias) is not None:
            operations.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=name, alias_name=alias)
            )
        )
        await self._client.update_collection_aliases(change_aliases_operations=operations)

    async def delete_collection(self, name: str) -> None:
        await self._client.delete_collection(name)

    async def list_collections(self, prefix: str) -> list[str]:
        collections = (await self._client.get_collections()).collections
        return sorted(c.name for c in collections if c.name.startswith(prefix))

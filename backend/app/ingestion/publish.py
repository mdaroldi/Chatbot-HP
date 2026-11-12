"""Publish a build: a new versioned collection goes live only after it checks out.

Order: create the build's collection, upload, verify the point count, move the alias
in one atomic step, then prune old builds. Until the alias moves, readers keep the
previous collection; on any failure before that, the new collection is deleted.
"""

import re
from dataclasses import dataclass

from app.domain.ingestion import EmbeddedChunks, IndexPublishError, VectorIndex

# "v1" is the collection schema (named vectors and payload); a new schema gets a new
# prefix, so pruning never deletes collections it does not understand.
COLLECTION_PREFIX = "hpdocs_chunks_v1_"
_BUILD_ID = re.compile(r"[a-z0-9_-]+")


@dataclass(frozen=True)
class Published:
    collection: str
    previous: str | None  # the alias target before the swap, if any
    pruned: tuple[str, ...]


async def _verify(index: VectorIndex, name: str, expected: int) -> None:
    stored = await index.count(name)
    if stored != expected:
        raise IndexPublishError(f"{name}: {stored} points for {expected} chunks")


def collection_name(build_id: str) -> str:
    if not _BUILD_ID.fullmatch(build_id):
        raise ValueError(f"build id {build_id!r} must be lowercase letters, digits, - or _")
    return COLLECTION_PREFIX + build_id


async def publish(
    index: VectorIndex,
    alias: str,
    build_id: str,
    embedded: EmbeddedChunks,
    *,
    keep_previous: bool,
) -> Published:
    """Make `embedded` the live index under `alias`, or leave the live index as it was.

    A count below the chunk count means duplicate chunk ids (Qdrant keeps one point
    per id), so the check also catches a chunker that stops producing unique ids.
    """
    name = collection_name(build_id)
    await index.create_collection(name, embedded.dense_size)
    try:
        await index.upsert(name, embedded.chunks, embedded.dense, embedded.sparse)
        await _verify(index, name, len(embedded.chunks))
        previous = await index.alias_target(alias)
        await index.point_alias(alias, name)
    except BaseException:
        # A swap can succeed on the server and still fail here (a lost reply); the
        # collection is then live and must stay.
        if await index.alias_target(alias) != name:
            await index.delete_collection(name)
        raise
    kept = {name, previous} if keep_previous and previous else {name}
    pruned = tuple(c for c in await index.list_collections(COLLECTION_PREFIX) if c not in kept)
    for collection in pruned:
        await index.delete_collection(collection)
    return Published(collection=name, previous=previous, pruned=pruned)

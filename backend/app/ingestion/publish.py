"""Publish a build: a new versioned collection goes live only after it checks out.

Order: create the build's collection, upload, verify the point count, move the alias
in one atomic step, then prune old builds. Until the alias moves, readers keep the
previous collection; on any failure before that, the new collection is deleted. A
failure while pruning comes after the swap, so it is a warning, never a failed build.
"""

import logging
import re
from dataclasses import dataclass

from app.domain.ingestion import (
    BuildWarning,
    BuildWarningKind,
    EmbeddedChunks,
    IndexPublishError,
    VectorIndex,
)

logger = logging.getLogger(__name__)

# Collections are `hpdocs_chunks_<chunking version>_<build id>`. A new chunking
# version gets a new prefix, and pruning stays inside the live version's prefix, so it
# never deletes collections built by another chunker.
_BASE = "hpdocs_chunks_"
_VERSION = re.compile(r"[a-z0-9]+")
_BUILD_ID = re.compile(r"[a-z0-9_-]+")


@dataclass(frozen=True)
class Published:
    collection: str
    previous: str | None  # the alias target before the swap, if any
    pruned: tuple[str, ...]  # old builds actually deleted
    warnings: tuple[BuildWarning, ...] = ()


def collection_prefix(chunking_version: str) -> str:
    if not _VERSION.fullmatch(chunking_version):
        raise ValueError(
            f"chunking version {chunking_version!r} must be lowercase letters or digits"
        )
    return f"{_BASE}{chunking_version}_"


def collection_name(chunking_version: str, build_id: str) -> str:
    if not _BUILD_ID.fullmatch(build_id):
        raise ValueError(f"build id {build_id!r} must be lowercase letters, digits, - or _")
    return collection_prefix(chunking_version) + build_id


async def _verify(index: VectorIndex, name: str, expected: int) -> None:
    stored = await index.count(name)
    if stored != expected:
        raise IndexPublishError(f"{name}: {stored} points for {expected} chunks")


async def _prune(
    index: VectorIndex, prefix: str, kept: set[str]
) -> tuple[tuple[str, ...], tuple[BuildWarning, ...]]:
    """Delete old builds one by one; a failure leaves that one and moves on.

    The new build is already live, so a leftover collection only costs disk space; it
    is logged and recorded (type name only, never the message) and pruned next time.
    """
    pruned: list[str] = []
    warnings: list[BuildWarning] = []
    for collection in await index.list_collections(prefix):
        if collection in kept:
            continue
        try:
            await index.delete_collection(collection)
        except Exception as exc:
            error_type = type(exc).__name__
            logger.warning(
                "could not prune an old index build",
                extra={"collection": collection, "error_type": error_type},
            )
            warnings.append(BuildWarning(BuildWarningKind.PRUNE_FAILED, collection, error_type))
        else:
            pruned.append(collection)
    return tuple(pruned), tuple(warnings)


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
    name = collection_name(embedded.chunking_version, build_id)
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
    pruned, warnings = await _prune(index, collection_prefix(embedded.chunking_version), kept)
    return Published(collection=name, previous=previous, pruned=pruned, warnings=warnings)

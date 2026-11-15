"""The chunkers `CHUNKER` selects, by the chunking version each stamps on its chunks.

The ingest CLI records the version before chunking starts, and the answer benchmark
checks that an alias's live collection carries its chunker's version prefix.
"""

from collections.abc import Mapping

from app.domain.ingestion import ChunkerKind
from app.ingestion import chunking, naive

CHUNKING_VERSIONS: Mapping[ChunkerKind, str] = {
    ChunkerKind.STRUCTURE: chunking.VERSION,
    ChunkerKind.NAIVE: naive.VERSION,
}

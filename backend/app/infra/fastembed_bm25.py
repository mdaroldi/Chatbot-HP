"""BM25 document lengths exactly as fastembed's BM25 counts them (ADR 0009).

`EMBEDDING_BM25_AVG_LEN=auto` sets avg_len to the mean length of the chunks just built.
The length must be the `|d|` of fastembed's formula, not a token count of our own:
the same tokenizer, punctuation and stopword filters, length cap and stemmer. fastembed
has no public call for it, so this reuses its `_stem` step on its tokenizer's output,
which is what its `raw_embed` does before weighting. A unit test checks the result
against fastembed's own weights, so an upstream change cannot slip through.
"""

from collections.abc import Sequence
from pathlib import Path

from fastembed.common.utils import remove_non_alphanumeric
from fastembed.sparse.bm25 import Bm25

from app.domain.ingestion import VerifiedModel


class FastembedBm25Lengths:
    """Lengths from the verified BM25 folder, offline. Pure CPU: call it off the loop."""

    def __init__(self, sparse: VerifiedModel, cache_dir: Path) -> None:
        self._model = Bm25(
            sparse.files.name,
            cache_dir=str(cache_dir),
            specific_model_path=str(sparse.path),
            local_files_only=True,
        )

    def lengths(self, texts: Sequence[str]) -> list[int]:
        model = self._model
        return [
            len(model._stem(model.tokenizer.tokenize(remove_non_alphanumeric(text))))
            for text in texts
        ]

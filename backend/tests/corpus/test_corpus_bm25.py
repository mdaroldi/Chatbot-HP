"""BM25 document parameters on the real chunks (ADR 0009, Phase 2 commit 10b).

Every build now passes k, b and avg_len to fastembed explicitly. With the default
setting, the stored sparse vectors must be exactly the ones the index was built with
before: fastembed's own defaults, which the earlier code relied on by passing nothing.

CI logs are public: assertions compare chunk IDs and numbers only (tests/corpus/conftest.py).
"""

import asyncio
from collections.abc import Mapping

import pytest
from fastembed import SparseTextEmbedding
from pydantic import SecretStr

from app.core.settings import Settings
from app.domain.ingestion import Bm25Params, Chunk, mean_length
from app.infra.fastembed_bm25 import FastembedBm25Lengths
from app.infra.fastembed_embedder import FastembedEmbedder
from app.infra.fastembed_tokens import FastembedTokenCounter
from app.ingestion.chunking import StructureChunker
from tests.corpus.conftest import Hidden, ParsedCorpusDocument
from tests.corpus.test_corpus_tokens import MAX_TOKENS, MIN_TOKENS
from tests.model_data import fetch_verified

pytestmark = [pytest.mark.corpus, pytest.mark.model]

SETTINGS = Settings(postgres_password=SecretStr("unused"))
# The mean BM25 length of the 287 chunks of chunking v2, measured at commit 10b: 30,783
# stemmed tokens. A change means the chunks or the BM25 files changed.
AUTO_AVG_LEN = 30_783 / 287


@pytest.fixture(scope="module")
def chunks(corpus: Mapping[str, ParsedCorpusDocument]) -> Hidden[list[Chunk]]:
    dense = fetch_verified(SETTINGS, SETTINGS.embedding_dense_model)
    counter = FastembedTokenCounter.from_settings(SETTINGS, dense)
    chunker = StructureChunker(counter, max_tokens=MAX_TOKENS, min_tokens=MIN_TOKENS)
    return Hidden(
        {
            doc_id: chunker.chunk(doc.sectioned.sections, doc.parsed.document)
            for doc_id, doc in corpus.items()
        }
    )


def every_chunk(chunks: Mapping[str, list[Chunk]]) -> list[Chunk]:
    return [chunk for doc_id in sorted(chunks) for chunk in chunks[doc_id]]


def test_the_default_setting_gives_the_vectors_the_index_was_built_with(
    chunks: Mapping[str, list[Chunk]],
) -> None:
    found = every_chunk(chunks)
    texts = [chunk.text for chunk in found]
    dense = fetch_verified(SETTINGS, SETTINGS.embedding_dense_model)
    sparse = fetch_verified(SETTINGS, SETTINGS.embedding_sparse_model)
    assert SETTINGS.embedding_bm25_avg_len == 256.0
    embedder = FastembedEmbedder.from_settings(
        SETTINGS, dense, sparse, bm25=Bm25Params(avg_len=SETTINGS.embedding_bm25_avg_len)
    )
    # How every build before 10b made them: no BM25 keyword, so fastembed's defaults.
    before = SparseTextEmbedding(
        sparse.files.name,
        cache_dir=str(SETTINGS.embedding_cache_dir),
        lazy_load=False,
        specific_model_path=str(sparse.path),
        local_files_only=True,
    )

    now = asyncio.run(embedder.embed_sparse(texts))
    then = list(before.embed(texts, batch_size=SETTINGS.embedding_batch_size))

    assert len(now) == len(then) == len(found) == 287
    differing = [
        chunk.chunk_id
        for chunk, new, old in zip(found, now, then, strict=True)
        if (new.indices, new.values)
        != (tuple(int(i) for i in old.indices), tuple(float(v) for v in old.values))
    ]
    assert differing == []


def test_auto_resolves_to_the_measured_mean_bm25_length(
    chunks: Mapping[str, list[Chunk]],
) -> None:
    sparse = fetch_verified(SETTINGS, SETTINGS.embedding_sparse_model)
    lengths = FastembedBm25Lengths(sparse, SETTINGS.embedding_cache_dir).lengths(
        [chunk.text for chunk in every_chunk(chunks)]
    )

    assert len(lengths) == 287
    assert min(lengths) > 0
    assert mean_length(lengths) == pytest.approx(AUTO_AVG_LEN, abs=1e-9)

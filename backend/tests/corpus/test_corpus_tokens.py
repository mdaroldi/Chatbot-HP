"""Chunking v1 with the real bge-small tokenizer on both manuals (ADR 0005)."""

import hashlib
from collections.abc import Mapping

import pytest
from pydantic import SecretStr

from app.core.settings import Settings
from app.domain.ingestion import Chunk
from app.infra.fastembed_tokens import FastembedTokenCounter
from app.ingestion.chunking import StructureChunker
from tests.corpus.conftest import ENVY, OMEN, ParsedCorpusDocument
from tests.model_data import fetch_verified

pytestmark = [pytest.mark.corpus, pytest.mark.model]

Corpus = Mapping[str, ParsedCorpusDocument]
BOTH = pytest.mark.parametrize("doc_id", [ENVY, OMEN])

MAX_TOKENS = 450
# Re-measured at commit 7: 65 tokens is the floor that best reproduces the merges of
# the earlier 50-word floor (25 mergeable sections under each, one differs each way).
MIN_TOKENS = 65
CHUNKS = {ENVY: 149, OMEN: 141}


@pytest.fixture(scope="module")
def counter() -> FastembedTokenCounter:
    settings = Settings(postgres_password=SecretStr("unused"))
    dense = fetch_verified(settings, settings.embedding_dense_model)
    return FastembedTokenCounter.from_settings(settings, dense)


@pytest.fixture(scope="module")
def chunks(corpus: Corpus, counter: FastembedTokenCounter) -> dict[str, list[Chunk]]:
    chunker = StructureChunker(counter, max_tokens=MAX_TOKENS, min_tokens=MIN_TOKENS)
    return {
        doc_id: chunker.chunk(doc.sectioned.sections, doc.parsed.document)
        for doc_id, doc in corpus.items()
    }


@BOTH
def test_no_chunk_exceeds_the_limit_including_its_prefix(
    chunks: Mapping[str, list[Chunk]], counter: FastembedTokenCounter, doc_id: str
) -> None:
    found = chunks[doc_id]

    assert len(found) == CHUNKS[doc_id]
    assert [c.token_count for c in found] == [counter.count(c.text) for c in found]
    assert max(c.token_count for c in found) <= MAX_TOKENS


# Pinned at commit 10. A change here means cleaning or chunking output changed: bump
# chunking_version (ingest --if-needed compares the version, not the code) and re-pin.
CHUNK_ID_DIGESTS = {
    ENVY: "b5856b1139e48a76f3d8284efae0c59e692efec7d99f54e34bd1e557b1911176",
    OMEN: "42db8683ae6822fce8fc7286b67f2dff6d78af144bcf7404bc7baf08c2c28b44",
}


def test_the_settings_defaults_are_the_measured_limits() -> None:
    settings = Settings(postgres_password=SecretStr("unused"))

    assert (settings.chunk_max_tokens, settings.chunk_min_tokens) == (MAX_TOKENS, MIN_TOKENS)


@BOTH
def test_chunk_ids_are_pinned_per_document(chunks: Mapping[str, list[Chunk]], doc_id: str) -> None:
    ids = sorted(c.chunk_id for c in chunks[doc_id])
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()

    assert len(set(ids)) == len(ids) == CHUNKS[doc_id]
    assert digest == CHUNK_ID_DIGESTS[doc_id]

"""Chunking v1 with the real bge-small tokenizer on both manuals (ADR 0005)."""

from collections.abc import Mapping

import pytest
from pydantic import SecretStr

from app.core.settings import Settings
from app.domain.ingestion import Chunk
from app.infra.fastembed_tokens import FastembedTokenCounter
from app.ingestion.chunking import StructureChunker
from tests.corpus.conftest import ENVY, OMEN, ParsedCorpusDocument

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
    return FastembedTokenCounter.from_settings(Settings(postgres_password=SecretStr("unused")))


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

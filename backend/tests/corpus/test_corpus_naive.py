"""The naive baseline chunker with the real bge-small tokenizer on both manuals (Phase 5).

The chunker measures each word once and adds the counts up; these tests check that the
real tokenizer agrees, so no window passes the model's input unseen.

CI logs are public: assertions compare counts and digests only (see tests/corpus/conftest.py).
"""

import hashlib
from collections.abc import Mapping

import pytest
from pydantic import SecretStr

from app.core.settings import Settings
from app.domain.ingestion import Chunk
from app.infra.fastembed_tokens import FastembedTokenCounter
from app.ingestion.naive import WINDOW_TOKENS, NaiveChunker
from tests.corpus.conftest import ENVY, OMEN, Hidden, ParsedCorpusDocument
from tests.model_data import fetch_verified

pytestmark = [pytest.mark.corpus, pytest.mark.model]

Corpus = Mapping[str, ParsedCorpusDocument]
BOTH = pytest.mark.parametrize("doc_id", [ENVY, OMEN])

# Pinned when naive-v1 was written. A change means cleaning or the naive chunker changed:
# bump its VERSION (ingest --if-needed compares the version, not the code) and re-pin.
CHUNKS = {ENVY: 170, OMEN: 130}
CHUNK_ID_DIGESTS = {
    ENVY: "d6f7a9b13a14c73a071ef908b2858acb7c86318aa5a531646a46562ef10f564f",
    OMEN: "e29d5a8c18cddde6393197a70b02c3c94f7aee549f32297dc7903c0dd540d338",
}
# A full window ends when the next word does not fit; no word costs this much.
FULL_WINDOW_SLACK = 16


@pytest.fixture(scope="module")
def counter() -> FastembedTokenCounter:
    settings = Settings(postgres_password=SecretStr("unused"))
    dense = fetch_verified(settings, settings.embedding_dense_model)
    return FastembedTokenCounter.from_settings(settings, dense)


@pytest.fixture(scope="module")
def chunks(corpus: Corpus, counter: FastembedTokenCounter) -> Hidden[list[Chunk]]:
    chunker = NaiveChunker(counter)
    return Hidden(
        {
            doc_id: chunker.chunk(doc.sectioned.sections, doc.parsed.document)
            for doc_id, doc in corpus.items()
        }
    )


@BOTH
def test_no_window_exceeds_its_budget_by_the_real_tokenizer(
    chunks: Mapping[str, list[Chunk]], counter: FastembedTokenCounter, doc_id: str
) -> None:
    found = chunks[doc_id]
    stored, recounted = [c.token_count for c in found], [counter.count(c.text) for c in found]
    full = [n for n in stored[:-1] if n < WINDOW_TOKENS - FULL_WINDOW_SLACK]

    assert stored == recounted
    assert max(stored) <= WINDOW_TOKENS
    assert full == []  # every window but the last is filled


@BOTH
def test_a_texts_count_is_its_words_counts_added_up(
    chunks: Mapping[str, list[Chunk]], counter: FastembedTokenCounter, doc_id: str
) -> None:
    empty = counter.count("")
    added = [
        empty + sum(counter.count(word) - empty for word in c.text.split()) for c in chunks[doc_id]
    ]
    differing = sum(
        1 for total, c in zip(added, chunks[doc_id], strict=True) if total != c.token_count
    )

    assert differing == 0


@BOTH
def test_chunk_ids_are_pinned_per_document(chunks: Mapping[str, list[Chunk]], doc_id: str) -> None:
    ids = sorted(c.chunk_id for c in chunks[doc_id])
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    unique, count = len(set(ids)), len(ids)

    assert unique == count == CHUNKS[doc_id]
    assert digest == CHUNK_ID_DIGESTS[doc_id]

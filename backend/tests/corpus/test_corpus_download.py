"""The checksum pin rejects a tampered copy of a real manual (served bytes altered)."""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest

from app.domain.ingestion import ChecksumMismatchError
from app.infra.corpus_http import HttpCorpusSource
from app.ingestion.corpus import load_corpus
from tests.corpus.conftest import ENVY

pytestmark = pytest.mark.corpus


def serving(body: bytes, corpus_dir: Path) -> HttpCorpusSource:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body)),
        follow_redirects=False,
    )
    return HttpCorpusSource(
        client, load_corpus(), corpus_dir=corpus_dir, max_bytes=32 << 20, timeout_seconds=60
    )


async def test_one_flipped_byte_fails_the_checksum(
    corpus_files: Mapping[str, Path], tmp_path: Path
) -> None:
    document = next(d for d in load_corpus() if d.doc_id == ENVY)
    real = corpus_files[ENVY].read_bytes()
    tampered = bytearray(real)
    tampered[len(real) // 2] ^= 0x01

    assert (await serving(real, tmp_path / "ok").fetch(document)).read_bytes() == real
    with pytest.raises(ChecksumMismatchError, match=hashlib.sha256(tampered).hexdigest()):
        await serving(bytes(tampered), tmp_path / "bad").fetch(document)

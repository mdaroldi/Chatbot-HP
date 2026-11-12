"""Behavior of the corpus manifest: the allowlist of source documents."""

import pathlib

import pytest

from app.domain.ingestion import CorpusDocument, CorpusManifestError
from app.ingestion.corpus import CORPUS_MANIFEST, load_corpus, parse_corpus

SHA_A = "a" * 64
SHA_B = "b" * 64

# The only documents the challenge allows (CLAUDE.md), with the pinned digests.
ALLOWED = {
    "https://h10032.www1.hp.com/ctg/Manual/c06584210.pdf": (
        "d7f346897f0660b78dea88719a76267c604cbb2ee81de686ceabce56ab7a34d1"
    ),
    "https://kaas.hpcloud.hp.com/pdf-public/pdf_10301542_en-US-1.pdf": (
        "e19b003d287283dd409271f72da5ccfa28fd34247a616628a65e198e469b5356"
    ),
}


def entry(doc_id: str = "demo-guide", url: str = "https://docs.test/a.pdf", **extra: str) -> str:
    fields = {
        "doc_id": doc_id,
        "title": "Demo Guide",
        "product": "Demo Printer 100",
        "url": url,
        "sha256": SHA_A,
        **extra,
    }
    body = "\n".join(f'{key} = "{value}"' for key, value in fields.items())
    return f"[[documents]]\n{body}\n"


def test_parses_documents_in_file_order() -> None:
    text = entry() + entry("other-guide", "https://docs.test/b.pdf", sha256=SHA_B)

    documents = parse_corpus(text)

    assert documents == (
        CorpusDocument(
            doc_id="demo-guide",
            title="Demo Guide",
            product="Demo Printer 100",
            url="https://docs.test/a.pdf",
            sha256=SHA_A,
        ),
        CorpusDocument(
            doc_id="other-guide",
            title="Demo Guide",
            product="Demo Printer 100",
            url="https://docs.test/b.pdf",
            sha256=SHA_B,
        ),
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("documents = [", "not valid TOML"),
        ("", "at least one document"),
        ("documents = []", "at least one document"),
        ('documents = "x"', "at least one document"),
        ('title = "x"\n' + entry(), "unknown top-level key"),
        ("documents = [1]", "must be a table"),
        (entry().replace('sha256 = "' + SHA_A + '"\n', ""), "missing key: sha256"),
        (entry(edition="2"), "unknown key: edition"),
        (entry().replace('doc_id = "demo-guide"', "doc_id = 7"), "doc_id must be a string"),
        (entry(doc_id="Demo Guide"), "doc_id must be a lowercase slug"),
        (entry(url="http://docs.test/a.pdf"), "url must use https"),
        (entry() + entry(url="https://docs.test/b.pdf"), "duplicate doc_id: demo-guide"),
        (entry() + entry("other-guide"), "duplicate url: https://docs.test/a.pdf"),
    ],
)
def test_invalid_manifests_are_rejected(text: str, message: str) -> None:
    with pytest.raises(CorpusManifestError, match=message):
        parse_corpus(text)


def test_load_reads_the_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "corpus.toml"
    path.write_text(entry())

    assert [d.doc_id for d in load_corpus(path)] == ["demo-guide"]


def test_load_of_a_missing_file_is_a_manifest_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(CorpusManifestError, match="cannot read"):
        load_corpus(tmp_path / "absent.toml")


def test_repository_manifest_allows_exactly_the_two_hp_documents() -> None:
    documents = load_corpus(CORPUS_MANIFEST)

    assert {d.url: d.sha256 for d in documents} == ALLOWED
    assert [d.doc_id for d in documents] == ["envy-6000-user-guide", "omen-17-service-guide"]

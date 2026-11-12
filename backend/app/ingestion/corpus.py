"""The corpus manifest: the allowlist of source documents with pinned checksums."""

import tomllib
from pathlib import Path

from app.domain.ingestion import CorpusDocument, CorpusManifestError

CORPUS_MANIFEST = Path(__file__).resolve().parents[2] / "corpus.toml"

_KEYS = ("doc_id", "title", "product", "url", "sha256")


def _document(raw: object, position: int) -> CorpusDocument:
    where = f"documents[{position}]"
    if not isinstance(raw, dict):
        raise CorpusManifestError(f"{where} must be a table")
    for key in _KEYS:
        if key not in raw:
            raise CorpusManifestError(f"{where} missing key: {key}")
        if not isinstance(raw[key], str):
            raise CorpusManifestError(f"{where} {key} must be a string")
    unknown = sorted(set(raw) - set(_KEYS))
    if unknown:
        raise CorpusManifestError(f"{where} unknown key: {', '.join(unknown)}")
    try:
        return CorpusDocument(**{key: raw[key] for key in _KEYS})
    except ValueError as exc:
        raise CorpusManifestError(f"{where} {exc}") from exc


def parse_corpus(text: str) -> tuple[CorpusDocument, ...]:
    """Fail on anything unexpected: the manifest is the corpus allowlist, so a typo
    must stop ingestion instead of widening or silently shrinking the corpus.
    """
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise CorpusManifestError(f"corpus manifest is not valid TOML: {exc}") from exc
    unknown = sorted(set(data) - {"documents"})
    if unknown:
        raise CorpusManifestError(f"unknown top-level key: {', '.join(unknown)}")
    raw_documents = data.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise CorpusManifestError("corpus manifest needs at least one document")
    documents = tuple(_document(raw, n) for n, raw in enumerate(raw_documents))
    for attribute in ("doc_id", "url"):
        seen: set[str] = set()
        for document in documents:
            value = getattr(document, attribute)
            if value in seen:
                raise CorpusManifestError(f"duplicate {attribute}: {value}")
            seen.add(value)
    return documents


def load_corpus(path: Path = CORPUS_MANIFEST) -> tuple[CorpusDocument, ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CorpusManifestError(f"cannot read corpus manifest {path}") from exc
    return parse_corpus(text)

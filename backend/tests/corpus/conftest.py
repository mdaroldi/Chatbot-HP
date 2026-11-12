"""Corpus fixtures: the real HP manuals, fetched and verified once per session.

A download problem fails with a CORPUS UNAVAILABLE message, so it can never be
mistaken for a parsing regression (which fails with an assertion error).
"""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.core.settings import Settings
from app.domain.ingestion import CorpusError, ParsedDocument, TextLine
from app.infra.corpus_http import HttpCorpusSource, create_corpus_http_client
from app.infra.pdfplumber_parser import PdfplumberParser
from app.ingestion.cleaning import CleanedDocument, clean
from app.ingestion.corpus import load_corpus
from app.ingestion.footer import FooterSplit, split_footers
from app.ingestion.lines import group_lines
from app.ingestion.sections import SectionedDocument, build_sections

ENVY = "envy-6000-user-guide"
OMEN = "omen-17-service-guide"


@dataclass(frozen=True)
class ParsedCorpusDocument:
    parsed: ParsedDocument
    lines: tuple[TextLine, ...]
    footers: FooterSplit
    cleaned: CleanedDocument
    sectioned: SectionedDocument


async def _fetch_all() -> dict[str, Path]:
    # The password is never used: corpus tests touch no database.
    settings = Settings(postgres_password=SecretStr("unused-by-corpus-tests"))
    documents = load_corpus()
    async with create_corpus_http_client(settings) as client:
        source = HttpCorpusSource.from_settings(client, documents, settings)
        return {d.doc_id: await source.fetch(d) for d in documents}


@pytest.fixture(scope="session")
def corpus_files() -> Mapping[str, Path]:
    try:
        return asyncio.run(_fetch_all())
    except CorpusError as exc:
        pytest.fail(f"CORPUS UNAVAILABLE (not a parsing failure): {exc}", pytrace=False)


@pytest.fixture(scope="session")
def corpus(corpus_files: Mapping[str, Path]) -> Mapping[str, ParsedCorpusDocument]:
    parser = PdfplumberParser()
    result: dict[str, ParsedCorpusDocument] = {}
    for document in load_corpus():
        parsed = parser.parse(corpus_files[document.doc_id], document)
        lines = group_lines(parsed.words, parsed.images)
        cleaned = clean(lines, parsed.outline, parsed.page_heights, parsed.rules)
        result[document.doc_id] = ParsedCorpusDocument(
            parsed=parsed,
            lines=lines,
            footers=split_footers(lines, parsed.page_heights),
            cleaned=cleaned,
            sectioned=build_sections(document.doc_id, cleaned, parsed.outline, parsed.rules),
        )
    return result

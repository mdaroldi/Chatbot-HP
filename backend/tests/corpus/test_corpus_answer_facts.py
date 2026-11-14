"""Every key fact of the frozen Phase 5 datasets is stated in its manual (PDF text).

Key facts were written from the PDFs, never from the index; this keeps them true to
the manuals: each key-fact and warning group needs an alternative that the answer
matcher finds in the raw page text of the item's anchor documents (either manual
when the item has no anchors). Assertions compare item ids and group positions only,
never document text (see tests/corpus/conftest.py).
"""

import pathlib
from collections.abc import Mapping

import pdfplumber
import pytest

from app.core.settings import Settings
from app.domain.answer_dataset import EvalItem, KeyFactGroup
from app.evaluation.answers.dataset import load_dataset
from app.evaluation.answers.matching import alternative_matches, normalize_answer
from app.ingestion.corpus import load_corpus
from tests.corpus.conftest import Hidden

pytestmark = pytest.mark.corpus

EVAL_DIR = pathlib.Path(__file__).resolve().parents[3] / "eval"


def _page_text(path: pathlib.Path) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _groups(item: EvalItem) -> list[tuple[str, KeyFactGroup]]:
    named = [(f"key_facts[{n}]", group) for n, group in enumerate(item.key_facts)]
    if item.warning_facts is not None:
        named.append(("warning_facts", item.warning_facts))
    return named


def test_every_key_fact_group_is_in_its_manual(corpus_files: Mapping[str, pathlib.Path]) -> None:
    texts = Hidden({doc_id: normalize_answer(_page_text(p)) for doc_id, p in corpus_files.items()})
    dataset = load_dataset(
        EVAL_DIR,
        {d.doc_id for d in load_corpus()},
        max_turn_chars=Settings.model_fields["chat_message_max_chars"].default,
    )
    missing = []
    for item in dataset.items:
        doc_ids = sorted({anchor.doc_id for anchor in item.anchors}) or sorted(texts)
        text = " ".join(texts[doc_id] for doc_id in doc_ids)
        missing += [
            f"{item.id} {name}"
            for name, group in _groups(item)
            if not any(alternative_matches(alternative, text) for alternative in group)
        ]

    assert missing == []

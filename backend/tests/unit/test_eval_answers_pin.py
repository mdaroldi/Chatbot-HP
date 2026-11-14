"""The frozen Phase 5 datasets (answers-v1 and slices-v1), checked on the real files.

Every Phase 5 decision is read on these files, so they must not change without a new
version, a new pin in the freeze gate and a record in ADR 0012. The checks below also
hold the files to the code that scores them: a later change to the normalizer, the
system prompt or the refusal text that would blind a metric fails here first.
"""

import hashlib
import pathlib
import re

import pytest

from app.core.settings import Settings
from app.domain.answer_dataset import REGEX_PREFIX, Attack, Dataset, DatasetFile, Slice
from app.domain.chat import REFUSAL_TEXT
from app.domain.evaluation import Split
from app.domain.prompt import ANSWER_SYSTEM_PROMPT
from app.evaluation.answers.answer_matrix import ItemSet
from app.evaluation.answers.dataset import ANSWERS_FILE, SLICES_FILE, load_dataset
from app.evaluation.answers.freeze import FROZEN, require_frozen, run_items, select_items
from app.evaluation.answers.matching import normalize_answer
from app.evaluation.normalize import normalize
from app.evaluation.runner import HeldoutRefusedError
from app.ingestion.corpus import load_corpus

EVAL_DIR = pathlib.Path(__file__).resolve().parents[3] / "eval"
ANSWERS_PIN = DatasetFile(
    "answers-v1.toml", "1.0.0", "2aac464af1c68940cb8f844ac41ebc793fd10693f28ac37109df332225b371de"
)
SLICES_PIN = DatasetFile(
    "slices-v1.toml", "1.0.0", "b9ec7a354bb635351658ad0729a3ba9adce7b76e3071a4fc019ace5d22384e3a"
)
MAX_TURN_CHARS = Settings.model_fields["chat_message_max_chars"].default


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return load_dataset(EVAL_DIR, {d.doc_id for d in load_corpus()}, max_turn_chars=MAX_TURN_CHARS)


def _alternatives(dataset: Dataset) -> list[tuple[str, str]]:
    """(item id, alternative) for every key-fact and warning alternative."""
    pairs = []
    for item in dataset.items:
        groups = (*item.key_facts, *((item.warning_facts,) if item.warning_facts else ()))
        pairs += [(item.id, alternative) for group in groups for alternative in group]
    return pairs


def test_the_answer_datasets_are_the_frozen_files() -> None:
    for pin, path in ((ANSWERS_PIN, ANSWERS_FILE), (SLICES_PIN, SLICES_FILE)):
        assert hashlib.sha256((EVAL_DIR / path).read_bytes()).hexdigest() == pin.sha256
        assert pin in FROZEN


def test_the_frozen_files_load_with_their_recorded_versions_and_counts(dataset: Dataset) -> None:
    require_frozen(dataset)
    assert set(dataset.files) == set(FROZEN)
    dev = dataset.in_split(Split.DEV)
    assert (len(dataset.items), len(dev), len(dataset.in_split(Split.HELDOUT))) == (135, 105, 30)
    assert sum(item.primary for item in dev) == 48
    assert sum(item.slice is None for item in dev) == 56
    by_slice = {kind: sum(item.slice is kind for item in dev) for kind in Slice}
    assert by_slice == {
        Slice.MULTI_TURN: 10,
        Slice.LONG_QUERY: 8,
        Slice.OUT_OF_SCOPE: 17,
        Slice.ADVERSARIAL: 14,
    }


def test_the_real_heldout_split_needs_the_flag(dataset: Dataset) -> None:
    with pytest.raises(HeldoutRefusedError):
        select_items(dataset, Split.HELDOUT, allow_heldout=False)
    assert len(select_items(dataset, Split.DEV, allow_heldout=False)) == 105


def test_no_plain_alternative_is_changed_by_markup_stripping(dataset: Dataset) -> None:
    """The answer loses tags and emphasis before matching; a fact holding such markup
    would be matched in a form nobody wrote.
    """
    changed = [
        item_id
        for item_id, alternative in _alternatives(dataset)
        if not alternative.startswith(REGEX_PREFIX)
        and normalize_answer(alternative) != normalize(alternative)
    ]

    assert changed == []


def test_no_regex_holds_a_character_the_normalized_answer_cannot_contain(
    dataset: Dataset,
) -> None:
    """Typographic dashes and quotes, non-breaking or repeated spaces never survive
    normalization, so a pattern holding one could never match.
    """
    blind = [
        item_id
        for item_id, alternative in _alternatives(dataset)
        if alternative.startswith(REGEX_PREFIX)
        and normalize(pattern := alternative.removeprefix(REGEX_PREFIX)) != pattern
    ]

    assert blind == []


def test_prompt_leak_patterns_match_the_answer_system_prompt(dataset: Dataset) -> None:
    leaks = [item for item in dataset.items if item.attack is Attack.PROMPT_LEAK]
    blind = [
        (item.id, n)
        for item in leaks
        for n, pattern in enumerate(item.forbidden)
        if re.search(pattern, ANSWER_SYSTEM_PROMPT) is None
    ]

    assert [item.id for item in leaks] == ["adv-006", "adv-007", "adv-008", "adv-015"]
    assert blind == []


def test_no_forbidden_pattern_matches_the_fixed_refusal_text(dataset: Dataset) -> None:
    """A refusal is the expected outcome of most attacks; it must be able to pass."""
    hits = [
        (item.id, n)
        for item in dataset.items
        for n, pattern in enumerate(item.forbidden)
        if re.search(pattern, REFUSAL_TEXT)
    ]

    assert hits == []


@pytest.mark.parametrize(
    ("item_set", "count"),
    [
        (ItemSet.DEV, 105),
        (ItemSet.ANSWERS_DEV, 56),
        (ItemSet.MULTI_TURN_DEV, 10),
        (ItemSet.HELDOUT, 30),
    ],
)
def test_each_item_set_selects_its_frozen_items(
    dataset: Dataset, item_set: ItemSet, count: int
) -> None:
    items = run_items(dataset, item_set, allow_heldout=True)
    assert len(items) == count
    dev_order = [i.id for i in select_items(dataset, Split.DEV, allow_heldout=False)]
    if item_set is ItemSet.ANSWERS_DEV:
        assert all(i.slice is None and i.split is Split.DEV for i in items)
    if item_set is ItemSet.MULTI_TURN_DEV:
        assert all(i.slice is Slice.MULTI_TURN and i.split is Split.DEV for i in items)
    if item_set is not ItemSet.HELDOUT:
        assert [i.id for i in items] == [i for i in dev_order if i in {x.id for x in items}]


def test_the_heldout_item_set_needs_the_explicit_flag(dataset: Dataset) -> None:
    with pytest.raises(HeldoutRefusedError):
        run_items(dataset, ItemSet.HELDOUT, allow_heldout=False)

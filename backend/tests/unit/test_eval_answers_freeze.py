"""The freeze gate and the held-out guard of the Phase 5 datasets.

No run on a file that is not the frozen one counts as evidence, and the held-out split
is scored once, for the final configuration only, when explicitly asked for (rule 8).
"""

import dataclasses

import pytest

from app.domain.answer_dataset import Dataset, DatasetFile, DatasetNotFrozenError, EvalItem
from app.domain.evaluation import Split
from app.evaluation.answers.freeze import FROZEN, require_frozen, select_items
from app.evaluation.runner import HeldoutRefusedError

RETRIEVAL = DatasetFile("retrieval-v1.toml", "1.0.1", "a" * 64)
ANSWERS = DatasetFile("answers-v1.toml", "1.0.0", "b" * 64)
SLICES = DatasetFile("slices-v1.toml", "1.0.0", "c" * 64)
PINS = (RETRIEVAL, ANSWERS, SLICES)

DEV = EvalItem(id="dev-1", split=Split.DEV, turns=("Q?",), in_scope=True)
HELD = EvalItem(id="held-1", split=Split.HELDOUT, turns=("Q?",), in_scope=True)
DATASET = Dataset(files=PINS, items=(DEV, HELD))


def test_dev_items_are_selected_when_every_file_is_frozen() -> None:
    assert select_items(DATASET, Split.DEV, allow_heldout=False, pins=PINS) == (DEV,)


def test_the_heldout_split_is_refused_without_the_flag() -> None:
    with pytest.raises(HeldoutRefusedError):
        select_items(DATASET, Split.HELDOUT, allow_heldout=False, pins=PINS)


def test_the_heldout_split_is_selected_with_the_flag() -> None:
    assert select_items(DATASET, Split.HELDOUT, allow_heldout=True, pins=PINS) == (HELD,)


def test_the_flag_does_not_open_an_unfrozen_dataset() -> None:
    with pytest.raises(DatasetNotFrozenError):
        select_items(DATASET, Split.HELDOUT, allow_heldout=True, pins=(RETRIEVAL, ANSWERS))


@pytest.mark.parametrize(
    "pins",
    [
        (RETRIEVAL, ANSWERS),
        (RETRIEVAL, ANSWERS, dataclasses.replace(SLICES, sha256="d" * 64)),
        (RETRIEVAL, ANSWERS, dataclasses.replace(SLICES, version="1.0.1")),
        (),
    ],
    ids=["no-pin", "other-hash", "other-version", "nothing-pinned"],
)
def test_a_file_that_is_not_the_pinned_one_is_refused(pins: tuple[DatasetFile, ...]) -> None:
    with pytest.raises(DatasetNotFrozenError) as caught:
        require_frozen(DATASET, pins)

    assert "slices-v1.toml" in str(caught.value)


def test_an_exactly_pinned_dataset_passes() -> None:
    require_frozen(DATASET, PINS)


def test_selection_checks_the_freeze_on_the_dev_split_too() -> None:
    with pytest.raises(DatasetNotFrozenError):
        select_items(DATASET, Split.DEV, allow_heldout=False, pins=(RETRIEVAL,))


def test_the_retrieval_set_is_pinned_at_its_phase_2_freeze() -> None:
    """The same record as ADR 0009 and test_eval_golden_pin.py."""
    assert (
        DatasetFile(
            "retrieval-v1.toml",
            "1.0.1",
            "91e754f19cbafb80ea0e1a120f386e5ce654c97daf1e5b5f0d8102f549e70416",
        )
        in FROZEN
    )

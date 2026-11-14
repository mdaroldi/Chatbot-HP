"""The freeze gate and the held-out guard of the Phase 5 benchmark.

A run counts as evidence only on the frozen dataset files: each file's version and
SHA-256 must equal a pin below. The answers and slices files are pinned by their freeze
commit, after review; until then no run on them is evidence. The held-out split is
scored once, for the final configuration only, and only when explicitly asked for
(Phase 5 rule 8).
"""

from app.domain.answer_dataset import (
    Dataset,
    DatasetFile,
    DatasetNotFrozenError,
    EvalItem,
    Slice,
)
from app.domain.evaluation import Split
from app.evaluation.answers.answer_matrix import ItemSet
from app.evaluation.runner import HeldoutRefusedError

# retrieval-v1: the Phase 2 freeze (ADR 0009; test_eval_golden_pin.py pins the same).
# answers-v1 and slices-v1: the Phase 5 freeze, after review and the pre-freeze checks
# (key facts found in the manual text, prompt-leak patterns matching the system prompt).
# Changing a file needs a new version and its pin moved here, with a record in ADR 0012.
FROZEN: tuple[DatasetFile, ...] = (
    DatasetFile(
        "retrieval-v1.toml",
        "1.0.1",
        "91e754f19cbafb80ea0e1a120f386e5ce654c97daf1e5b5f0d8102f549e70416",
    ),
    DatasetFile(
        "answers-v1.toml",
        "1.0.0",
        "2aac464af1c68940cb8f844ac41ebc793fd10693f28ac37109df332225b371de",
    ),
    DatasetFile(
        "slices-v1.toml",
        "1.0.0",
        "b9ec7a354bb635351658ad0729a3ba9adce7b76e3071a4fc019ace5d22384e3a",
    ),
)


def require_frozen(dataset: Dataset, pins: tuple[DatasetFile, ...] = FROZEN) -> None:
    unpinned = [file.name for file in dataset.files if file not in pins]
    if unpinned:
        raise DatasetNotFrozenError(
            f"not the frozen file: {', '.join(unpinned)}; no run on it is evidence"
        )


def select_items(
    dataset: Dataset,
    split: Split,
    *,
    allow_heldout: bool,
    pins: tuple[DatasetFile, ...] = FROZEN,
) -> tuple[EvalItem, ...]:
    if split is Split.HELDOUT and not allow_heldout:
        raise HeldoutRefusedError("the held-out split needs --allow-heldout")
    require_frozen(dataset, pins)
    return dataset.in_split(split)


def run_items(dataset: Dataset, item_set: ItemSet, *, allow_heldout: bool) -> tuple[EvalItem, ...]:
    """A matrix run's items, in dataset order, through the freeze gate and the held-out
    guard: answers-dev is the answers-v1 dev questions, multi-turn-dev that dev slice.
    """
    if item_set is ItemSet.HELDOUT:
        return select_items(dataset, Split.HELDOUT, allow_heldout=allow_heldout)
    dev = select_items(dataset, Split.DEV, allow_heldout=False)
    if item_set is ItemSet.ANSWERS_DEV:
        return tuple(item for item in dev if item.slice is None)
    if item_set is ItemSet.MULTI_TURN_DEV:
        return tuple(item for item in dev if item.slice is Slice.MULTI_TURN)
    return dev

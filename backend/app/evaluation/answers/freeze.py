"""The freeze gate and the held-out guard of the Phase 5 benchmark.

A run counts as evidence only on the frozen dataset files: each file's version and
SHA-256 must equal a pin below. The answers and slices files are pinned by their freeze
commit, after review; until then no run on them is evidence. The held-out split is
scored once, for the final configuration only, and only when explicitly asked for
(Phase 5 rule 8).
"""

from app.domain.answer_dataset import Dataset, DatasetFile, DatasetNotFrozenError, EvalItem
from app.domain.evaluation import Split
from app.evaluation.runner import HeldoutRefusedError

# retrieval-v1: the Phase 2 freeze (ADR 0009; test_eval_golden_pin.py pins the same).
FROZEN: tuple[DatasetFile, ...] = (
    DatasetFile(
        "retrieval-v1.toml",
        "1.0.1",
        "91e754f19cbafb80ea0e1a120f386e5ce654c97daf1e5b5f0d8102f549e70416",
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

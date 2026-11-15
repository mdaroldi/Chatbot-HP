"""Human labels for the judge's verdicts, and how far the judge agrees with them.

The judge pass saves the exact inputs it judged (`write_inputs`), so the label sheet is
built from files alone, on the host. Twenty first-pass items are sampled with the matrix
seed, stratified by the code's key-fact pass and the judge's correctness: one item per
stratum per round, the agreement stratum (pass, correct) last, so every answer the judge
did not mark correct and every item where the two disagree is taken while the rare strata
are small. (Faithfulness does not stratify: on reference-1 the judge found every answer
faithful.) The sheet shows what the judge saw (conversation, excerpts, reference, answer)
and two blank columns whose names give the allowed values; the verdict is hidden, so the
labeller is not anchored on it. A sheet is never overwritten: labels may be in it.

Agreement is reported as k/n and Cohen's kappa for faithful and for correctness. Kappa is
undefined (None) when chance agreement is 1, as when both raters use one label only.
Files hold HP excerpt text and live in the git-ignored run directory; error messages
name files, rows, items and fields, never cell values.
"""

import csv
import os
import random
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from app.evaluation.answers.jsonl import read_objects
from app.evaluation.answers.judge import Correctness, JudgeInput, Verdict, cache_key, fact_text
from app.evaluation.answers.judge_run import JudgeRecord
from app.evaluation.answers.matching import match_key_facts

FAITHFUL_COLUMN = "faithful (yes|no)"
CORRECTNESS_COLUMN = "correctness (correct|partial|incorrect)"
SHEET_COLUMNS = (
    "item_id",
    "conversation",
    "excerpts",
    "reference",
    "answer",
    FAITHFUL_COLUMN,
    CORRECTNESS_COLUMN,
)
SAMPLE_SIZE = 20
# Key-fact pass x judge correctness in a fixed order (so the seed alone fixes the
# sample): misses first, the agreement stratum (pass, correct) last.
Stratum = tuple[bool, Correctness]
STRATA: tuple[Stratum, ...] = tuple(
    (passed, correctness)
    for passed in (False, True)
    for correctness in (Correctness.INCORRECT, Correctness.PARTIAL, Correctness.CORRECT)
)
_FAITHFUL = {"yes": True, "no": False}


class LabelError(Exception):
    """A label file cannot be used; names files, rows, items and fields only."""


@dataclass(frozen=True)
class Label:
    faithful: bool
    correctness: Correctness


@dataclass(frozen=True)
class Agreement:
    n: int
    agreed: int
    kappa: float | None  # None when chance agreement is 1

    @property
    def raw(self) -> float:
        return self.agreed / self.n


# ---------------------------------------------------------------- inputs file


class _InputLine(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    item_id: str
    turns: list[str]
    excerpts: list[str]
    passages: list[str]
    key_facts: list[list[str]]
    answer: str


def write_inputs(path: Path, inputs: Iterable[JudgeInput]) -> None:
    """Replaces the file whole (written aside, then renamed): it mirrors one judge call."""
    lines = [
        _InputLine(
            item_id=judged.item_id,
            turns=list(judged.turns),
            excerpts=list(judged.excerpts),
            passages=list(judged.passages),
            key_facts=[list(group) for group in judged.key_facts],
            answer=judged.answer,
        ).model_dump_json()
        + "\n"
        for judged in inputs
    ]
    aside = path.with_suffix(path.suffix + ".tmp")
    aside.write_text("".join(lines), encoding="utf-8")
    os.replace(aside, path)


def read_inputs(path: Path) -> dict[str, JudgeInput]:
    inputs = {}
    for number, obj in read_objects(path, LabelError):
        try:
            line = _InputLine.model_validate(obj)
        except ValidationError:
            raise LabelError(f"{path.name} line {number}: not a judge input") from None
        inputs[line.item_id] = JudgeInput(
            item_id=line.item_id,
            turns=tuple(line.turns),
            excerpts=tuple(line.excerpts),
            passages=tuple(line.passages),
            key_facts=tuple(tuple(group) for group in line.key_facts),
            answer=line.answer,
        )
    return inputs


# ---------------------------------------------------------------- sample


def label_sample(
    strata: Mapping[str, Stratum], seed: int, size: int = SAMPLE_SIZE
) -> tuple[str, ...]:
    """`strata` maps a judged item to (key-fact pass, judge correctness). Each stratum
    is shuffled with the seed, then one item is taken per stratum per round.
    """
    rng = random.Random(seed)  # noqa: S311  # nosec B311 - a reproducible sample, not a secret
    pools = []
    for stratum in STRATA:
        pool = sorted(item_id for item_id, found in strata.items() if found == stratum)
        rng.shuffle(pool)
        pools.append(pool)
    picked: list[str] = []
    while len(picked) < size and any(pools):
        for pool in pools:
            if pool and len(picked) < size:
                picked.append(pool.pop(0))
    return tuple(sorted(picked))


# ---------------------------------------------------------------- sheet


def sheet_row(judged: JudgeInput) -> dict[str, str]:
    facts = "\n".join(f"- {fact_text(group)}" for group in judged.key_facts)
    return {
        "item_id": judged.item_id,
        "conversation": "\n".join(f"[{n}] {turn}" for n, turn in enumerate(judged.turns, 1)),
        "excerpts": "\n\n".join(judged.excerpts),
        "reference": "\n\n".join([*judged.passages, f"Key facts:\n{facts}"]),
        "answer": judged.answer,
        FAITHFUL_COLUMN: "",
        CORRECTNESS_COLUMN: "",
    }


def write_sheet(path: Path, inputs: Iterable[JudgeInput]) -> None:
    try:
        handle = path.open("x", newline="", encoding="utf-8")
    except FileExistsError:
        raise LabelError(f"{path.name} exists and may hold labels; move it first") from None
    with handle:
        writer = csv.DictWriter(handle, SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(sheet_row(judged) for judged in inputs)


# ---------------------------------------------------------------- labels


def _label(row: Mapping[str, str | None], where: str) -> Label:
    faithful = _FAITHFUL.get((row[FAITHFUL_COLUMN] or "").strip().lower())
    if faithful is None:
        raise LabelError(f"{where}: faithful must be yes or no")
    try:
        correctness = Correctness((row[CORRECTNESS_COLUMN] or "").strip().lower())
    except ValueError:
        raise LabelError(f"{where}: correctness must be correct, partial or incorrect") from None
    return Label(faithful, correctness)


def read_labels(path: Path) -> dict[str, Label]:
    try:
        handle = path.open(newline="", encoding="utf-8")
    except FileNotFoundError:
        raise LabelError(f"{path.name}: no label sheet") from None
    with handle:
        reader = csv.DictReader(handle)
        if not {"item_id", FAITHFUL_COLUMN, CORRECTNESS_COLUMN} <= set(reader.fieldnames or ()):
            raise LabelError(f"{path.name}: the label columns are missing")
        labels = {}
        for number, row in enumerate(reader, start=1):
            item_id = row["item_id"]
            labels[item_id] = _label(row, f"row {number} ({item_id})")
    return labels


# ---------------------------------------------------------------- agreement


def agreement(pairs: Sequence[tuple[Hashable, Hashable]]) -> Agreement:
    """(human, judge) pairs. Kappa = (p_o - p_e) / (1 - p_e), p_e from each rater's
    label frequencies.
    """
    n = len(pairs)
    if n == 0:
        raise LabelError("no labelled items to compare")
    agreed = sum(human == judge for human, judge in pairs)
    humans = Counter(human for human, _ in pairs)
    judges = Counter(judge for _, judge in pairs)
    chance = sum(humans[label] * judges[label] for label in humans) / (n * n)
    if chance == 1:
        return Agreement(n, agreed, None)
    return Agreement(n, agreed, (agreed / n - chance) / (1 - chance))


# ---------------------------------------------------------------- glue


def _first_verdicts(records: Iterable[JudgeRecord]) -> dict[str, tuple[str, Verdict]]:
    """(cache key, verdict) of each first pass with a verdict: repeat passes and judge
    errors are not labelled.
    """
    return {
        r.item_id: (r.cache_key, r.verdict)
        for r in records
        if r.judge_pass == 1 and r.verdict is not None
    }


def sheet_items(
    records: Iterable[JudgeRecord], inputs: Mapping[str, JudgeInput], model: str, seed: int
) -> list[JudgeInput]:
    """The sample, each input checked against its verdict's cache key: the sheet must
    show exactly what the judge saw.
    """
    strata = {}
    for item_id, (key, verdict) in _first_verdicts(records).items():
        shown = inputs.get(item_id)
        if shown is None or cache_key(model, shown) != key:
            raise LabelError(f"{item_id}: the saved inputs are not what the judge saw")
        passed = match_key_facts(shown.key_facts, shown.answer).passed
        strata[item_id] = (passed, verdict.correctness)
    return [inputs[item_id] for item_id in label_sample(strata, seed)]


def compare(
    labels: Mapping[str, Label], records: Iterable[JudgeRecord]
) -> tuple[Agreement, Agreement]:
    """(faithful, correctness) agreement of the labels with the first-pass verdicts."""
    first = _first_verdicts(records)
    faithful, correctness = [], []
    for item_id, label in labels.items():
        if item_id not in first:
            raise LabelError(f"{item_id}: labelled but has no first-pass verdict")
        _, verdict = first[item_id]
        faithful.append((label.faithful, verdict.faithful))
        correctness.append((label.correctness, verdict.correctness))
    return agreement(faithful), agreement(correctness)

"""Human labels for the judge: the stratified sample, the sheet (verdict hidden), reading
the labels back strictly, and agreement (raw and Cohen's kappa) with the judge.
"""

import csv
from pathlib import Path

import pytest

from app.evaluation.answers.judge import Correctness, JudgeInput, Verdict, cache_key
from app.evaluation.answers.judge_run import JudgeRecord
from app.evaluation.answers.labels import (
    CORRECTNESS_COLUMN,
    FAITHFUL_COLUMN,
    SHEET_COLUMNS,
    Agreement,
    Label,
    LabelError,
    agreement,
    compare,
    label_sample,
    read_inputs,
    read_labels,
    read_labels_when_done,
    sheet_items,
    sheet_row,
    write_inputs,
    write_sheet,
)


def judged(item_id: str = "q-1") -> JudgeInput:
    return JudgeInput(
        item_id=item_id,
        turns=("Earlier?", "Now?"),
        excerpts=("<excerpt S1>one</excerpt>", "<excerpt S2>two</excerpt>"),
        passages=("Gold one.", "Gold two."),
        key_facts=(("3 seconds", "re:3 ?s"), ("power",)),
        answer="Hold it 3 seconds [S1].",
    )


# ---------------------------------------------------------------- inputs file


def test_written_inputs_read_back_equal(tmp_path: Path) -> None:
    path = tmp_path / "judge-inputs.jsonl"
    inputs = (judged("q-1"), judged("q-2"))
    write_inputs(path, inputs)
    assert read_inputs(path) == {"q-1": inputs[0], "q-2": inputs[1]}


def test_writing_inputs_again_replaces_the_file(tmp_path: Path) -> None:
    path = tmp_path / "judge-inputs.jsonl"
    write_inputs(path, (judged("q-1"), judged("q-2")))
    write_inputs(path, (judged("q-3"),))
    assert list(read_inputs(path)) == ["q-3"]


def test_a_bad_inputs_line_names_the_line_only(tmp_path: Path) -> None:
    path = tmp_path / "judge-inputs.jsonl"
    path.write_text('{"item_id": "q-1", "turns": "secret wording"}\n', encoding="utf-8")
    with pytest.raises(LabelError, match="line 1") as raised:
        read_inputs(path)
    assert "secret" not in str(raised.value)


# ---------------------------------------------------------------- sample

OK, PART, WRONG = Correctness.CORRECT, Correctness.PARTIAL, Correctness.INCORRECT


def strata(counts: dict[tuple[bool, Correctness], int]) -> dict[str, tuple[bool, Correctness]]:
    """Item ids start with the stratum: p/m (key facts pass/miss) + c/p/i (judge)."""
    return {
        f"{'p' if cell[0] else 'm'}{cell[1].value[0]}-{n:02d}": cell
        for cell, count in counts.items()
        for n in range(count)
    }


def by_cell(ids: tuple[str, ...]) -> dict[str, int]:
    found: dict[str, int] = {}
    for item_id in ids:
        found[item_id[:2]] = found.get(item_id[:2], 0) + 1
    return found


# The reference-1 shape: one dominant agreement cell and four rare items.
REFERENCE = {(True, OK): 63, (True, PART): 1, (False, OK): 1, (False, PART): 1, (False, WRONG): 1}


def test_every_rare_item_is_taken_and_the_rest_comes_from_the_dominant_cell() -> None:
    sample = label_sample(strata(REFERENCE), seed=7)
    assert len(sample) == 20
    assert by_cell(sample) == {"pc": 16, "pp": 1, "mc": 1, "mp": 1, "mi": 1}


def test_the_strata_are_filled_as_evenly_as_their_sizes_allow() -> None:
    sizes = {
        (True, OK): 30,
        (True, PART): 2,
        (True, WRONG): 6,
        (False, OK): 6,
        (False, PART): 6,
        (False, WRONG): 6,
    }
    found = by_cell(label_sample(strata(sizes), seed=7))
    # One item per stratum per round, misses first, the agreement cell (pass, correct)
    # last: the two-item stratum gives both, the last round stops after three strata.
    assert found == {"mi": 4, "mp": 4, "mc": 4, "pi": 3, "pp": 2, "pc": 3}


def test_the_sample_is_reproducible_from_the_seed_and_sorted() -> None:
    first = label_sample(strata(REFERENCE), seed=7)
    assert first == label_sample(strata(REFERENCE), seed=7)
    assert first == tuple(sorted(first))
    assert first != label_sample(strata(REFERENCE), seed=8)


def test_fewer_items_than_the_sample_size_are_all_taken() -> None:
    few = strata({(True, OK): 3, (False, WRONG): 2})
    assert label_sample(few, seed=7) == tuple(sorted(few))


# ---------------------------------------------------------------- sheet


def test_a_sheet_row_shows_what_the_judge_saw_and_blank_labels() -> None:
    row = sheet_row(judged())
    assert tuple(row) == SHEET_COLUMNS
    assert row["item_id"] == "q-1"
    assert row["conversation"] == "[1] Earlier?\n[2] Now?"
    assert row["excerpts"] == "<excerpt S1>one</excerpt>\n\n<excerpt S2>two</excerpt>"
    assert row["reference"] == (
        "Gold one.\n\nGold two.\n\nKey facts:\n- 3 seconds | pattern: 3 ?s\n- power"
    )
    assert row["answer"] == "Hold it 3 seconds [S1]."
    assert (row[FAITHFUL_COLUMN], row[CORRECTNESS_COLUMN]) == ("", "")


def test_the_label_columns_name_their_allowed_values() -> None:
    assert FAITHFUL_COLUMN == "faithful (yes|no)"
    assert CORRECTNESS_COLUMN == "correctness (correct|partial|incorrect)"
    assert SHEET_COLUMNS[-2:] == (FAITHFUL_COLUMN, CORRECTNESS_COLUMN)


def test_the_sheet_is_a_csv_with_one_row_per_item(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    write_sheet(path, [judged("q-1"), judged("q-2")])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["item_id"] for r in rows] == ["q-1", "q-2"]
    assert rows[0]["conversation"] == "[1] Earlier?\n[2] Now?"


def test_an_existing_sheet_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("labels in progress", encoding="utf-8")
    with pytest.raises(LabelError, match=r"labels\.csv"):
        write_sheet(path, [judged()])
    assert path.read_text(encoding="utf-8") == "labels in progress"


# ---------------------------------------------------------------- labels


def labelled(tmp_path: Path, *rows: tuple[str, str, str]) -> Path:
    path = tmp_path / "labels.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, SHEET_COLUMNS)
        writer.writeheader()
        for item_id, faithful, correctness in rows:
            row = dict.fromkeys(SHEET_COLUMNS, "x")
            row |= {"item_id": item_id, FAITHFUL_COLUMN: faithful, CORRECTNESS_COLUMN: correctness}
            writer.writerow(row)
    return path


def test_labels_read_back_ignoring_case_and_spaces(tmp_path: Path) -> None:
    path = labelled(tmp_path, ("q-1", " Yes ", "CORRECT"), ("q-2", "no", " partial"))
    assert read_labels(path) == {
        "q-1": Label(faithful=True, correctness=Correctness.CORRECT),
        "q-2": Label(faithful=False, correctness=Correctness.PARTIAL),
    }


@pytest.mark.parametrize(
    ("faithful", "correctness", "field"),
    [("", "correct", "faithful"), ("maybe", "correct", "faithful"), ("yes", "", "correctness")],
)
def test_a_missing_or_unknown_label_names_the_row_and_field(
    tmp_path: Path, faithful: str, correctness: str, field: str
) -> None:
    path = labelled(tmp_path, ("q-1", "yes", "correct"), ("q-2", faithful, correctness))
    with pytest.raises(LabelError, match=f"row 2 .q-2.: {field}"):
        read_labels(path)


def test_a_sheet_without_the_label_columns_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("item_id,faithful,correctness\nq-1,yes,correct\n", encoding="utf-8")
    with pytest.raises(LabelError, match="columns"):
        read_labels(path)


def test_a_missing_sheet_is_an_error_naming_it(tmp_path: Path) -> None:
    with pytest.raises(LabelError, match=r"labels\.csv"):
        read_labels(tmp_path / "labels.csv")


def test_a_sheet_not_yet_labelled_reads_as_none(tmp_path: Path) -> None:
    assert read_labels_when_done(tmp_path / "labels.csv") is None
    blank = labelled(tmp_path, ("q-1", "", ""), ("q-2", " ", ""))
    assert read_labels_when_done(blank) is None


def test_a_labelled_sheet_reads_back_and_a_half_labelled_one_is_an_error(
    tmp_path: Path,
) -> None:
    path = labelled(tmp_path, ("q-1", "yes", "correct"))
    assert read_labels_when_done(path) == {"q-1": Label(True, Correctness.CORRECT)}
    labelled(tmp_path, ("q-1", "yes", "correct"), ("q-2", "", ""))
    with pytest.raises(LabelError, match="row 2"):
        read_labels_when_done(path)


# ---------------------------------------------------------------- agreement


def test_perfect_agreement_has_kappa_one() -> None:
    pairs = [(True, True), (False, False), (True, True), (False, False)]
    assert agreement(pairs) == Agreement(n=4, agreed=4, kappa=1.0)


def test_kappa_corrects_raw_agreement_for_chance() -> None:
    # 10 pairs: both yes 6, both no 1, human yes / judge no 2, human no / judge yes 1.
    pairs = [("y", "y")] * 6 + [("n", "n")] + [("y", "n")] * 2 + [("n", "y")]
    result = agreement(pairs)
    assert (result.n, result.agreed) == (10, 7)
    # p_o = 0.7; p_e = 0.8 * 0.7 + 0.2 * 0.3 = 0.62; kappa = 0.08 / 0.38
    assert result.kappa == pytest.approx(0.08 / 0.38)
    assert result.raw == pytest.approx(0.7)


def test_kappa_is_undefined_when_both_raters_use_one_label() -> None:
    assert agreement([("y", "y"), ("y", "y")]) == Agreement(n=2, agreed=2, kappa=None)


def test_agreement_needs_at_least_one_pair() -> None:
    with pytest.raises(LabelError, match="no labelled"):
        agreement([])


# ---------------------------------------------------------------- glue

MODEL = "claude-opus-5-5"


def verdict(*, faithful: bool, correctness: str = "correct") -> Verdict:
    support = "supported" if faithful else "unsupported"
    return Verdict.model_validate(
        {
            "claims": [{"claim": "c", "support": support, "tags": []}],
            "correctness": correctness,
            "reasons": ["none"],
        }
    )


def record(
    judged: JudgeInput, *, judge_pass: int = 1, faithful: bool = True, **kw: str
) -> JudgeRecord:
    return JudgeRecord(
        judged.item_id,
        judge_pass,
        cache_key(MODEL, judged),
        verdict(faithful=faithful, **kw),
        None,
        cached=False,
    )


def with_answer(item_id: str, answer: str) -> JudgeInput:
    base = judged(item_id)
    return JudgeInput(
        base.item_id, base.turns, base.excerpts, base.passages, base.key_facts, answer
    )


def test_sheet_items_stratify_first_pass_verdicts_by_key_fact_pass_and_correctness() -> None:
    passing = "Hold it 3 seconds, power."  # both key-fact groups match
    counts = {"pc": 30, "pp": 3, "mc": 3, "mi": 3}
    inputs = {
        f"{kind}-{n}": with_answer(f"{kind}-{n}", passing if kind[0] == "p" else "Nothing.")
        for kind, count in counts.items()
        for n in range(count)
    }
    judged_as = {"c": "correct", "p": "partial", "i": "incorrect"}
    # Faithfulness no longer stratifies: half the agreement cell is unfaithful, every
    # rare item faithful, and still every rare item is taken.
    records = [
        record(j, faithful=item_id[:2] != "pc" or n % 2 == 0, correctness=judged_as[item_id[1]])
        for n, (item_id, j) in enumerate(inputs.items())
    ]
    records += [record(inputs["pc-0"], judge_pass=2)]  # repeat passes are not labelled
    records += [JudgeRecord("err-1", 1, "k", None, "refusal", cached=False)]  # judge errors

    chosen = sheet_items(records, inputs, MODEL, seed=7)

    assert by_cell(tuple(j.item_id for j in chosen)) == {"pc": 11, "pp": 3, "mc": 3, "mi": 3}


def test_sheet_items_refuse_inputs_that_are_not_what_the_judge_saw() -> None:
    inputs = {"q-1": judged("q-1")}
    stale = record(with_answer("q-1", "another answer"))
    with pytest.raises(LabelError, match="q-1"):
        sheet_items([stale], inputs, MODEL, seed=7)


def test_sheet_items_refuse_a_verdict_without_its_inputs() -> None:
    with pytest.raises(LabelError, match="q-1"):
        sheet_items([record(judged("q-1"))], {}, MODEL, seed=7)


def test_compare_pairs_each_label_with_the_first_pass_verdict() -> None:
    records = [
        record(judged("q-1"), faithful=True, correctness="correct"),
        record(judged("q-2"), faithful=False, correctness="partial"),
        record(judged("q-2"), judge_pass=2, faithful=True, correctness="correct"),
    ]
    labels = {
        "q-1": Label(faithful=True, correctness=Correctness.CORRECT),
        "q-2": Label(faithful=True, correctness=Correctness.PARTIAL),
    }

    faithful, correctness = compare(labels, records)

    assert (faithful.n, faithful.agreed) == (2, 1)
    assert (correctness.n, correctness.agreed) == (2, 2)


def test_compare_refuses_a_label_without_a_first_pass_verdict() -> None:
    labels = {"q-9": Label(faithful=True, correctness=Correctness.CORRECT)}
    with pytest.raises(LabelError, match="q-9"):
        compare(labels, [record(judged("q-1"))])

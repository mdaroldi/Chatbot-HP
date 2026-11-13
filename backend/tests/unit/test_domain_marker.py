"""The refusal marker detector: never emit a character that could still be the marker."""

import pytest

from app.domain.chat import REFUSAL_MARKER as M
from app.domain.marker import MarkerDetector


def run(deltas: list[str]) -> tuple[list[str], MarkerDetector]:
    """Feed every delta, then finish; the emitted pieces end with what finish gave."""
    detector = MarkerDetector()
    emitted = [detector.feed(delta) for delta in deltas]
    return [*emitted, detector.finish()], detector


def splits(text: str) -> list[list[str]]:
    """The text split in two at every position."""
    return [[text[:k], text[k:]] for k in range(1, len(text))]


# --- Refusal: the marker leads the answer ---------------------------------------------


def test_a_whole_marker_is_a_refusal_and_nothing_is_emitted() -> None:
    emitted, detector = run([M])

    assert detector.refused
    assert not detector.late_marker
    assert "".join(emitted) == ""


@pytest.mark.parametrize("deltas", splits(M))
def test_a_marker_split_at_any_position_is_a_refusal(deltas: list[str]) -> None:
    emitted, detector = run(deltas)

    assert detector.refused
    assert emitted == ["", "", ""]


def test_a_marker_sent_one_character_at_a_time_is_a_refusal() -> None:
    emitted, detector = run(list(M))

    assert detector.refused
    assert set(emitted) == {""}


@pytest.mark.parametrize("lead", [" ", "\n", "  \n\t ", "\n\n"])
def test_whitespace_before_the_marker_does_not_hide_it(lead: str) -> None:
    emitted, detector = run([lead, *list(M)])

    assert detector.refused
    assert set(emitted) == {""}


def test_text_after_a_leading_marker_is_still_a_refusal_and_never_emitted() -> None:
    emitted, detector = run([M + " Sorry, the manuals do not say."])

    assert detector.refused
    assert "".join(emitted) == ""


def test_the_detector_must_not_be_fed_after_a_refusal() -> None:
    detector = MarkerDetector()
    detector.feed(M)

    with pytest.raises(RuntimeError, match="the answer was refused"):
        detector.feed("more")


# --- Answers: text passes through, minus any marker -----------------------------------


def test_plain_text_passes_through_as_it_arrives() -> None:
    emitted, detector = run(["Open ", "the cover", " [S1]."])

    assert emitted == ["Open ", "the cover", " [S1].", ""]
    assert not detector.refused
    assert not detector.late_marker


def test_leading_whitespace_is_stripped_from_the_answer() -> None:
    emitted, _ = run([" \n", " Open it."])

    assert emitted == ["", "Open it.", ""]


@pytest.mark.parametrize("deltas", splits("Open it. " + M))
def test_a_marker_after_text_is_removed_and_flagged_wherever_it_splits(
    deltas: list[str],
) -> None:
    emitted, detector = run(deltas)

    assert "".join(emitted) == "Open it. "
    assert not detector.refused
    assert detector.late_marker
    assert all("[" not in piece for piece in emitted)


def test_two_markers_after_text_are_both_removed() -> None:
    emitted, detector = run(["A", M, "B", M[:5], M[5:], "C"])

    assert "".join(emitted) == "ABC"
    assert detector.late_marker


def test_a_marker_rebuilt_by_removing_another_is_removed_too() -> None:
    emitted, detector = run(["A " + M[:7] + M + M[7:]])

    assert "".join(emitted) == "A "
    assert detector.late_marker


@pytest.mark.parametrize("prefix", ["[", "[[", "[[NOT", M[:-1]])
def test_a_near_miss_at_the_start_is_held_then_emitted_unchanged(prefix: str) -> None:
    emitted, detector = run([prefix, " is not the marker."])

    assert emitted == ["", prefix + " is not the marker.", ""]
    assert not detector.refused
    assert not detector.late_marker


def test_a_near_miss_after_text_is_held_only_while_it_could_be_the_marker() -> None:
    emitted, _ = run(["See [[NOT", "ES] here."])

    assert emitted == ["See ", "[[NOTES] here.", ""]


def test_a_possible_marker_start_at_the_end_is_emitted_by_finish() -> None:
    emitted, detector = run(["See [", "["])

    assert emitted == ["See ", "", "[["]
    assert not detector.late_marker


def test_an_answer_that_is_only_a_marker_prefix_is_emitted_by_finish() -> None:
    emitted, detector = run(["  [[NOT"])

    assert emitted == ["", "[[NOT"]
    assert not detector.refused


def test_an_empty_or_blank_answer_emits_nothing() -> None:
    assert run([])[0] == [""]
    assert run([" \n"])[0] == ["", ""]


def test_the_earliest_possible_marker_start_is_held_back() -> None:
    emitted, _ = run(["x [[ [[NOT"])

    assert emitted == ["x [[ ", "[[NOT"]

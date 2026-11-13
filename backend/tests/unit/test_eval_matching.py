"""Anchor matching: when a chunk satisfies an anchor of the golden set."""

from app.domain.evaluation import Anchor, IndexedChunk
from app.evaluation.matching import anchor_matches, matching_chunks, satisfied_anchors


def chunk(
    chunk_id: str = "c1",
    doc_id: str = "omen-guide",
    path: tuple[str, ...] = ("Removal and replacement", "Bottom cover"),
    text: str = "OMEN Guide > Removal and replacement > Bottom cover\n\nSpare part N96208-001",
) -> IndexedChunk:
    return IndexedChunk(chunk_id=chunk_id, doc_id=doc_id, section_path=path, text=text)


def test_a_chunk_with_every_fact_of_the_document_matches() -> None:
    assert anchor_matches(Anchor("omen-guide", ("N96208-001", "spare part")), chunk())


def test_one_missing_fact_means_no_match() -> None:
    assert not anchor_matches(Anchor("omen-guide", ("N96208-001", "keyboard")), chunk())


def test_another_document_never_matches() -> None:
    assert not anchor_matches(Anchor("envy-guide", ("N96208-001",)), chunk())


def test_facts_match_case_dash_and_space_variants() -> None:
    variant = chunk(text="Spare\u00a0Part  N96208\u2013001")

    assert anchor_matches(Anchor("omen-guide", ("spare part n96208-001",)), variant)


def test_facts_match_the_breadcrumb_prefix_too() -> None:
    assert anchor_matches(
        Anchor("omen-guide", ("bottom cover",)), chunk(text="OMEN > Bottom cover")
    )


def test_a_section_fragment_must_occur_in_the_section_path() -> None:
    anchor = Anchor("omen-guide", ("N96208-001",), section="bottom COVER")

    assert anchor_matches(anchor, chunk())
    assert anchor_matches(
        Anchor("omen-guide", ("N96208-001",), section="replacement > bottom"), chunk()
    )
    assert not anchor_matches(Anchor("omen-guide", ("N96208-001",), section="Keyboard"), chunk())


def test_the_section_is_not_searched_in_the_text() -> None:
    # The fragment appears in the text, but the chunk's path is the parent section.
    merged = chunk(path=("Removal and replacement",), text="Bottom cover\nN96208-001")

    assert not anchor_matches(Anchor("omen-guide", ("N96208-001",), section="Bottom cover"), merged)


def test_matching_chunks_lists_every_chunk_that_satisfies_the_anchor_in_order() -> None:
    chunks = [chunk("a"), chunk("b", text="nothing here"), chunk("c")]

    assert matching_chunks(Anchor("omen-guide", ("N96208-001",)), chunks) == ("a", "c")
    assert matching_chunks(Anchor("omen-guide", ("missing",)), chunks) == ()


def test_satisfied_anchors_names_anchor_positions() -> None:
    anchors = (
        Anchor("omen-guide", ("keyboard",)),
        Anchor("omen-guide", ("N96208-001",)),
        Anchor("omen-guide", ("Bottom cover",)),
    )

    assert satisfied_anchors(anchors, chunk()) == frozenset({1, 2})
    assert satisfied_anchors((), chunk()) == frozenset()

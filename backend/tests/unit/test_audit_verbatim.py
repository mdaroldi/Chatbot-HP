"""The 12-word audit: no run of 12 consecutive manual words in tracked files or history.

Synthetic text only: the "manual" here is made-up words, never HP text.
"""

import os
import subprocess  # nosec B404 - fixed git commands in a temporary repository
from pathlib import Path

import pytest

from app.audit import verbatim
from app.audit.verbatim import Hit, find_runs, manual_index, summary, words
from tests.pdf_builder import Page, Text, build_pdf

MANUAL = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray"
)
TWELVE = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"
ELEVEN = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo"
THIRTEEN = TWELVE + " mike"


@pytest.fixture
def index() -> frozenset[str]:
    return manual_index([MANUAL])


def test_words_fold_case_punctuation_and_compatibility_forms() -> None:
    assert words("Remove the ﬁlter — then PRESS “OK”, (twice).") == [
        "remove",
        "the",
        "filter",
        "then",
        "press",
        "ok",
        "twice",
    ]


def test_eleven_words_are_not_a_hit_twelve_and_thirteen_are(index: frozenset[str]) -> None:
    assert find_runs(index, ELEVEN, "a.md") == []
    assert find_runs(index, TWELVE, "a.md") == [Hit(source="a.md", line=1)]
    assert find_runs(index, THIRTEEN, "a.md") == [Hit(source="a.md", line=1)]


def test_a_run_is_found_through_punctuation_case_and_line_breaks(index: frozenset[str]) -> None:
    text = (
        "intro line\nAlpha, BRAVO; charlie delta\necho foxtrot golf hotel india juliet kilo lima.\n"
    )

    assert find_runs(index, text, "notes.md") == [Hit(source="notes.md", line=2)]


def test_one_hit_per_line_however_many_windows_start_there(index: frozenset[str]) -> None:
    assert find_runs(index, MANUAL, "copy.md") == [Hit(source="copy.md", line=1)]


def test_hits_on_separate_lines_are_reported_in_order(index: frozenset[str]) -> None:
    text = f"{TWELVE}\nunrelated words here\n{TWELVE}\n"

    assert find_runs(index, text, "x") == [Hit("x", 1), Hit("x", 3)]


def test_text_shorter_than_the_window_has_no_hit(index: frozenset[str]) -> None:
    assert find_runs(index, "alpha bravo", "x") == []
    assert manual_index(["too short"]) == frozenset()


def test_the_window_size_is_twelve() -> None:
    assert verbatim.WINDOW == 12


def test_the_summary_gives_counts_and_locations_never_text() -> None:
    hits = [Hit("README.md", 7), Hit("commit 0123456789ab", 2)]

    text = summary(hits, files=10, messages=4)

    assert text.splitlines() == [
        "scanned 10 text(s) and 4 commit message(s): 2 hit(s)",
        "README.md:7",
        "commit 0123456789ab:2",
    ]
    assert "alpha" not in text


def test_the_summary_of_a_clean_scan() -> None:
    assert summary([], files=3, messages=1) == "scanned 3 text(s) and 1 commit message(s): 0 hit(s)"


# ------------------------------------------------------------------ PDF and git input


def test_pdf_text_reads_every_page(tmp_path: Path) -> None:
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(
        build_pdf(
            (
                Page(texts=(Text(x=72, top=72, size=11, text="alpha bravo charlie"),)),
                Page(texts=(Text(x=72, top=72, size=11, text="delta echo foxtrot"),)),
            )
        )
    )

    assert words(verbatim.pdf_text(pdf)) == [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
    ]


def git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    done = subprocess.run(  # noqa: S603 # nosec B603 B607 - fixed git argv in tmp_path
        ["git", *args],  # noqa: S607 - git from PATH, like the audit itself
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    (root / "clean.md").write_text("nothing from any manual\n")
    (root / "image.png").write_bytes(b"\x89PNG\x00\x01" + TWELVE.encode())
    (root / "doc.pdf").write_bytes(TWELVE.encode())
    (root / "latin1.txt").write_bytes(b"caf\xe9 " + TWELVE.encode())
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "first", "-m", f"body quoting {TWELVE}")
    (root / "plan.md").write_text(f"draft\n{TWELVE}\n")
    git(root, "add", "plan.md")
    git(root, "commit", "-q", "-m", "plan with a run")
    (root / "plan.md").write_text("draft, cleaned\n")
    git(root, "commit", "-q", "-am", "clean the plan")
    git(root, "mv", "plan.md", "notes.md")
    git(root, "commit", "-q", "-m", "rename")
    return root


def test_tracked_texts_skip_binaries_pdfs_and_undecodable_files(repo: Path) -> None:
    names = [name for name, _ in verbatim.tracked_texts(repo)]

    assert names == ["clean.md", "notes.md"]


def test_commit_messages_cover_every_commit_with_its_short_sha(repo: Path) -> None:
    messages = list(verbatim.commit_messages(repo))

    assert len(messages) == 4
    assert all(
        name.startswith("commit ") and len(name) == len("commit ") + 12 for name, _ in messages
    )
    assert any(TWELVE in text for _, text in messages)


def test_path_history_returns_every_version_across_renames(repo: Path) -> None:
    versions = list(verbatim.path_history(repo, "notes.md"))

    assert len(versions) == 3
    assert [name.split(":", 1)[1] for name, _ in versions] == ["notes.md", "plan.md", "plan.md"]
    assert any(TWELVE in text for _, text in versions)


def test_path_history_skips_binary_versions(repo: Path) -> None:
    assert list(verbatim.path_history(repo, "image.png")) == []


def test_path_history_of_an_unknown_path_is_an_error(repo: Path) -> None:
    with pytest.raises(LookupError, match="no history"):
        list(verbatim.path_history(repo, "missing.md"))


# ------------------------------------------------------------------ CLI


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Two synthetic manuals under the corpus's real document IDs."""
    directory = tmp_path / "corpus"
    directory.mkdir()
    halves = (MANUAL[: len(MANUAL) // 2], MANUAL[len(MANUAL) // 2 :])
    for doc_id, text in zip(verbatim.manual_ids(), halves, strict=True):
        (directory / f"{doc_id}.pdf").write_bytes(b"%PDF")
        monkeypatch.setitem(TEXTS, str(directory / f"{doc_id}.pdf"), text)
    monkeypatch.setattr(verbatim, "pdf_text", lambda path: TEXTS[str(path)])
    return directory


TEXTS: dict[str, str] = {}


def test_the_corpus_has_the_two_manuals() -> None:
    assert verbatim.manual_ids() == ("envy-6000-user-guide", "omen-17-service-guide")


def test_main_reports_hits_in_files_and_messages_and_exits_one(
    repo: Path, corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "clean.md").write_text(f"now with {TWELVE}\n")
    git(repo, "commit", "-q", "-am", "add a run")

    code = verbatim.main(["--repo", str(repo), "--corpus-dir", str(corpus)])

    out = capsys.readouterr().out
    assert code == 1
    assert out.splitlines()[0] == "scanned 2 text(s) and 5 commit message(s): 2 hit(s)"
    assert "clean.md:1" in out
    assert "alpha" not in out


def test_main_is_clean_and_exits_zero_when_nothing_matches(
    repo: Path, corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = verbatim.main(
        ["--repo", str(repo), "--corpus-dir", str(corpus), "--history", "clean.md"]
    )

    assert code == 0
    assert capsys.readouterr().out.startswith("scanned 1 text(s) and 0 commit message(s): 0 hit(s)")


def test_main_scans_every_historical_version_of_a_path(
    repo: Path, corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = verbatim.main(
        ["--repo", str(repo), "--corpus-dir", str(corpus), "--history", "notes.md"]
    )

    out = capsys.readouterr().out
    assert code == 1
    assert out.splitlines()[0] == "scanned 3 text(s) and 0 commit message(s): 1 hit(s)"
    assert out.splitlines()[1].endswith(":plan.md:2")


def test_main_fails_loudly_when_a_manual_is_missing(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = verbatim.main(["--repo", str(repo), "--corpus-dir", str(tmp_path / "empty")])

    assert code == 2
    assert "make fetch-test-assets" in capsys.readouterr().err

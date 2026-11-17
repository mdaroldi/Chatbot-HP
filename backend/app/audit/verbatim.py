"""The 12-word audit: no run of 12 consecutive manual words in the repository.

The HP manuals are copyrighted and must never be committed, nor long extracts of
them (CLAUDE.md). This audit extracts the words of both verified manuals, and reports
every place in a tracked text file, a commit message (every ref) or a past version of
a given path where 12 consecutive words match the manuals. Words are compared after
Unicode compatibility folding, case folding and dropping punctuation, so a copy with
different quotes, ligatures, case or line breaks still matches.

The output names locations and counts only (`file:line`), never the matched text:
it runs in CI, whose logs are public. Run it with `make audit-verbatim` after
`make fetch-test-assets`.
"""

import argparse
import re
import subprocess  # nosec B404 - fixed git commands, no shell
import sys
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from app.ingestion.corpus import load_corpus

WINDOW = 12
_WORD = re.compile(r"[a-z0-9]+")
_SKIPPED_SUFFIXES = (".pdf",)


@dataclass(frozen=True)
class Hit:
    source: str
    line: int


def words(text: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def _windows(tokens: Sequence[str], n: int) -> Iterator[str]:
    for start in range(len(tokens) - n + 1):
        yield " ".join(tokens[start : start + n])


def manual_index(texts: Iterable[str], n: int = WINDOW) -> frozenset[str]:
    """Every n-word window of the manuals, as space-joined folded words."""
    return frozenset(window for text in texts for window in _windows(words(text), n))


def find_runs(index: frozenset[str], text: str, source: str, n: int = WINDOW) -> list[Hit]:
    """One hit per line on which a matching run starts, in line order."""
    tokens: list[str] = []
    lines: list[int] = []
    for number, line in enumerate(text.splitlines(), start=1):
        found = words(line)
        tokens.extend(found)
        lines.extend([number] * len(found))
    hit_lines = sorted(
        {lines[start] for start, window in enumerate(_windows(tokens, n)) if window in index}
    )
    return [Hit(source=source, line=line) for line in hit_lines]


def summary(hits: Sequence[Hit], *, files: int, messages: int) -> str:
    head = f"scanned {files} text(s) and {messages} commit message(s): {len(hits)} hit(s)"
    return "\n".join([head, *(f"{hit.source}:{hit.line}" for hit in hits)])


def pdf_text(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def manual_ids() -> tuple[str, ...]:
    return tuple(document.doc_id for document in load_corpus())


def _git(repo: Path, *args: str) -> bytes:
    done = subprocess.run(  # noqa: S603 - fixed git argv, no shell  # nosec B603 B607
        ["git", *args],  # noqa: S607 - git from PATH, like make and CI
        cwd=repo,
        capture_output=True,
        check=True,
    )
    return done.stdout


def _decoded(raw: bytes) -> str | None:
    """Text, or None for a binary (a NUL byte) or a file that is not UTF-8."""
    if b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def tracked_texts(repo: Path) -> Iterator[tuple[str, str]]:
    """Each tracked text file of the working tree, sorted by path; PDFs are skipped."""
    names = sorted(name for name in _git(repo, "ls-files", "-z").decode().split("\0") if name)
    for name in names:
        if name.casefold().endswith(_SKIPPED_SUFFIXES):
            continue
        text = _decoded((repo / name).read_bytes())
        if text is not None:
            yield name, text


def commit_messages(repo: Path) -> Iterator[tuple[str, str]]:
    """The message of every commit reachable from any ref."""
    raw = _git(repo, "log", "--all", "--format=%H%x00%B%x1e").decode()
    for record in raw.split("\x1e"):
        sha, separator, message = record.strip("\n").partition("\0")
        if separator:
            yield f"commit {sha[:12]}", message


def path_history(repo: Path, path: str) -> Iterator[tuple[str, str]]:
    """Every committed version of a path, following renames, as `<sha12>:<name>`."""
    raw = _git(repo, "log", "--all", "--follow", "--format=%x1e%H", "--name-only", "--", path)
    versions = []
    for record in raw.decode().split("\x1e"):
        fields = record.split()
        if len(fields) == 2:
            versions.append((fields[0], fields[1]))
    if not versions:
        raise LookupError(f"no history for {path}")
    for sha, name in versions:
        if name.casefold().endswith(_SKIPPED_SUFFIXES):
            continue
        text = _decoded(_git(repo, "show", f"{sha}:{name}"))
        if text is not None:
            yield f"{sha[:12]}:{name}", text


def _manual_texts(corpus_dir: Path) -> list[str] | None:
    paths = [corpus_dir / f"{doc_id}.pdf" for doc_id in manual_ids()]
    if not all(path.is_file() for path in paths):
        return None
    return [pdf_text(path) for path in paths]


def main(argv: Sequence[str] | None = None) -> int:
    """Exit 0 when clean, 1 on any hit, 2 when a manual is missing."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"))
    parser.add_argument(
        "--history",
        action="append",
        default=[],
        help="scan every past version of this path instead of the tree and messages",
    )
    args = parser.parse_args(argv)

    texts = _manual_texts(args.corpus_dir)
    if texts is None:
        print(  # noqa: T201 - CLI output
            f"manuals not found in {args.corpus_dir}: run make fetch-test-assets first",
            file=sys.stderr,
        )
        return 2
    index = manual_index(texts)

    if args.history:
        sources = [item for path in args.history for item in path_history(args.repo, path)]
        messages: list[tuple[str, str]] = []
    else:
        sources = list(tracked_texts(args.repo))
        messages = list(commit_messages(args.repo))
    hits = [hit for name, text in [*sources, *messages] for hit in find_runs(index, text, name)]
    print(summary(hits, files=len(sources), messages=len(messages)))  # noqa: T201 - CLI output
    return 1 if hits else 0

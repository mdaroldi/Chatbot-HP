"""Key-fact and safety-note matching on answers (Phase 5 matching rules, fixed).

The answer is normalized before matching: citation tags (`[S1]`, `[S1, S3]`) and
Markdown emphasis markers and backticks are removed, then the text goes through the
same normalization as anchor matching (dashes, quotes and non-breaking spaces unified,
case-folded, whitespace collapsed). A plain alternative is normalized the same way and
matched as a substring; a `re:` alternative is searched in the normalized answer. A
group passes when any alternative matches; an answer passes when every group passes.

A safety note is carried when a line, after its emphasis markers and a leading list
marker are removed, starts with `WARNING:` or `CAUTION:` in upper case (checked before
case folding), and that same line matches the warning group.
"""

import re
from dataclasses import dataclass

from app.domain.answer_dataset import REGEX_PREFIX, KeyFactGroup
from app.evaluation.normalize import normalize

_CITATION = re.compile(r"\s*\[\s*S\d+(?:\s*,\s*S\d+)*\s*\]")
# Asterisks and backticks always; underscores only as delimiters, so hp_smart stays.
_EMPHASIS = re.compile(r"[*`]|(?<!\w)_+|_+(?!\w)")
_LIST_MARKER = re.compile(r"^\s*(?:[-+•]|\d+[.)])\s+")
_WARNING_LABELS = ("WARNING:", "CAUTION:")


@dataclass(frozen=True)
class KeyFactResult:
    groups: tuple[bool, ...]  # one verdict per key-fact group, in file order

    @property
    def passed(self) -> bool:
        return all(self.groups)

    @property
    def coverage(self) -> float | None:
        """Share of groups passed; None without groups (out of scope)."""
        return sum(self.groups) / len(self.groups) if self.groups else None


def _strip_markup(text: str) -> str:
    return _EMPHASIS.sub("", _CITATION.sub("", text))


def normalize_answer(text: str) -> str:
    return normalize(_strip_markup(text))


def alternative_matches(alternative: str, normalized: str) -> bool:
    """`normalized` is an answer (or line) already passed through normalize_answer."""
    if alternative.startswith(REGEX_PREFIX):
        return re.search(alternative.removeprefix(REGEX_PREFIX), normalized) is not None
    return normalize_answer(alternative) in normalized


def _group_passes(group: KeyFactGroup, normalized: str) -> bool:
    return any(alternative_matches(alternative, normalized) for alternative in group)


def match_key_facts(groups: tuple[KeyFactGroup, ...], answer: str) -> KeyFactResult:
    normalized = normalize_answer(answer)
    return KeyFactResult(tuple(_group_passes(group, normalized) for group in groups))


def warning_lines(answer: str) -> tuple[str, ...]:
    """The answer's lines that open with an upper-case WARNING: or CAUTION: label."""

    def labelled(line: str) -> bool:
        bare = _LIST_MARKER.sub("", _EMPHASIS.sub("", line), count=1).lstrip()
        return bare.startswith(_WARNING_LABELS)

    return tuple(line for line in answer.splitlines() if labelled(line))


def warning_carried(answer: str, warning_facts: KeyFactGroup) -> bool:
    return any(
        _group_passes(warning_facts, normalize_answer(line)) for line in warning_lines(answer)
    )

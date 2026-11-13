"""Citation tags in answers (pure): validate them, strip them, quote their sources.

The prompt asks for one bracket per tag (`[S1][S3]`), but models also write groups
(`[S1, S3]`), so one pattern accepts both. Validation keeps only tags of excerpts the
model was actually given; stripping removes every tag, for prompt history where old
tags would point at excerpts the model can no longer see.
"""

import re
from collections.abc import Collection
from dataclasses import dataclass

# The horizontal whitespace before a tag goes with it, so "Do it [S9]." becomes
# "Do it." when the tag is removed; a line break before a tag is kept.
CITATION = re.compile(r"(?P<lead>[ \t]*)\[\s*(?P<tags>S\d+(?:\s*,\s*S\d+)*)\s*\]")

_TAG_SEPARATOR = re.compile(r"\s*,\s*")
_BREADCRUMB_END = "\n\n"
_ELLIPSIS = "…"


@dataclass(frozen=True)
class Citations:
    """The answer with unknown tags removed, the tags it cites (each once, by number)
    and how many tag occurrences were removed.
    """

    text: str
    tags: tuple[str, ...]
    invalid: int


def strip_citations(text: str) -> str:
    return CITATION.sub("", text)


def _number(tag: str) -> int:
    return int(tag[1:])


def validate_citations(text: str, valid_tags: Collection[str]) -> Citations:
    cited: set[str] = set()
    invalid = 0

    def keep_valid(match: re.Match[str]) -> str:
        nonlocal invalid
        members = _TAG_SEPARATOR.split(match["tags"])
        kept = [tag for tag in members if tag in valid_tags]
        cited.update(kept)
        invalid += len(members) - len(kept)
        if len(kept) == len(members):
            return match[0]
        if not kept:
            return ""
        return f"{match['lead']}[{', '.join(kept)}]"

    checked = CITATION.sub(keep_valid, text)
    return Citations(checked, tuple(sorted(cited, key=_number)), invalid)


def snippet(text: str, max_chars: int) -> str:
    """The start of a chunk's body for a source card: the breadcrumb line is left out
    (the card shows the section already), and a cut falls on a word boundary.
    """
    _, separator, body = text.partition(_BREADCRUMB_END)
    words = " ".join((body if separator else text).split())
    if len(words) <= max_chars:
        return words
    head = words[:max_chars]
    if words[max_chars] != " " and " " in head:
        head = head.rsplit(" ", 1)[0]
    return head + _ELLIPSIS

"""Short display text (pure): one line, cut on a word boundary."""

ELLIPSIS = "…"


def shorten(text: str, max_chars: int) -> str:
    """`text` on one line with whitespace collapsed; when longer than `max_chars`, cut at
    the last word that fits (or mid-word for a single long word) and end with "…".
    """
    words = " ".join(text.split())
    if len(words) <= max_chars:
        return words
    head = words[:max_chars]
    if words[max_chars] != " " and " " in head:
        head = head.rsplit(" ", 1)[0]
    return head + ELLIPSIS

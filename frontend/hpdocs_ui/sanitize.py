"""Make answer Markdown safe to render: no images, no live links, no HTML, no math.

The answer is model output shaped by retrieved text and the user's question, so it is
untrusted. Streamlit's markdown leaves raw HTML unrendered unless a call opts in, and
no call here does; that is the primary guard. This is defence in depth against what
it still renders: images (requests to any host), links (a clickable phishing target)
and math (ADR 0011).

Code is left byte for byte, because it renders literally, but only where the
sanitizer's reading of code provably matches a CommonMark renderer's:

- a fenced block counts as code only when every fence-like line in the text starts
  at column 0; then no list or quote can own a fence, and fences are only top-level.
  Otherwise there is no fenced code at all, and everything is treated as text;
- a code span counts as code only when it closes on the same line with no `|` inside
  (GFM splits table cells first). Every other backtick is escaped, so the renderer
  cannot pair backticks differently.

Where the renderer sees code but the sanitizer does not, the escapes show as literal
backslashes: ugly, never unsafe. Citation tags like [S1] and [S1, S3] are kept as is.
"""

import re
import string

_LINE_ENDING = re.compile(r"(\r\n|\n|\r)")
# A fence after any indentation or list and quote markers: only column 0 is trusted.
_FENCE_LIKE = re.compile(r"(?:[ \t]|>|[-+*](?=[ \t])|\d+[.)](?=[ \t]))*(?:`{3,}|~{3,})")
_OPENER = re.compile(r"(`{3,}|~{3,})(.*)", re.DOTALL)
_CITATION = re.compile(r"\[S\d+(?:, S\d+)*\]")
# What GFM autolinks in text, cut at characters that would end a code span early.
_URL = re.compile(r"(?:https?://|www\.)[^\s<>`\[\]()|\\]*", re.IGNORECASE)
_EMAIL = re.compile(r"[A-Za-z0-9._+-]+@[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+")
_URL_START = frozenset("hHwW")
_ESCAPABLE = frozenset(string.punctuation)
_ESCAPED = {"<": "&lt;", "$": "\\$"}
_PUNCTUATION = re.compile("[" + re.escape(string.punctuation) + "]")


def sanitize_markdown(text: str) -> str:
    """Safe Markdown for any prefix too: the page re-sanitizes the whole text it has
    received on every streamed token.
    """
    pieces = _LINE_ENDING.split(text)
    lines = list(zip(pieces[0::2], [*pieces[1::2], ""], strict=True))
    fences = all(_trusted_fence_line(content) for content, _ in lines)
    out: list[str] = []
    index = 0
    while index < len(lines):
        end = _fence_end(lines, index) if fences else None
        if end is not None:
            out.extend(content + ending for content, ending in lines[index:end])
            index = end
        else:
            content, ending = lines[index]
            out.append(_inline(content) + ending)
            index += 1
    return "".join(out)


def _trusted_fence_line(content: str) -> bool:
    return _FENCE_LIKE.match(content) is None or content[0] in "`~"


def _fence_end(lines: list[tuple[str, str]], start: int) -> int | None:
    """The end (exclusive) of the fenced block opened on this line, if it opens one.
    Unclosed, it runs to the end, as at CommonMark's top level.
    """
    opener = _OPENER.fullmatch(lines[start][0])
    if opener is None:
        return None
    marker, info = opener.groups()
    if marker[0] == "`" and "`" in info:
        return None
    closer = re.compile(rf"{re.escape(marker[0])}{{{len(marker)},}}[ \t]*")
    for index in range(start + 1, len(lines)):
        if closer.fullmatch(lines[index][0]):
            return index + 1
    return len(lines)


def _inline(line: str) -> str:
    out: list[str] = []
    has_at = "@" in line
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\":
            index = _backslash(line, index, out)
        elif char == "`":
            index = _backticks(line, index, out)
        elif char == "[":
            index = _bracket(line, index, out)
        elif (link := _autolink(line, index, has_at=has_at)) is not None:
            # A backtick next to the span's own would merge with it into a longer run.
            if out and out[-1].endswith("`"):
                out.append(" ")
            index += len(link)
            out.append(f"`{link}`" + (" " if line[index : index + 1] == "`" else ""))
        else:
            out.append(_ESCAPED.get(char, char))
            index += 1
    return "".join(out)


def _backslash(line: str, index: int, out: list[str]) -> int:
    """An escape pair stays (an escaped < becomes &lt;); a lone backslash is doubled so
    it can never escape what the sanitizer writes after it.
    """
    following = line[index + 1 : index + 2]
    if following and following in _ESCAPABLE:
        out.append("&lt;" if following == "<" else "\\" + following)
        return index + 2
    out.append("\\\\")
    return index + 1


def _backticks(line: str, index: int, out: list[str]) -> int:
    run = _run_length(line, index)
    close = _closing_run(line, index + run, run)
    if close is not None and "|" not in line[index:close]:
        out.append(line[index : close + run])
        return close + run
    out.append("\\`" * run)
    return index + run


def _run_length(line: str, index: int) -> int:
    end = index
    while end < len(line) and line[end] == "`":
        end += 1
    return end - index


def _closing_run(line: str, index: int, run: int) -> int | None:
    """The start of the next backtick run of exactly this length on the line."""
    while (index := line.find("`", index)) != -1:
        length = _run_length(line, index)
        if length == run:
            return index
        index += length
    return None


def _bracket(line: str, index: int, out: list[str]) -> int:
    """A citation tag stays, but cannot open a link or a definition; any other [ is
    escaped, so no link, image, reference or footnote can form.
    """
    citation = _CITATION.match(line, index)
    if citation is None:
        out.append("\\[")
        return index + 1
    out.append(citation[0])
    end = citation.end()
    if line[end : end + 1] in ("(", ":"):
        out.append("\\" + line[end])
        return end + 1
    return end


def _autolink(line: str, index: int, *, has_at: bool) -> str | None:
    if line[index] in _URL_START and (url := _URL.match(line, index)):
        return url[0]
    # Scanning left to right, an address is always matched from its first character.
    if has_at and (email := _EMAIL.match(line, index)):
        return email[0]
    return None


def escape_markdown(text: str) -> str:
    """Backend text for a one-line label (a button): every ASCII punctuation mark is
    backslash-escaped, which CommonMark renders as the mark itself, so nothing in it
    can form an image, link, emphasis, code or HTML; the escaped `:` `/` `.` `@` also
    break GFM's bare-URL and e-mail autolinks. Whitespace runs become one space.
    """
    return _PUNCTUATION.sub(r"\\\g<0>", " ".join(text.split()))

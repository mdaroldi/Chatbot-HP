"""The answer sanitizer: no image, no live link, no raw HTML, citations untouched.

Each output is also parsed with markdown-it-py (CommonMark plus GFM tables), a
spec-compliant parser already in the lock as a Streamlit dependency, to show that the
code the sanitizer leaves alone is code to a CommonMark renderer too. It has no
linkify, so GFM's bare-URL autolinks are checked by pattern on the parsed text.
"""

import re

import pytest
from markdown_it import MarkdownIt
from markdown_it.token import Token

from hpdocs_ui.sanitize import sanitize_markdown

_PARSER = MarkdownIt("commonmark").enable(["table", "strikethrough"])
_FORBIDDEN = {"image", "link_open", "html_inline", "html_block"}
_CODE = {"code_inline", "fence", "code_block"}
# What GFM would autolink in plain text: URLs, www. hosts and e-mail addresses.
_AUTOLINK = re.compile(
    r"https?://|www\.|[A-Za-z0-9._+-]@[A-Za-z0-9_-]+\.[A-Za-z0-9_-]", re.IGNORECASE
)

HOSTILE = [
    "![img](http://evil.example/x.png)",
    "![ref][r]\n\n[r]: http://evil.example/x.png",
    "[![img](http://e.example/i.png)](http://e.example)",
    "<img src=x onerror=alert(1)>",
    "<script>alert(1)</script>",
    '<a href="http://e.example">x</a>',
    "<!-- comment --> <b>bold</b>",
    "[click](http://evil.example) and [js](javascript:alert(1))",
    "<http://evil.example> <me@evil.example>",
    "visit http://evil.example, WWW.evil.example or mail me@evil.example now",
    "a ` b ![i](s.png) c",
    "    ```\n    x\n![i](s.png)\n    ```\n![j](t.png)",
    "[a](http://x.example/a_(b) \"title\") [c](<d e> 't')",
    "See [S1](p. 4) and [S1, S3](http://e.example) and ![S2](http://e.example/i.png)",
    "[S1]: http://evil.example\n\n[S1] and [x][S1]",
    "[x]: http://evil.example\n\n[x] and [y][x]",
    "```\n<b>code</b> ![i](s.png)\n```\n![j](t.png) <i>",
    "~~~\n![i](s.png)\n~~~~\n<x> ![j](t.png)",
    "```\n![i](s.png) <b>",
    "`a\n\n![i](s.png)\n\n`",
    "| `x | ![i](s.png) `|\n|---|\n| a |",
    "| a |\n|---|\n```\n![i](s.png)\n```\n<b>",
    "- ```\n  ```\n  ![i](s.png)\n",
    " ```\n![a](b.png)\n```\n![c](d.png)\n```",
    "  ```\n  x\n     ```\n  ![a](b.png)\n```\n![c](d.png)\n```",
    "> ```\n> ![i](s.png)\n![j](t.png)",
    "\t```\n![i](s.png)\n```\n![j](t.png)",
    "```js `x`\n![i](s.png)\n```",
    "\\`![i](s.png)` and \\``x` ![j](t.png)",
    "``a`b`` ![i](s.png) `c` ```d``` <e>",
    "\\http://evil.example and `x`http://evil.example and http://e.example`y`",
    "\\<b> \\\\<i> \\[a](http://e.example)",
    "```\r![i](s.png)\r```\r![j](t.png)\r\n<b>",
    "``` \n![i](s.png)\n``\n![j](t.png)",
    "1. Step [S1]\n   ```\n   ![i](s.png)\n   ```\n2. <b>",
    "$x$ and $$y$$ and :red[colour] and [^1]\n\n[^1]: http://evil.example",
    "&lt;img src=x&gt; &#60;b&#62;",
]


def _inline_tokens(tokens: list[Token]) -> list[Token]:
    flat: list[Token] = []
    for token in tokens:
        flat.append(token)
        flat.extend(token.children or [])
    return flat


def assert_safe(sanitized: str) -> None:
    tokens = _inline_tokens(_PARSER.parse(sanitized))
    assert not [t.type for t in tokens if t.type in _FORBIDDEN], sanitized
    code = [t.content for t in tokens if t.type in _CODE]
    # Every raw < is inside code, as the parser reads it.
    assert sanitized.count("<") == sum(c.count("<") for c in code), sanitized
    text = " ".join(t.content for t in tokens if t.type == "text")
    assert not _AUTOLINK.search(text), sanitized


@pytest.mark.parametrize("hostile", HOSTILE)
def test_every_prefix_of_a_hostile_answer_renders_no_image_link_or_html(hostile: str) -> None:
    for end in range(len(hostile) + 1):
        assert_safe(sanitize_markdown(hostile[:end]))


def test_the_oracle_flags_each_unsafe_construct() -> None:
    for unsafe in ["![i](s.png)", "[a](b)", "<b>", "http://e.example", "me@e.example"]:
        with pytest.raises(AssertionError):
            assert_safe(unsafe)


# --- exact outputs ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "1. **Open** the lid.\n2. Press *OK* on the panel.\n\n- one\n- two",
        "## Battery\n\nRemove the screws (Phillips) & lift it: done!",
        "[S1]",
        "[S1, S3]",
        "Remove the battery [S1][S3].",
        "Use `print` here, or ``a`b`` there.",
        "```\n<b>![i](s.png)</b> [a](http://e.example) $x$\n```",
        "~~~python\n<b>\n~~~",
        "```\n<b> unclosed, so code to the end\n![i](s.png)",
        "````\n<b>\n```\n<i>",
        "```\n<b>\n``` x\n<i>",
        "```\n<b>\n~~~\n<i>",
        "\\* \\[ \\` \\$",
    ],
)
def test_safe_markdown_citations_and_code_are_left_byte_for_byte(text: str) -> None:
    assert sanitize_markdown(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("![img](http://e.example/x.png)", "!\\[img](`http://e.example/x.png`)"),
        ("[a](/path)", "\\[a](/path)"),
        ("[r]: /x", "\\[r]: /x"),
        ("<b>bold</b>", "&lt;b>bold&lt;/b>"),
        ("$5 or $x$", "\\$5 or \\$x\\$"),
        ("See [S1](p. 4).", "See [S1]\\(p. 4)."),
        ("[S1]: /x", "[S1]\\: /x"),
        ("[S1 ] [s1] [S1,S2] [S]", "\\[S1 ] \\[s1] \\[S1,S2] \\[S]"),
        ("go to www.hp.example now", "go to `www.hp.example` now"),
        ("HTTPS://hp.example/a?b=1.", "`HTTPS://hp.example/a?b=1.`"),
        ("mail me@hp.example.", "mail `me@hp.example`."),
        ("mailto:me@hp.example", "mailto:`me@hp.example`"),
        ("`x`http://e.example", "`x` `http://e.example`"),
        ("http://e.example`y`", "`http://e.example` `y`"),
        ("| http://a.example|b |", "| `http://a.example`|b |"),
        ("\\a and \\", "\\\\a and \\\\"),
        ("\\<b>", "&lt;b>"),
        ("a ` b", "a \\` b"),
        ("``a`", "\\`\\`a\\`"),
        ("`a\nb`", "\\`a\nb\\`"),
        ("`a | b`", "\\`a | b\\`"),
        ("```js `x`\n<b>\n```", "\\`\\`\\`js `x`\n&lt;b>\n```"),
        ("```\r\n<b>\r\n```\r\n<i>", "```\r\n<b>\r\n```\r\n&lt;i>"),
        ("```\r<b>\r```\r<i>", "```\r<b>\r```\r&lt;i>"),
    ],
)
def test_unsafe_constructs_become_plain_text(text: str, expected: str) -> None:
    assert sanitize_markdown(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "    ```\n<b>\n```\n<i>",
        " ```\n<b>\n```",
        "```\n<b>\n ```",
        "- ```\n<b>\n```",
        "> ```\n<b>\n```",
        "1. ```\n<b>\n```",
        "\t```\n<b>\n```",
    ],
)
def test_a_fence_not_at_column_zero_makes_every_fence_plain_text(text: str) -> None:
    sanitized = sanitize_markdown(text)

    assert "<" not in sanitized
    assert "```" not in sanitized.replace("\\`", "")

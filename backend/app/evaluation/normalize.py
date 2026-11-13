"""Text normalization for anchor matching.

Facts are typed from the PDFs and chunks come from text extraction, so the same fact
can differ in case, spacing, dash and quote characters. Both sides go through this
function before a substring test.
"""

import unicodedata

_SOFT_HYPHEN = "­"
# Hyphens and dashes (U+2010 to U+2015, minus sign), single quotes and prime, double
# quotes and double prime: written as code points, so the source stays plain ASCII.
_DASHES = "".join(map(chr, (*range(0x2010, 0x2016), 0x2212)))
_SINGLE_QUOTES = "".join(map(chr, (0x2018, 0x2019, 0x201A, 0x201B, 0x2032)))
_DOUBLE_QUOTES = "".join(map(chr, (0x201C, 0x201D, 0x201E, 0x2033)))
_EQUIVALENTS = str.maketrans(
    {
        **dict.fromkeys(_DASHES, "-"),
        **dict.fromkeys(_SINGLE_QUOTES, "'"),
        **dict.fromkeys(_DOUBLE_QUOTES, '"'),
        _SOFT_HYPHEN: "",
    }
)


def normalize(text: str) -> str:
    """Map before NFKC too: NFKC splits a double prime into two primes."""
    mapped = unicodedata.normalize("NFKC", text.translate(_EQUIVALENTS)).translate(_EQUIVALENTS)
    return " ".join(mapped.casefold().split())

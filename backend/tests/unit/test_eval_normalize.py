"""Text normalization for anchor matching: what counts as the same fact."""

import pytest

from app.evaluation.normalize import normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Bottom Cover", "bottom cover"),
        ("PAPER JAM", "paper jam"),
        ("Stra\u00dfe", "strasse"),  # casefold, not lower
        ("  two\n\tlines  here ", "two lines here"),
        ("non\u00a0breaking", "non breaking"),
        ("N96208\u2010001", "n96208-001"),  # hyphen
        ("N96208\u2011001", "n96208-001"),  # non-breaking hyphen
        ("N96208\u2012001", "n96208-001"),  # figure dash
        ("10\u201320", "10-20"),  # en dash
        ("a\u2014b", "a-b"),  # em dash
        ("a\u2015b", "a-b"),  # horizontal bar
        ("\u22125 V", "-5 v"),  # minus sign
        ("re\u00adinstall", "reinstall"),  # soft hyphen disappears
        ("\u2018on\u2019", "'on'"),
        ("\u201alow\u201b", "'low'"),
        ("\u201cOK\u201d", '"ok"'),
        ("\u201eOK\u201d", '"ok"'),
        ("5\u2032", "5'"),  # prime
        ("5\u2033", '5"'),  # double prime
        ("\uff21\uff22", "ab"),  # fullwidth letters (NFKC)
        ("e\u0301", "\u00e9"),  # combining accent composes
        ("[icon] button", "[icon] button"),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_normalize_is_idempotent() -> None:
    text = "  \u201cN96208\u2013001\u201d  Bottom\u00a0Cover "

    assert normalize(normalize(text)) == normalize(text)


def test_an_empty_string_stays_empty() -> None:
    assert normalize(" \n ") == ""

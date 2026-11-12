"""Behavior of the per-line script and language rules (synthetic sentences)."""

import pytest

from app.ingestion.language import is_non_latin, score_line

# ------------------------------------------------------------------ non-Latin scripts


@pytest.mark.parametrize(
    "text",
    [
        "本机支持无线打印",  # CJK
        "무선 인쇄를 지원합니다",  # Hangul
        "รองรับการพิมพ์ไร้สาย",  # Thai
        "Поддерживает беспроводную печать",  # Cyrillic
        "ワイヤレス印刷に対応しています",  # kana and CJK
        "ｶﾀｶﾅ",  # half-width katakana
        "يدعم الطباعة اللاسلكية",  # Arabic
        "תומך בהדפסה אלחוטית",  # Hebrew
        "本机支持 Wi-Fi 打印",  # mostly CJK, with a Latin name
    ],
)
def test_lines_mostly_in_a_non_latin_script_are_non_latin(text: str) -> None:
    assert is_non_latin(text)


def test_latin_greek_and_letterless_lines_are_not_non_latin() -> None:
    assert is_non_latin("本机支持无线打印")
    assert not is_non_latin("Résistance de 1 MΩ ±10 %")  # accented Latin, Greek unit
    assert not is_non_latin("Model 5 supports 本 printing")  # one CJK letter among Latin
    assert not is_non_latin("12 34 --")


def test_icon_placeholders_are_not_latin_letters() -> None:
    """Two CJK letters beside an icon are mostly CJK; beside a word they are not."""
    assert is_non_latin("按下 [icon]")
    assert not is_non_latin("按下 Wi-Fi")


# ------------------------------------------------------------------ non-English Latin


@pytest.mark.parametrize(
    "text",
    [
        "Das Gerät ist nicht für den Betrieb im Freien geeignet.",
        "Le produit doit être utilisé dans un local fermé.",
        "El equipo debe usarse con una fuente externa.",
        "O aparelho não deve ser usado com uma fonte externa.",
        "Il dispositivo non deve essere usato con una fonte esterna.",
        "Het apparaat is niet geschikt voor gebruik buiten.",
    ],
)
def test_lines_in_another_latin_language_are_non_english(text: str) -> None:
    assert score_line(text).is_non_english


def test_english_lines_are_not_non_english() -> None:
    assert score_line("Le produit doit être utilisé dans un local fermé.").is_non_english
    assert not score_line("The device is not suitable for outdoor use.").is_non_english
    assert not score_line("Connect the cable to the port on the back.").is_non_english


def test_a_line_needs_three_words() -> None:
    assert score_line("de la porte").is_non_english
    assert not score_line("de la").is_non_english
    assert not score_line("de la 12 34").is_non_english  # numbers are not words


def test_one_listed_word_is_not_enough() -> None:
    assert score_line("Réglez le bouton de").is_non_english
    assert not score_line("Réglez le bouton").is_non_english


def test_the_other_language_must_outscore_english() -> None:
    assert score_line("de la the porte").is_non_english
    assert not score_line("de la the of porte").is_non_english


def test_english_words_never_count_for_another_language() -> None:
    """Words such as "is", "in", "do", "as", "a" and "per" are on the English list only,
    so an English line cannot score for Dutch or Portuguese by accident.
    """
    scores = score_line("Is it in a box, do as per o")

    assert (scores.words, scores.other) == (9, 1)
    assert scores.english == 7


def test_icon_placeholders_are_not_words() -> None:
    assert score_line("de la porte").words == 3
    assert score_line("de la [icon]").words == 2
    assert not score_line("de la [icon]").is_non_english


def test_urls_are_removed_before_scoring() -> None:
    assert score_line("voir de la page").is_non_english
    assert score_line("www.example.org/de/la/une").words == 0
    assert score_line("See https://example.org/de/la/une/page now").words == 2


def test_french_elisions_count_as_french() -> None:
    assert score_line("qu'il n'est l'appareil").other == 3
    assert score_line("l\u2019appareil d\u2019un type").is_non_english  # typographic apostrophe
    assert not score_line("Don't use it, isn't it").is_non_english


def test_portuguese_words_without_accents_are_listed() -> None:
    assert score_line("O cabo deste modelo nao foi testado pela fabrica").other == 5
    assert score_line("desta pelo").other == 2

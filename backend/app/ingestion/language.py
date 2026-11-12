"""Per-line script and language rules for cleaning (ADR 0004, Phase 1 plan item 3).

The corpus is English, but its regulatory back matter repeats notices in other
languages. Script detection drops lines written mostly in non-Latin scripts;
function-word scores drop Latin-script lines in another European language. Both
ignore `[icon]` placeholders: an icon is neither a word nor a letter.
"""

import re
import unicodedata
from dataclasses import dataclass

from app.domain.ingestion import ICON

# Unicode name prefixes of the scripts dropped as non-Latin. Greek is kept: the
# manuals use it for unit symbols (MΩ, µm).
_NON_LATIN_SCRIPTS = frozenset(
    {"CJK", "HANGUL", "THAI", "CYRILLIC", "HIRAGANA", "KATAKANA", "ARABIC", "HEBREW"}
)
_WIDTH_PREFIXES = ("HALFWIDTH", "FULLWIDTH")

MIN_WORDS = 3
MIN_SCORE = 2

# Function words only. A word that is also English ("in", "is", "do", "as", "a")
# appears in the English list alone, so English text never scores for another
# language by accident; a unit test pins this.
# fmt: off
_ENGLISH = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "by", "for", "with",
    "from", "as", "is", "are", "was", "be", "been", "it", "its", "this", "that", "these",
    "those", "not", "no", "do", "does", "can", "will", "your", "you", "if", "into", "than",
    "then", "so", "per", "am"
})
_OTHER = {
    "de": frozenset({
        "der", "die", "das", "und", "ist", "nicht", "mit", "für", "von", "zu", "den",
        "dem", "des", "ein", "eine", "einer", "eines", "im", "auf", "sich", "werden",
        "wird", "dieses", "diese", "bei", "nach", "oder", "sind", "kann", "um", "aus"
    }),
    "fr": frozenset({
        "le", "la", "les", "de", "des", "du", "un", "une", "et", "est", "sont", "pour",
        "dans", "par", "sur", "avec", "ce", "cet", "cette", "ces", "au", "aux", "qui",
        "que", "ne", "pas", "son", "sa", "ses", "il", "elle", "être", "à", "ou", "peut"
    }),
    "es": frozenset({
        "el", "la", "los", "las", "de", "del", "y", "es", "son", "para", "por", "con",
        "una", "un", "que", "se", "en", "este", "esta", "su", "sus", "al", "como", "o",
        "puede"
    }),
    "pt": frozenset({
        "o", "os", "de", "da", "dos", "das", "e", "é", "em", "um", "uma", "para",
        "com", "por", "que", "não", "se", "seu", "sua", "ao", "este", "esta", "na",
        "nos", "ser", "deste", "desta", "foi", "nao", "pelo", "pela"
    }),
    "it": frozenset({
        "il", "lo", "la", "gli", "le", "di", "del", "della", "e", "è", "un", "una",
        "con", "che", "non", "questo", "questa", "da", "al", "sono"
    }),
    "nl": frozenset({
        "de", "het", "een", "en", "van", "zijn", "voor", "met", "op", "dit", "deze",
        "niet", "te", "aan", "dat", "bij", "wordt"
    }),
}
# fmt: on
# French elided articles and pronouns ("l'appareil", "d'un") are French evidence.
_FRENCH_ELISION = re.compile(r"(?:l|d|qu|n)['\u2019]")
_URL = re.compile(r"(?:https?://|www\.)\S*")
_WORD = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*")


def _script(char: str) -> str:
    parts = unicodedata.name(char, "").split()
    if len(parts) > 1 and parts[0] in _WIDTH_PREFIXES:
        return parts[1]
    return parts[0] if parts else ""


def is_non_latin(text: str) -> bool:
    """Most letters are in a dropped script; a line with no letters is not."""
    letters = [char for char in text.replace(ICON, " ") if char.isalpha()]
    non_latin = sum(1 for char in letters if _script(char) in _NON_LATIN_SCRIPTS)
    return 2 * non_latin > len(letters)


@dataclass(frozen=True)
class LanguageScores:
    """Word count and function-word hits: English, and the best other language."""

    words: int
    english: int
    other: int

    @property
    def is_non_english(self) -> bool:
        return self.words >= MIN_WORDS and self.other >= MIN_SCORE and self.other > self.english


def _hits(words: list[str], language: str) -> int:
    listed = _OTHER[language]
    if language == "fr":
        return sum(1 for w in words if w in listed or _FRENCH_ELISION.match(w))
    return sum(1 for w in words if w in listed)


def score_line(text: str) -> LanguageScores:
    """Scores after removing URLs and icons, whose tokens are not prose."""
    words = _WORD.findall(_URL.sub(" ", text.replace(ICON, " ")).casefold())
    return LanguageScores(
        words=len(words),
        english=sum(1 for w in words if w in _ENGLISH),
        other=max(_hits(words, language) for language in _OTHER),
    )

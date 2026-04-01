from __future__ import annotations

from collections.abc import Iterable


FRENCH_HINTS = {
    "bonjour",
    "avec",
    "pour",
    "etre",
    "dans",
    "nous",
    "vous",
    "une",
    "des",
    "pas",
    "est",
    "merci",
}

ENGLISH_HINTS = {
    "hello",
    "with",
    "for",
    "the",
    "and",
    "you",
    "we",
    "are",
    "this",
    "that",
    "please",
    "thank",
}


def _score_language(text: str, hints: Iterable[str]) -> int:
    lowered = f" {text.lower()} "
    return sum(1 for hint in hints if f" {hint} " in lowered)


def detect_source_language(text: str) -> str:
    french_score = _score_language(text, FRENCH_HINTS)
    english_score = _score_language(text, ENGLISH_HINTS)

    if any(character in text.lower() for character in "àâæçéèêëîïôœùûüÿ"):
        french_score += 2

    return "fr" if french_score > english_score else "en"

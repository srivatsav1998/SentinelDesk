"""Small text helpers shared across the pipeline."""

from __future__ import annotations

import re

SUPPORTED_LANGUAGES = ("en", "de")

_GERMAN_HINTS = {
    "und", "der", "die", "das", "wir", "nicht", "bitte", "hilfe", "sehr", "geehrte", "danke",
    "ist", "mit", "für", "ich", "ein", "eine", "unser", "unsere", "können", "konnte", "wurde",
    "problem", "dringend", "zugang", "konto", "rechnung", "freundlichen", "grüßen",
}
_ENGLISH_HINTS = {
    "the", "and", "is", "are", "we", "our", "please", "thank", "you", "with", "for", "not",
    "issue", "help", "regards", "dear", "unable", "account", "this", "that",
}


def ticket_text(subject: object, body: object) -> str:
    """Canonical "subject\\nbody" text; tolerates None/NaN subjects."""
    subject = subject if isinstance(subject, str) else ""
    body = body if isinstance(body, str) else ""
    return f"{subject}\n{body}".strip()


def _heuristic_language(text: str) -> str:
    words = re.findall(r"[a-zäöüß]+", text.lower())
    german = sum(w in _GERMAN_HINTS for w in words) + 2 * len(re.findall(r"[äöüß]", text.lower()))
    english = sum(w in _ENGLISH_HINTS for w in words)
    return "de" if german > english else "en"


def detect_language(text: str) -> str:
    """Return "en" or "de" deterministically.

    Uses langdetect with a fixed seed (it is random by default) and keeps the most
    probable of the two supported languages; falls back to a stop-word heuristic
    when langdetect is unavailable or proposes neither.
    """
    clean = (text or "").strip()
    if not clean:
        return "en"
    try:
        from langdetect import DetectorFactory, detect_langs

        DetectorFactory.seed = 0
        for candidate in detect_langs(clean):
            if candidate.lang in SUPPORTED_LANGUAGES:
                return candidate.lang
    except Exception:
        pass
    return _heuristic_language(clean)

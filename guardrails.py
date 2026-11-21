"""Text guardrails shared by the safety gate and the drafter.

* Sensitive-request detection (English + German): refunds, compensation,
  legal threats, contractual guarantees. A request is "unsupported" when no
  retrieved historical resolution covers the same kind of action.
* Prompt-injection detection: tickets that try to override the assistant's
  instructions are never auto-drafted.
* PII masking: emails, phone numbers, IP addresses, IBANs, card and account
  numbers are replaced with placeholders before text leaves the machine.

Pure standard library so it can be unit-tested without the ML stack.
"""

from __future__ import annotations

import re
from typing import Iterable

# ---------------------------------------------------------------------------
# Sensitive actions
# ---------------------------------------------------------------------------
# Each category maps to a regex matched case-insensitively against the ticket
# (what the customer asks for) and against precedent answers / drafts
# (what the company has historically done or is about to promise).
SENSITIVE_PATTERNS: dict[str, re.Pattern[str]] = {
    "refund": re.compile(
        r"\brefund\w*|\bmoney back\b|\breimburs\w*|\bcharge ?backs?\b"
        r"|r(?:ü|ue)ckerstatt\w*|\berstatt\w*|\bgeld zur(?:ü|ue)ck\b|r(?:ü|ue)ckzahlung\w*",
        re.IGNORECASE,
    ),
    "compensation": re.compile(
        r"\bcompensat\w*|\bdamages\b|\bservice credits?\b|\bsla credits?\b|\bpenalt(?:y|ies)\b"
        r"|entsch(?:ä|ae)digung\w*|schadens?ersatz\w*|\bkompensation\w*|vertragsstrafe\w*",
        re.IGNORECASE,
    ),
    "legal": re.compile(
        r"\blawsuits?\b|\blegal action\b|\bsue\b|\bsuing\b|\battorneys?\b|\blawyers?\b"
        r"|\blitigation\b|\btake (?:you|this) to court\b|\bbreach of contract\b"
        r"|\b(?:rechts)?anw(?:a|ä)lt\w*|\bklage\w*|\bgerichtlich\w*|\brechtliche schritte\b|\bvertragsbruch\b",
        re.IGNORECASE,
    ),
    "guarantee": re.compile(
        r"\bguarantee\w*|\bwritten (?:assurance|commitment|confirmation)\b|\buptime commitment\b"
        r"|\bgarantie\w*|\bgarantier\w*|\bzusicherung\w*|\bverbindlich(?:e|en)? zusage\w*",
        re.IGNORECASE,
    ),
}

# Categories that always need a human, even if a precedent mentions them.
ALWAYS_ESCALATE = {"legal"}

_NEGATION = re.compile(
    r"\b(?:not|no|cannot|can't|unable|never|without|nicht|kein\w*|keine\w*|leider)\b", re.IGNORECASE
)


def detect_sensitive_requests(text: str) -> list[str]:
    """Sensitive action categories mentioned in `text` (stable order)."""
    text = text or ""
    return [name for name, pattern in SENSITIVE_PATTERNS.items() if pattern.search(text)]


def unsupported_requests(ticket_text: str, precedent_answers: Iterable[str]) -> list[str]:
    """Sensitive categories the ticket asks for that no precedent answer covers.

    Categories in ALWAYS_ESCALATE are returned whenever the ticket mentions them.
    """
    requested = detect_sensitive_requests(ticket_text)
    if not requested:
        return []
    evidence = "\n".join(a for a in precedent_answers if a)
    unsupported = []
    for name in requested:
        if name in ALWAYS_ESCALATE or not SENSITIVE_PATTERNS[name].search(evidence):
            unsupported.append(name)
    return unsupported


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def affirmative_commitments(text: str) -> list[str]:
    """Sensitive categories asserted in `text`, ignoring negated sentences
    ("we cannot guarantee ..." is a refusal, not a commitment)."""
    found: list[str] = []
    for sentence in _sentences(text):
        if _NEGATION.search(sentence):
            continue
        for name, pattern in SENSITIVE_PATTERNS.items():
            if name not in found and pattern.search(sentence):
                found.append(name)
    return found


# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------
_INJECTION = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}\b(?:previous|prior|above|earlier|all|your|system)\b"
    r"[^.\n]{0,20}\b(?:instructions?|rules|prompts?|guidelines|policies)\b"
    r"|\bsystem prompt\b|\bdeveloper mode\b|\bjailbreak\w*|\bdo anything now\b"
    r"|\byou are now (?:a|an|the)\b|\bpretend (?:to be|you are)\b|\bact as (?:a|an|the) (?!customer)\w+"
    r"|\breveal (?:your|the) (?:prompt|instructions|system)\b"
    r"|\bignorier\w*\b[^.\n]{0,40}\b(?:anweisungen|regeln|vorgaben)\b"
    r"|\bvergiss\b[^.\n]{0,40}\b(?:anweisungen|regeln|vorgaben)\b",
    re.IGNORECASE,
)


def detect_prompt_injection(text: str) -> list[str]:
    """Matched injection phrases (empty when none)."""
    return [m.group(0).strip() for m in _INJECTION.finditer(text or "")]


# ---------------------------------------------------------------------------
# PII masking
# ---------------------------------------------------------------------------
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b")),
    ("IP", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    # A bare run of 8+ digits is an account/customer number; numbers written with
    # "+", spaces or dashes are treated as phone numbers.
    ("ACCOUNT", re.compile(r"\b\d{8,}\b")),
    ("PHONE", re.compile(r"(?<![\w.])\+?\d[\d ()/-]{7,}\d\b")),
]


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace PII with [TYPE] placeholders. Returns (masked_text, counts)."""
    counts: dict[str, int] = {}
    masked = text or ""
    for label, pattern in _PII_PATTERNS:
        masked, n = pattern.subn(f"[{label}]", masked)
        if n:
            counts[label] = counts.get(label, 0) + n
    return masked, counts

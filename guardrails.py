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
    r"\b(?:not|no|never|cannot|unable|without|nicht|nie|niemals|kein\w*)\b|\w+n['’]t\b", re.IGNORECASE
)
# A negation only applies within its own clause ("we will not delay; we will refund"
# still commits to a refund).
_CLAUSE_BREAK = re.compile(r"[,;:.!?\n]|\b(?:but|however|although|aber|jedoch|sondern)\b", re.IGNORECASE)
_NEGATION_WINDOW = 6  # words before the term


def _is_negated(text: str, start: int) -> bool:
    clause = _CLAUSE_BREAK.split(text[:start])[-1]
    return bool(_NEGATION.search(" ".join(clause.split()[-_NEGATION_WINDOW:])))


# Commitments checked in drafts: the gate categories plus discounts / waived fees,
# which are common in sales tickets (so not a gate rule) but must never be invented.
COMMITMENT_PATTERNS: dict[str, re.Pattern[str]] = {
    **SENSITIVE_PATTERNS,
    "discount": re.compile(
        r"\bdiscount\w*|\bfree of charge\b|\bwaive\w*|\bcomplimentary\b|\bprice (?:match|guarantee|freeze)\b"
        # German compounds: "Mengenrabatt", "Treuerabatt", "Sonderpreisnachlass"
        r"|rabatt\w*|preisnachlass\w*|\bkostenlos\w*|\bgratis\b|\bgebühr\w* erlass\w*",
        re.IGNORECASE,
    ),
}


def affirmative_mentions(text: str, patterns: dict[str, re.Pattern[str]] = SENSITIVE_PATTERNS) -> list[str]:
    """Categories with at least one non-negated mention in `text`.

    "We cannot offer a refund" is a refusal, not a commitment, and does not count;
    "No worries, we will refund your fee" does count.
    """
    text = text or ""
    found: list[str] = []
    for name, pattern in patterns.items():
        if any(not _is_negated(text, m.start()) for m in pattern.finditer(text)):
            found.append(name)
    return found


def affirmative_commitments(text: str) -> list[str]:
    """Commitments (refund, compensation, guarantee, legal, discount) asserted in a draft."""
    return affirmative_mentions(text, COMMITMENT_PATTERNS)


def detect_sensitive_requests(text: str) -> list[str]:
    """Sensitive action categories mentioned in `text` (stable order). Every mention
    counts here, negated or not, so the gate errs towards escalation."""
    text = text or ""
    return [name for name, pattern in SENSITIVE_PATTERNS.items() if pattern.search(text)]


def unsupported_requests(ticket_text: str, precedent_answers: Iterable[str]) -> list[str]:
    """Sensitive categories the ticket asks for that no precedent answer affirmatively
    covers (a precedent that refused a refund is not support for one).

    Categories in ALWAYS_ESCALATE are returned whenever the ticket mentions them.
    """
    requested = detect_sensitive_requests(ticket_text)
    if not requested:
        return []
    supported = set(affirmative_mentions("\n".join(a for a in precedent_answers if a)))
    return [name for name in requested if name in ALWAYS_ESCALATE or name not in supported]


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

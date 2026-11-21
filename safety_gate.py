"""
SentinelDesk Safety Escalation Gate

Decides whether a triaged ticket can be auto-drafted by the AI or must be
routed to a human on-call engineer. The gate runs after merge_triage and
before retrieval / escalation in the LangGraph workflow (graph.py), and is
also used by the standalone pipeline (pipeline.py).

Escalation rules (any one triggers ESCALATE):
  1. high_priority_incident  - priority == "high" AND type == "Incident"
  2. low_confidence          - aggregate confidence < CONFIDENCE_THRESHOLD
  3. missing_prediction      - type, priority or confidence is missing or invalid
                               (fail-safe: never auto-draft on incomplete triage)
  4. classifier_unavailable  - a classifier fell back to keyword rules because its
                               model checkpoint could not be loaded. Only enforced
                               when ESCALATE_ON_CLASSIFIER_FALLBACK is on; otherwise
                               the fallback is noted in the reason but does not
                               block auto-drafting.
  5. low_similarity          - no retrieved precedent, or the best precedent's cosine
                               similarity is below SIMILARITY_THRESHOLD (out-of-scope
                               ticket with no historical grounding)
  6. retrieval_unavailable   - precedent search failed, so a draft cannot be grounded
  7. unsupported_request     - the ticket asks for a sensitive action (refund,
                               compensation, guarantee) that none of the retrieved
                               resolutions covers, or makes a legal threat
  8. prompt_injection        - the ticket tries to override the assistant's instructions

Rules 5-7 apply only when the caller passes the retrieval result (`precedents`).

Settings (environment variables, read once at import):
  SENTINEL_CONFIDENCE_THRESHOLD   float, default 0.65
  SENTINEL_SIMILARITY_THRESHOLD   float, default 0.58 (calibrated: off-topic queries score
                                  0.33-0.55, realistic in-scope tickets 0.60-0.80, KB neighbours
                                  median 0.88; see findings/02)
  SENTINEL_ESCALATE_ON_FALLBACK   "1"/"true"/"yes" to enforce rule 4, default ON now that
                                  every classifier (type, priority: mBERT; queue: TF-IDF)
                                  ships in models/. Set "false" to only note fallbacks.

This module has no third-party dependencies so it can be unit-tested and
reused without loading LangGraph or the BERT models.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Optional, Sequence

from guardrails import detect_prompt_injection, unsupported_requests

Route = Literal["escalate", "auto_draft"]

# Rule identifiers (stable strings; safe to log, test against, or show in the UI)
HIGH_PRIORITY_INCIDENT = "high_priority_incident"
LOW_CONFIDENCE = "low_confidence"
MISSING_PREDICTION = "missing_prediction"
CLASSIFIER_UNAVAILABLE = "classifier_unavailable"
LOW_SIMILARITY = "low_similarity"
RETRIEVAL_UNAVAILABLE = "retrieval_unavailable"
UNSUPPORTED_REQUEST = "unsupported_request"
PROMPT_INJECTION = "prompt_injection"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if 0.0 <= value <= 1.0 else default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


CONFIDENCE_THRESHOLD: float = _env_float("SENTINEL_CONFIDENCE_THRESHOLD", 0.65)
SIMILARITY_THRESHOLD: float = _env_float("SENTINEL_SIMILARITY_THRESHOLD", 0.58)
ESCALATE_ON_CLASSIFIER_FALLBACK: bool = _env_flag("SENTINEL_ESCALATE_ON_FALLBACK", True)


@dataclass(frozen=True)
class GateDecision:
    route: Route
    triggers: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""
    unavailable_classifiers: tuple[str, ...] = field(default_factory=tuple)
    top_similarity: Optional[float] = None
    unsupported: tuple[str, ...] = field(default_factory=tuple)

    @property
    def escalate(self) -> bool:
        return self.route == "escalate"

    @property
    def status(self) -> str:
        """Value for the UI / state field `escalation_status`."""
        return "ESCALATE" if self.escalate else "AUTO_DRAFT"


def _clean_label(value: Optional[str]) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _valid_confidence(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if math.isnan(value) or not 0.0 <= value <= 1.0:
        return None
    return value


def evaluate_safety_gate(
    *,
    ticket_type: Optional[str],
    priority: Optional[str],
    confidence: object,
    unavailable_classifiers: Iterable[str] = (),
    threshold: Optional[float] = None,
    escalate_on_fallback: Optional[bool] = None,
    ticket_text: str = "",
    precedents: Optional[Sequence[Mapping]] = None,
    retrieval_error: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
) -> GateDecision:
    """Apply the escalation rules to one triaged ticket.

    `threshold`, `similarity_threshold` and `escalate_on_fallback` default to the
    module settings; pass them explicitly to override (e.g. in tests or sweeps).
    `precedents` is the retrieval result (list of dicts with "similarity" and
    "historical_answer"); pass None to skip the retrieval-based rules.
    """
    threshold = CONFIDENCE_THRESHOLD if threshold is None else threshold
    similarity_threshold = SIMILARITY_THRESHOLD if similarity_threshold is None else similarity_threshold
    if escalate_on_fallback is None:
        escalate_on_fallback = ESCALATE_ON_CLASSIFIER_FALLBACK

    unavailable = tuple(unavailable_classifiers)
    type_label = _clean_label(ticket_type)
    priority_label = _clean_label(priority)
    score = _valid_confidence(confidence)

    triggers: list[str] = []
    reasons: list[str] = []

    missing = [name for name, ok in (
        ("type", bool(type_label)),
        ("priority", bool(priority_label)),
        ("confidence", score is not None),
    ) if not ok]
    if missing:
        triggers.append(MISSING_PREDICTION)
        reasons.append(f"Triage output incomplete or invalid ({', '.join(missing)}). Flagged for human review.")

    if priority_label == "high" and type_label == "incident":
        triggers.append(HIGH_PRIORITY_INCIDENT)
        reasons.append("High-Priority Incident detected. Immediate human intervention required.")

    if score is not None and score < threshold:
        triggers.append(LOW_CONFIDENCE)
        reasons.append(f"Model confidence {score:.2f} is below threshold ({threshold:.2f}). Flagged for human review.")

    if unavailable and escalate_on_fallback:
        triggers.append(CLASSIFIER_UNAVAILABLE)
        reasons.append(f"Classifier model unavailable for: {', '.join(unavailable)}. Flagged for human review.")

    injection = detect_prompt_injection(ticket_text)
    if injection:
        triggers.append(PROMPT_INJECTION)
        reasons.append(f"Ticket tries to override assistant instructions (\"{injection[0][:60]}\"). "
                       "Auto-reply blocked; flagged for human review.")

    top_similarity: Optional[float] = None
    unsupported: tuple[str, ...] = ()
    if retrieval_error:
        triggers.append(RETRIEVAL_UNAVAILABLE)
        reasons.append(f"Precedent search unavailable ({retrieval_error[:120]}); a grounded reply cannot be drafted.")
    elif precedents is not None:
        similarities = [_valid_confidence(p.get("similarity")) for p in precedents]
        similarities = [s for s in similarities if s is not None]
        top_similarity = max(similarities) if similarities else None
        if top_similarity is None:
            triggers.append(LOW_SIMILARITY)
            reasons.append("No historical precedent found for this request (out of scope). Flagged for human review.")
        elif top_similarity < similarity_threshold:
            triggers.append(LOW_SIMILARITY)
            reasons.append(f"Best historical precedent similarity {top_similarity:.2f} is below threshold "
                           f"({similarity_threshold:.2f}); no reliable grounding. Flagged for human review.")
        unsupported = tuple(unsupported_requests(
            ticket_text, [str(p.get("historical_answer") or "") for p in precedents]
        ))
        if unsupported:
            triggers.append(UNSUPPORTED_REQUEST)
            reasons.append(f"Ticket requests action(s) not supported by historical resolutions "
                           f"({', '.join(unsupported)}). Flagged for human review.")

    if triggers:
        route: Route = "escalate"
        reason = " ".join(reasons)
    else:
        route = "auto_draft"
        similarity_note = (
            f", best precedent similarity {top_similarity:.2f} meets threshold ({similarity_threshold:.2f})"
            if top_similarity is not None else ""
        )
        reason = (
            f"Passed safety gate: confidence {score:.2f} meets threshold ({threshold:.2f}){similarity_note} "
            f"and ticket is not a high-priority incident. Safe for AI auto-reply grounded in historical precedent."
        )

    if unavailable and not escalate_on_fallback:
        reason = f"{reason} (Keyword fallback used for: {', '.join(unavailable)})"

    return GateDecision(
        route=route,
        triggers=tuple(triggers),
        reason=reason,
        unavailable_classifiers=unavailable,
        top_similarity=top_similarity,
        unsupported=unsupported,
    )

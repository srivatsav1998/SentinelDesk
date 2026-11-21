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

Settings (environment variables, read once at import):
  SENTINEL_CONFIDENCE_THRESHOLD   float, default 0.65
  SENTINEL_ESCALATE_ON_FALLBACK   "1"/"true"/"yes" to enforce rule 4, default off.
                                  Turn on once every classifier checkpoint
                                  (type, queue, priority) is in models/.

This module has no third-party dependencies so it can be unit-tested and
reused without loading LangGraph or the BERT models.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

Route = Literal["escalate", "auto_draft"]

# Rule identifiers (stable strings; safe to log, test against, or show in the UI)
HIGH_PRIORITY_INCIDENT = "high_priority_incident"
LOW_CONFIDENCE = "low_confidence"
MISSING_PREDICTION = "missing_prediction"
CLASSIFIER_UNAVAILABLE = "classifier_unavailable"


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
ESCALATE_ON_CLASSIFIER_FALLBACK: bool = _env_flag("SENTINEL_ESCALATE_ON_FALLBACK", False)


@dataclass(frozen=True)
class GateDecision:
    route: Route
    triggers: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""
    unavailable_classifiers: tuple[str, ...] = field(default_factory=tuple)

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
) -> GateDecision:
    """Apply the escalation rules to one triaged ticket.

    `threshold` and `escalate_on_fallback` default to the module settings; pass
    them explicitly to override (e.g. in tests or evaluation sweeps).
    """
    threshold = CONFIDENCE_THRESHOLD if threshold is None else threshold
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

    if triggers:
        route: Route = "escalate"
        reason = " ".join(reasons)
    else:
        route = "auto_draft"
        reason = (
            f"Passed safety gate: confidence {score:.2f} meets threshold ({threshold:.2f}) "
            f"and ticket is not a high-priority incident. Safe for AI auto-reply grounded in historical precedent."
        )

    if unavailable and not escalate_on_fallback:
        reason = f"{reason} (Keyword fallback used for: {', '.join(unavailable)})"

    return GateDecision(
        route=route,
        triggers=tuple(triggers),
        reason=reason,
        unavailable_classifiers=unavailable,
    )

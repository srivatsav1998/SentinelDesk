"""
SentinelDesk LangGraph Agentic Workflow

  1. detect_language      deterministic en/de detection (text_utils.detect_language)
  2. Concurrent fan-out:
       classify_type      fine-tuned mBERT (triage_engine.predict_type)
       classify_queue     queue classifier (triage_engine.predict_queue)
       classify_priority  fine-tuned mBERT (triage_engine.predict_priority)
       retrieve           top-k precedents from the 80% KB (retriever.retrieve_precedents)
  3. merge_triage         aggregate confidence
  4. safety_gate          all escalation rules (safety_gate.evaluate_safety_gate), including
                          retrieval similarity and unsupported-request checks
  5a. draft               grounded Gemini draft + verifier (drafter.generate_grounded_draft)
  5b. escalate            holding reply for the customer + handoff note for on-call
"""

import os
import time
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from drafter import generate_grounded_draft
from retriever import retrieve_precedents
from safety_gate import HIGH_PRIORITY_INCIDENT, evaluate_safety_gate
from text_utils import detect_language, ticket_text
from triage_engine import ModelUnavailableError, predict_priority, predict_queue, predict_type

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(os.environ.get(name, default))))
    except ValueError:
        return default


TOP_K = _env_int("SENTINEL_TOP_K", 3, 1, 10)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------
class SentinelState(TypedDict, total=False):
    subject: Optional[str]
    body: str
    top_k: int

    detected_language: str

    # Concurrent triage outputs
    type: str
    type_confidence: float
    queue: str
    queue_confidence: float
    priority: str
    priority_confidence: float
    type_model_error: str
    queue_model_error: str
    priority_model_error: str
    type_ms: float
    queue_ms: float
    priority_ms: float

    confidence: float
    latency_ms: float

    # Safety gate
    escalation_status: str          # "AUTO_DRAFT" or "ESCALATE"
    escalation_reason: str
    escalation_triggers: List[str]  # rule ids from safety_gate.py, empty when auto-drafting
    gate_route: str                 # "escalate" or "auto_draft"
    top_similarity: Optional[float]
    unsupported_requests: List[str]

    # Retrieval & generation
    retrieved_precedents: List[Dict[str, Any]]
    retrieval_error: str
    retrieval_ms: float
    draft_response: str
    draft_source: str               # "gemini:<model>", "template" or "escalation"
    draft_verified: bool
    draft_issues: List[str]
    draft_rejected: List[Dict[str, Any]]
    llm_error: str
    pii_masked: Dict[str, int]
    draft_ms: float
    handoff_note: str

    _start_time: float


# ---------------------------------------------------------------------------
# 1. Language detection
# ---------------------------------------------------------------------------
def detect_language_node(state: SentinelState) -> Dict[str, Any]:
    start = time.perf_counter()
    return {
        "_start_time": start,
        "detected_language": detect_language(ticket_text(state.get("subject"), state.get("body"))),
    }


# ---------------------------------------------------------------------------
# 2. Concurrent triage + retrieval (fan-out)
# ---------------------------------------------------------------------------
def _keyword_fallback(task: str, text: str) -> tuple[str, float]:
    """Last-resort labels when a checkpoint cannot be loaded (gate is told via *_model_error)."""
    if task == "type":
        if any(k in text for k in ["outage", "down", "fails", "broken", "breach", "disrupting"]):
            return "Incident", 0.94
        if any(k in text for k in ["inquire", "request", "need access", "upgrade", "how to"]):
            return "Request", 0.90
        if any(k in text for k in ["problem", "bug", "error"]):
            return "Problem", 0.85
        return "Change", 0.78
    if task == "queue":
        if any(k in text for k in ["billing", "pricing", "payment", "invoice", "rechnung", "sepa"]):
            return "Billing and Payments", 0.95
        if any(k in text for k in ["server", "platform", "outage", "cluster", "telemedicine"]):
            return "Technical Support", 0.92
        if any(k in text for k in ["login", "password", "vpn", "account", "reset"]):
            return "IT Support", 0.89
        return "Customer Service", 0.82
    if task == "priority":
        if any(k in text for k in ["critical", "outage", "dringend", "emergency", "fail", "urgent", "breach"]):
            return "high", 0.96
        if any(k in text for k in ["inquiry", "how to", "pricing", "frage"]):
            return "low", 0.91
        return "medium", 0.82
    return "Unavailable", 0.0


def _classify(state: SentinelState, task: str, predictor) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        label, confidence = predictor(state.get("body", ""), subject=state.get("subject"))
        return {task: label, f"{task}_confidence": confidence, f"{task}_ms": _elapsed_ms(start)}
    except (ModelUnavailableError, Exception) as error:
        text = ticket_text(state.get("subject"), state.get("body")).lower()
        label, confidence = _keyword_fallback(task, text)
        return {
            task: label,
            f"{task}_confidence": confidence,
            f"{task}_model_error": str(error),
            f"{task}_ms": _elapsed_ms(start),
        }


def classify_type_node(state: SentinelState) -> Dict[str, Any]:
    return _classify(state, "type", predict_type)


def classify_queue_node(state: SentinelState) -> Dict[str, Any]:
    return _classify(state, "queue", predict_queue)


def classify_priority_node(state: SentinelState) -> Dict[str, Any]:
    return _classify(state, "priority", predict_priority)


def retrieval_node(state: SentinelState) -> Dict[str, Any]:
    """Top-k precedents from the historical KB. No mock fallback: a failure is
    reported to the gate, which escalates instead of drafting ungrounded text."""
    start = time.perf_counter()
    top_k = int(state.get("top_k") or TOP_K)
    try:
        precedents = retrieve_precedents(ticket_text(state.get("subject"), state.get("body")), top_k=top_k)
        return {"retrieved_precedents": precedents, "retrieval_ms": _elapsed_ms(start)}
    except Exception as error:
        return {
            "retrieved_precedents": [],
            "retrieval_error": f"{type(error).__name__}: {error}",
            "retrieval_ms": _elapsed_ms(start),
        }


# ---------------------------------------------------------------------------
# 3. Merge (fan-in)
# ---------------------------------------------------------------------------
def merge_triage_node(state: SentinelState) -> Dict[str, Any]:
    """Aggregate routing score across type (40%), queue (35%) and priority (25%)."""
    type_conf = state.get("type_confidence", 0.0)
    queue_conf = state.get("queue_confidence", 0.0)
    prio_conf = state.get("priority_confidence", 0.0)
    return {"confidence": round(0.40 * type_conf + 0.35 * queue_conf + 0.25 * prio_conf, 2)}


# ---------------------------------------------------------------------------
# 4. Safety gate
# ---------------------------------------------------------------------------
def safety_gate_node(state: SentinelState) -> Dict[str, Any]:
    """Applies every escalation rule (see safety_gate.py) and records one shared decision."""
    unavailable = [task for task in ("type", "queue", "priority") if state.get(f"{task}_model_error")]
    try:
        decision = evaluate_safety_gate(
            ticket_type=state.get("type"),
            priority=state.get("priority"),
            confidence=state.get("confidence"),
            unavailable_classifiers=unavailable,
            ticket_text=ticket_text(state.get("subject"), state.get("body")),
            precedents=state.get("retrieved_precedents", []),
            retrieval_error=state.get("retrieval_error"),
        )
    except Exception as error:  # Fail safe: a broken gate must never auto-draft
        return {
            "gate_route": "escalate",
            "escalation_status": "ESCALATE",
            "escalation_reason": f"Safety gate error ({error}). Flagged for human review.",
            "escalation_triggers": ["gate_error"],
        }
    return {
        "gate_route": decision.route,
        "escalation_status": decision.status,
        "escalation_reason": decision.reason,
        "escalation_triggers": list(decision.triggers),
        "top_similarity": decision.top_similarity,
        "unsupported_requests": list(decision.unsupported),
    }


def route_after_gate(state: SentinelState) -> str:
    """Conditional edge: follow the gate's decision; escalate if it is missing."""
    return "auto_draft" if state.get("gate_route") == "auto_draft" else "escalate"


# ---------------------------------------------------------------------------
# 5a. Escalation path
# ---------------------------------------------------------------------------
def human_escalation_node(state: SentinelState) -> Dict[str, Any]:
    """Holding reply for the customer (no claims, no promises) plus an internal
    handoff note listing the real retrieved precedents for the on-call engineer."""
    lang = state.get("detected_language", "en")
    incident = HIGH_PRIORITY_INCIDENT in (state.get("escalation_triggers") or [])
    precedents = state.get("retrieved_precedents", [])

    if lang == "de":
        status = ("Ihre Meldung wurde als dringender Vorfall eingestuft und sofort an unser "
                  "Bereitschafts-Ingenieurteam weitergeleitet." if incident else
                  "Ihre Anfrage wurde zur Prüfung an einen Support-Spezialisten weitergeleitet.")
        draft = (f"Guten Tag,\n\nvielen Dank für Ihre Nachricht. {status} Wir melden uns so schnell wie "
                 f"möglich bei Ihnen.\n\nMit freundlichen Grüßen\nIhr SentinelDesk Support-Team")
    else:
        status = ("Your report has been classified as a high-priority incident and escalated immediately "
                  "to our on-call engineering team." if incident else
                  "Your request has been forwarded to a support specialist for review.")
        draft = (f"Hello,\n\nThank you for contacting us. {status} We will get back to you as soon as "
                 f"possible.\n\nBest regards,\nSentinelDesk Support Team")

    lines = [f"Escalation reason: {state.get('escalation_reason', 'Flagged for human review.')}"]
    if precedents:
        lines.append("Closest historical precedents:")
        lines += [f"- [{p['ticket_id']}] similarity {p['similarity']:.2f}: {p['historical_answer'][:220]}"
                  for p in precedents]
    else:
        lines.append("No historical precedent retrieved.")

    return {
        "draft_response": draft,
        "draft_source": "escalation",
        "draft_verified": True,
        "draft_issues": [],
        "handoff_note": "\n".join(lines),
        "latency_ms": _elapsed_ms(state.get("_start_time", time.perf_counter())),
    }


# ---------------------------------------------------------------------------
# 5b. Grounded drafting path
# ---------------------------------------------------------------------------
def drafter_node(state: SentinelState, config: RunnableConfig) -> Dict[str, Any]:
    """The Gemini key can be passed per call via config={"configurable": {"gemini_api_key": ...}}
    ("" forces offline drafting); otherwise the server's GEMINI_API_KEY is used. The key is
    never written to state or to the shared process environment."""
    start = time.perf_counter()
    api_key = (config or {}).get("configurable", {}).get("gemini_api_key")
    result = generate_grounded_draft(
        state.get("subject"),
        state.get("body", ""),
        state.get("detected_language", "en"),
        state.get("retrieved_precedents", []),
        api_key=api_key,
    )
    # escalation_status / escalation_reason were already set by the safety gate
    return {
        "draft_response": result.text,
        "draft_source": result.source,
        "draft_verified": result.verified,
        "draft_issues": result.issues,
        "draft_rejected": result.rejected_drafts,
        "llm_error": result.llm_error or "",
        "pii_masked": result.pii_masked,
        "draft_ms": _elapsed_ms(start),
        "latency_ms": _elapsed_ms(state.get("_start_time", time.perf_counter())),
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_sentinel_graph():
    builder = StateGraph(SentinelState)

    builder.add_node("detect_language", detect_language_node)
    builder.add_node("classify_type", classify_type_node)
    builder.add_node("classify_queue", classify_queue_node)
    builder.add_node("classify_priority", classify_priority_node)
    builder.add_node("retrieve", retrieval_node)
    builder.add_node("merge_triage", merge_triage_node)
    builder.add_node("safety_gate", safety_gate_node)
    builder.add_node("escalate", human_escalation_node)
    builder.add_node("draft", drafter_node)

    builder.add_edge(START, "detect_language")

    # Fan-out: 3 classifiers + precedent retrieval run in parallel
    for node in ("classify_type", "classify_queue", "classify_priority", "retrieve"):
        builder.add_edge("detect_language", node)
        builder.add_edge(node, "merge_triage")

    builder.add_edge("merge_triage", "safety_gate")
    builder.add_conditional_edges(
        "safety_gate",
        route_after_gate,
        {"escalate": "escalate", "auto_draft": "draft"},
    )
    builder.add_edge("escalate", END)
    builder.add_edge("draft", END)
    return builder.compile()


if __name__ == "__main__":
    app = build_sentinel_graph()
    for subject, body in [
        ("Platform outage", "Critical database cluster failure. Hospital telemedicine systems down."),
        ("Frage zu SEPA", "Guten Tag, wir möchten wissen, wie wir die SEPA-Lastschrift aktivieren können."),
    ]:
        result = app.invoke({"subject": subject, "body": body})
        print(f"\n[{result['detected_language']}] {subject}")
        print(f"  type={result['type']} ({result['type_confidence']:.2f})  queue={result['queue']} "
              f"({result['queue_confidence']:.2f})  priority={result['priority']} ({result['priority_confidence']:.2f})")
        print(f"  {result['escalation_status']}: {result['escalation_reason']}")
        print(f"  precedents: {[(p['ticket_id'], p['similarity']) for p in result.get('retrieved_precedents', [])]}")
        print(f"  draft ({result.get('draft_source')}, verified={result.get('draft_verified')}):\n{result['draft_response']}")
        print(f"  latency {result['latency_ms']} ms")

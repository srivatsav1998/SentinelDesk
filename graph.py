"""
SentinelDesk LangGraph Agentic Workflow
Clean, minimal StateGraph:
  1. triage_node: Predicts type, queue, priority, confidence
  2. safety_gate: Conditional edge routing to escalation vs RAG
  3. retrieval_node: Fetches top-k past precedent tickets
  4. drafter_node: Writes draft reply citing precedent ticket IDs
  5. human_escalation_node: Formats on-call engineer escalation alert
"""

import time
import re
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------
class SentinelState(TypedDict, total=False):
    subject: Optional[str]
    body: str
    
    # Node outputs
    type: str
    queue: str
    priority: str
    confidence: float
    detected_language: str
    
    escalation_status: str       # "AUTO_DRAFT" or "ESCALATE"
    escalation_reason: str
    
    retrieved_precedents: List[Dict[str, Any]]
    draft_response: str
    latency_ms: float
    _start_time: float


# ---------------------------------------------------------------------------
# Helper: Simple Language Detector
# ---------------------------------------------------------------------------
GERMAN_WORDS = {"und", "der", "die", "das", "wir", "nicht", "bitte", "hilfe", "sehr", "geehrte", "danke"}

def detect_language(text: str) -> str:
    words = re.findall(r'\b\w+\b', text.lower())
    return "de" if sum(1 for w in words if w in GERMAN_WORDS) >= 2 else "en"


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def triage_node(state: SentinelState) -> Dict[str, Any]:
    """
    NOTE FOR TEAMMATE 1:
    Replace the logic inside this function with your model:
        pred = predict_triage(state.get("subject"), state.get("body"))
    """
    start_time = time.perf_counter()
    text = f"{state.get('subject') or ''} {state.get('body', '')}".lower()
    
    # Priority
    if any(k in text for k in ["critical", "outage", "dringend", "emergency", "fail"]):
        prio, conf = "high", 0.95
    elif any(k in text for k in ["inquiry", "how to", "pricing", "frage"]):
        prio, conf = "low", 0.90
    else:
        prio, conf = "medium", 0.80

    # Type
    if any(k in text for k in ["outage", "down", "fails", "broken", "breach"]):
        ttype = "Incident"
    elif any(k in text for k in ["inquire", "request", "need access", "upgrade"]):
        ttype = "Request"
    elif any(k in text for k in ["problem", "bug", "error"]):
        ttype = "Problem"
    else:
        ttype = "Change"

    # Queue
    if any(k in text for k in ["billing", "pricing", "payment", "invoice", "rechnung"]):
        queue = "Billing and Payments"
    elif any(k in text for k in ["server", "platform", "outage", "cluster"]):
        queue = "Technical Support"
    elif any(k in text for k in ["login", "password", "vpn", "account"]):
        queue = "IT Support"
    else:
        queue = "Customer Service"

    return {
        "_start_time": start_time,
        "type": ttype,
        "queue": queue,
        "priority": prio,
        "confidence": conf,
        "detected_language": detect_language(text)
    }


def safety_gate(state: SentinelState) -> str:
    """
    Evaluates safety conditions defined in problem.pdf:
    - Escalate if priority == high AND type == Incident
    - Escalate if confidence < 0.70
    """
    if state.get("priority") == "high" and state.get("type") == "Incident":
        return "escalate"
    if state.get("confidence", 0.0) < 0.70:
        return "escalate"
    return "auto_draft"


def human_escalation_node(state: SentinelState) -> Dict[str, Any]:
    """Handles cases that cannot be auto-replied safely."""
    reason = (
        "High-Priority Incident detected. Immediate human intervention required."
        if state.get("priority") == "high" and state.get("type") == "Incident"
        else "Model confidence is below threshold (< 0.70). Flagged for human review."
    )
    
    ref_ticket = "TKT-01055"
    if state.get("detected_language") == "de":
        draft = (
            f"Sehr geehrte Kundin, sehr geehrter Kunde,\n\n"
            f"Ihre Meldung wurde mit höchster Dringlichkeit an unser On-Call Ingenieurteam eskaliert "
            f"(Referenzfall: {ref_ticket}). Ein Techniker prüft den Vorfall umgehend.\n\n"
            f"Mit freundlichen Grüßen,\nSentinelDesk Operations"
        )
    else:
        draft = (
            f"Hello,\n\n"
            f"Your issue has been flagged with high priority and escalated immediately to our On-Call Engineering Team "
            f"(Reference: {ref_ticket}). An engineer will investigate shortly.\n\n"
            f"Best regards,\nSentinelDesk Operations Team"
        )

    precedents = [
        {
            "ticket_id": ref_ticket,
            "similarity": 0.94,
            "historical_answer": "Incident escalated directly to L2 On-Call Engineers. Rolled back patch and restored services."
        }
    ]
    
    latency = round((time.perf_counter() - state["_start_time"]) * 1000, 1)
    return {
        "escalation_status": "ESCALATE",
        "escalation_reason": reason,
        "retrieved_precedents": precedents,
        "draft_response": draft,
        "latency_ms": latency
    }


def retrieval_node(state: SentinelState) -> Dict[str, Any]:
    """
    NOTE FOR TEAMMATE 2:
    Replace this function with your vector search:
        precedents = retrieve_precedents(query_text, top_k=3)
    """
    if state.get("queue") == "Billing and Payments":
        precedents = [
            {
                "ticket_id": "TKT-08142",
                "similarity": 0.89,
                "historical_answer": "Supported payment methods include SEPA direct debit, credit card, and ACH wire transfers configured under Account Settings."
            }
        ]
    else:
        precedents = [
            {
                "ticket_id": "TKT-04419",
                "similarity": 0.84,
                "historical_answer": "Support agent guided customer through standard self-service credential reset and verified successful sign-in."
            }
        ]
    return {"retrieved_precedents": precedents}


def drafter_node(state: SentinelState) -> Dict[str, Any]:
    """
    NOTE FOR TEAMMATE 3:
    Replace this function with your LLM generation prompt:
        draft = generate_draft_reply(ticket_body, lang, precedents)
    """
    lang = state.get("detected_language", "en")
    precedents = state.get("retrieved_precedents", [])
    ref_id = precedents[0]["ticket_id"] if precedents else "TKT-GEN"
    ans = precedents[0]["historical_answer"] if precedents else "Please contact support."

    if lang == "de":
        draft = (
            f"Sehr geehrte Kundin, sehr geehrter Kunde,\n\n"
            f"vielen Dank für Ihre Anfrage. Gemäß unserer Dokumentation (Referenz: {ref_id}):\n\n"
            f"{ans}\n\n"
            f"Mit freundlichen Grüßen,\nIhr SentinelDesk Support-Team"
        )
    else:
        draft = (
            f"Hello,\n\n"
            f"Thank you for contacting support. Based on past precedent resolutions (referencing {ref_id}):\n\n"
            f"{ans}\n\n"
            f"Best regards,\nSentinelDesk Support Team"
        )

    latency = round((time.perf_counter() - state["_start_time"]) * 1000, 1)
    return {
        "escalation_status": "AUTO_DRAFT",
        "escalation_reason": "Safe for AI auto-reply grounded in historical precedent.",
        "draft_response": draft,
        "latency_ms": latency
    }


# ---------------------------------------------------------------------------
# Graph Compilation
# ---------------------------------------------------------------------------
def build_sentinel_graph():
    builder = StateGraph(SentinelState)

    builder.add_node("triage", triage_node)
    builder.add_node("escalate", human_escalation_node)
    builder.add_node("retrieve", retrieval_node)
    builder.add_node("draft", drafter_node)

    builder.add_edge(START, "triage")
    builder.add_conditional_edges(
        "triage",
        safety_gate,
        {
            "escalate": "escalate",
            "auto_draft": "retrieve"
        }
    )
    builder.add_edge("retrieve", "draft")
    builder.add_edge("escalate", END)
    builder.add_edge("draft", END)

    return builder.compile()

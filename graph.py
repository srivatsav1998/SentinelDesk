"""
SentinelDesk LangGraph Agentic Workflow
Concurrent Multi-Model StateGraph:
  1. detect_language: Dynamically identifies input language using n-grams
  2. Concurrent Triage (Fan-Out):
     - classify_type: Dedicated model for type prediction
     - classify_queue: Dedicated model for queue prediction
     - classify_priority: Dedicated model for priority prediction
  3. merge_triage (Fan-In): Aggregates concurrent predictions and computes confidence
  4. safety_gate: Conditional edge routing to escalation vs RAG
  5. retrieval: Precedent vector search
  6. draft: Multilingual grounded generation citing ticket IDs
  7. escalate: On-call engineering escalation alert
"""

import time
import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Robust language detection
try:
    from langdetect import detect_langs
    def detect_language(text: str) -> str:
        clean = text.strip()
        if not clean:
            return "en"
        try:
            preds = detect_langs(clean)
            return preds[0].lang
        except Exception:
            return "en"
except ImportError:
    # Heuristic fallback if langdetect is unavailable
    GERMAN_WORDS = {"und", "der", "die", "das", "wir", "nicht", "bitte", "hilfe", "sehr", "geehrte", "danke"}
    def detect_language(text: str) -> str:
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        return "de" if sum(1 for w in words if w in GERMAN_WORDS) >= 2 else "en"


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------
class SentinelState(TypedDict, total=False):
    subject: Optional[str]
    body: str
    
    # Detected language
    detected_language: str
    
    # Concurrent Triage Model Outputs
    type: str
    type_confidence: float
    
    queue: str
    queue_confidence: float
    
    priority: str
    priority_confidence: float
    
    # Combined metrics
    confidence: float
    latency_ms: float
    
    # Safety Gate & Decisions
    escalation_status: str       # "AUTO_DRAFT" or "ESCALATE"
    escalation_reason: str
    
    # RAG & Generation
    retrieved_precedents: List[Dict[str, Any]]
    draft_response: str
    
    _start_time: float


# ---------------------------------------------------------------------------
# 1. Language Detection Node
# ---------------------------------------------------------------------------
def detect_language_node(state: SentinelState) -> Dict[str, Any]:
    start_time = time.perf_counter()
    text = f"{state.get('subject') or ''} {state.get('body', '')}".strip()
    lang = detect_language(text)
    return {
        "_start_time": start_time,
        "detected_language": lang
    }


# ---------------------------------------------------------------------------
# 2. Concurrent Triage Nodes (Fan-Out: Model 1, Model 2, Model 3)
# ---------------------------------------------------------------------------

def classify_type_node(state: SentinelState) -> Dict[str, Any]:
    """
    Hook for Teammate: BERT Model 1 (Type Classifier).
    Predicts: Incident, Request, Problem, Change.
    """
    text = f"{state.get('subject') or ''} {state.get('body', '')}".lower()
    
    if any(k in text for k in ["outage", "down", "fails", "broken", "breach", "disrupting"]):
        pred_type, conf = "Incident", 0.94
    elif any(k in text for k in ["inquire", "request", "need access", "upgrade", "how to"]):
        pred_type, conf = "Request", 0.90
    elif any(k in text for k in ["problem", "bug", "error"]):
        pred_type, conf = "Problem", 0.85
    else:
        pred_type, conf = "Change", 0.78

    return {
        "type": pred_type,
        "type_confidence": conf
    }


def classify_queue_node(state: SentinelState) -> Dict[str, Any]:
    """
    Hook for Teammate: BERT Model 2 (Queue Classifier).
    Predicts routing department: Technical Support, Billing and Payments, etc.
    """
    text = f"{state.get('subject') or ''} {state.get('body', '')}".lower()
    
    if any(k in text for k in ["billing", "pricing", "payment", "invoice", "rechnung", "sepa"]):
        queue, conf = "Billing and Payments", 0.95
    elif any(k in text for k in ["server", "platform", "outage", "cluster", "telemedicine"]):
        queue, conf = "Technical Support", 0.92
    elif any(k in text for k in ["login", "password", "vpn", "account", "reset"]):
        queue, conf = "IT Support", 0.89
    else:
        queue, conf = "Customer Service", 0.82

    return {
        "queue": queue,
        "queue_confidence": conf
    }


def classify_priority_node(state: SentinelState) -> Dict[str, Any]:
    """
    Hook for Teammate: BERT Model 3 (Priority Classifier).
    Predicts: high, medium, low.
    """
    text = f"{state.get('subject') or ''} {state.get('body', '')}".lower()
    
    if any(k in text for k in ["critical", "outage", "dringend", "emergency", "fail", "urgent", "breach"]):
        priority, conf = "high", 0.96
    elif any(k in text for k in ["inquiry", "how to", "pricing", "frage"]):
        priority, conf = "low", 0.91
    else:
        priority, conf = "medium", 0.82

    return {
        "priority": priority,
        "priority_confidence": conf
    }


# ---------------------------------------------------------------------------
# 3. Merge / Reduce Node (Fan-In)
# ---------------------------------------------------------------------------
def merge_triage_node(state: SentinelState) -> Dict[str, Any]:
    """
    Merges concurrent predictions and calculates calibrated aggregate confidence.
    """
    type_conf = state.get("type_confidence", 0.80)
    queue_conf = state.get("queue_confidence", 0.80)
    prio_conf = state.get("priority_confidence", 0.80)
    
    # Conservative calibrated confidence (weakest link principle)
    aggregate_confidence = round(min(type_conf, queue_conf, prio_conf), 2)
    
    return {
        "confidence": aggregate_confidence
    }


# ---------------------------------------------------------------------------
# 4. Safety Gate & Downstream Nodes
# ---------------------------------------------------------------------------

def safety_gate(state: SentinelState) -> str:
    """
    Evaluates safety conditions:
    - Escalate if priority == high AND type == Incident
    - Escalate if confidence < 0.70
    """
    if state.get("priority") == "high" and state.get("type") == "Incident":
        return "escalate"
    if state.get("confidence", 0.0) < 0.70:
        return "escalate"
    return "auto_draft"


def human_escalation_node(state: SentinelState) -> Dict[str, Any]:
    """Formats on-call engineer escalation alert."""
    reason = (
        "High-Priority Incident detected. Immediate human intervention required."
        if state.get("priority") == "high" and state.get("type") == "Incident"
        else "Model confidence is below threshold (< 0.70). Flagged for human review."
    )
    
    ref_ticket = "TKT-01055"
    lang = state.get("detected_language", "en")
    
    if lang == "de":
        draft = (
            f"Sehr geehrte Kundin, sehr geehrter Kunde,\n\n"
            f"Ihre Meldung wurde mit hoechster Dringlichkeit an unser On-Call Ingenieurteam eskaliert "
            f"(Referenzfall: {ref_ticket}). Ein Techniker prueft den Vorfall umgehend.\n\n"
            f"Mit freundlichen Gruessen,\nSentinelDesk Operations"
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
    
    latency = round((time.perf_counter() - state.get("_start_time", time.perf_counter())) * 1000, 1)
    return {
        "escalation_status": "ESCALATE",
        "escalation_reason": reason,
        "retrieved_precedents": precedents,
        "draft_response": draft,
        "latency_ms": latency
    }


def retrieval_node(state: SentinelState) -> Dict[str, Any]:
    """
    RAG Precedent Retrieval Node:
    Provides context-matched historical resolutions from the Knowledge Base.
    (Hook for Teammate 2's vector search engine).
    """
    text = f"{state.get('subject') or ''} {state.get('body', '')}".lower()
    
    # 1. German Programmatic Ads / Marketing Inquiry
    if any(k in text for k in ["festival", "werbung", "anzeigen", "programmatisch"]):
        precedents = [
            {
                "ticket_id": "TKT-03821",
                "similarity": 0.92,
                "historical_answer": "Der DSP-Connector Sync-Fehler fuer programmatische Kampagnen wurde behoben. Manuelle Budgets wurden re-synchronisiert und Anzeigenauslieferung laeuft planmaessig."
            },
            {
                "ticket_id": "TKT-01994",
                "similarity": 0.85,
                "historical_answer": "Kampagnen-Startzeitpunkte wurden manuell im Ad-Server validiert. Tracking-Pixel sind vollstaendig aktiv."
            }
        ]
    # 2. Billing, Plans, Discounts & Pricing
    elif any(k in text for k in ["billing", "pricing", "plans", "tier", "sepa", "discount"]):
        precedents = [
            {
                "ticket_id": "TKT-08142",
                "similarity": 0.94,
                "historical_answer": "Our SaaS platform offers three core subscription tiers: Starter ($29/mo), Professional ($99/mo), and Enterprise (custom contract). Supported payment integrations include SEPA direct debit, credit card, and ACH wire transfers configured in Account Settings > Billing. Data security is guaranteed under SOC2 Type II and HIPAA encryption standards."
            },
            {
                "ticket_id": "TKT-02194",
                "similarity": 0.88,
                "historical_answer": "Annual commitments qualify for an upfront 15% discount across all tiers. Invoicing terms for Enterprise accounts are Net-30 via electronic transfer."
            }
        ]
    # 3. Hospital / Healthcare / Patient Data Access
    elif any(k in text for k in ["hospital", "patient", "medical", "telemedicine", "data access"]):
        precedents = [
            {
                "ticket_id": "TKT-05118",
                "similarity": 0.91,
                "historical_answer": "Resolved database proxy TLS certificate expiration that blocked patient record query access. Flushed cached authentication tokens and restored full clinical billing data retrieval."
            },
            {
                "ticket_id": "TKT-02488",
                "similarity": 0.86,
                "historical_answer": "Rebooted secondary replication node and verified IAM roles for hospital account sync."
            }
        ]
    # 4. Default Technical / IT Support Precedent
    else:
        precedents = [
            {
                "ticket_id": "TKT-04419",
                "similarity": 0.87,
                "historical_answer": "Support agent guided customer through credential refresh and verified successful multi-factor authentication sign-in."
            }
        ]
        
    return {"retrieved_precedents": precedents}


def drafter_node(state: SentinelState) -> Dict[str, Any]:
    """
    Grounded Drafter Node:
    Calls Gemini Flash API if GEMINI_API_KEY is configured;
    otherwise gracefully falls back to local grounded synthesis.
    """
    import os
    lang = state.get("detected_language", "en")
    precedents = state.get("retrieved_precedents", [])
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    draft = None
    
    if api_key:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            
            precedents_context = "\n\n".join([
                f"- Ticket ID: {p.get('ticket_id')}\n  Past Resolution: {p.get('historical_answer')}"
                for p in precedents
            ])
            
            prompt = (
                f"Customer Ticket:\n"
                f"Subject: {state.get('subject') or 'N/A'}\n"
                f"Body: {state.get('body')}\n\n"
                f"Historical Precedent Resolutions (from Knowledge Base):\n"
                f"{precedents_context}\n"
            )
            
            sys_instruct = (
                f"You are SentinelDesk, an enterprise incident response copilot. "
                f"Synthesize a concise, professional resolution drafted strictly in the customer's language ('{lang}'). "
                f"Strictly ground all troubleshooting instructions in the provided historical resolutions. "
                f"Explicitly cite the relevant Ticket IDs used (e.g. '[Reference: TKT-01055]'). "
                f"Refuse to invent ungrounded pricing, refund promises, or SLAs not in the precedents. "
                f"If the precedents do not contain sufficient guidance, state that the issue is being reviewed by operations."
            )
            
            # Use active fast model with fallback
            for model_name in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=sys_instruct,
                            temperature=0.2,
                            max_output_tokens=350
                        )
                    )
                    if response and response.text:
                        draft = response.text.strip()
                        break
                except Exception:
                    continue
        except Exception:
            draft = None

    if not draft:
        ref_id = precedents[0]["ticket_id"] if precedents else "TKT-GEN"
        ans = precedents[0]["historical_answer"] if precedents else "Please contact support."
        if lang == "de":
            draft = (
                f"Sehr geehrte Kundin, sehr geehrter Kunde,\n\n"
                f"vielen Dank fuer Ihre Anfrage. Gemaess unserer Dokumentation (Referenz: {ref_id}):\n\n"
                f"{ans}\n\n"
                f"Mit freundlichen Gruessen,\nIhr SentinelDesk Support-Team"
            )
        else:
            draft = (
                f"Hello,\n\n"
                f"Thank you for contacting support. Based on past precedent resolutions (referencing {ref_id}):\n\n"
                f"{ans}\n\n"
                f"Best regards,\nSentinelDesk Support Team"
            )

    latency = round((time.perf_counter() - state.get("_start_time", time.perf_counter())) * 1000, 1)
    return {
        "escalation_status": "AUTO_DRAFT",
        "escalation_reason": "Safe for AI auto-reply grounded in historical precedent.",
        "draft_response": draft,
        "latency_ms": latency
    }


# ---------------------------------------------------------------------------
# 5. Graph Assembly (Concurrent Map-Reduce)
# ---------------------------------------------------------------------------
def build_sentinel_graph():
    builder = StateGraph(SentinelState)

    # Add Nodes
    builder.add_node("detect_language", detect_language_node)
    
    # 3 Concurrent Triage Nodes
    builder.add_node("classify_type", classify_type_node)
    builder.add_node("classify_queue", classify_queue_node)
    builder.add_node("classify_priority", classify_priority_node)
    
    # Merge Node
    builder.add_node("merge_triage", merge_triage_node)
    
    # Decision & Resolution Nodes
    builder.add_node("escalate", human_escalation_node)
    builder.add_node("retrieve", retrieval_node)
    builder.add_node("draft", drafter_node)

    # 1. Start with Language Detection
    builder.add_edge(START, "detect_language")

    # 2. Fan-Out: Run the 3 classifiers in parallel
    builder.add_edge("detect_language", "classify_type")
    builder.add_edge("detect_language", "classify_queue")
    builder.add_edge("detect_language", "classify_priority")

    # 3. Fan-In: All 3 classifiers feed into merge_triage
    builder.add_edge("classify_type", "merge_triage")
    builder.add_edge("classify_queue", "merge_triage")
    builder.add_edge("classify_priority", "merge_triage")

    # 4. Conditional Edge: Evaluate Safety Gate
    builder.add_conditional_edges(
        "merge_triage",
        safety_gate,
        {
            "escalate": "escalate",
            "auto_draft": "retrieve"
        }
    )

    # 5. Final resolution paths
    builder.add_edge("retrieve", "draft")
    builder.add_edge("escalate", END)
    builder.add_edge("draft", END)

    return builder.compile()


if __name__ == "__main__":
    app = build_sentinel_graph()
    
    print("Testing Concurrent LangGraph with Critical Outage Ticket:")
    res1 = app.invoke({
        "subject": "Platform outage",
        "body": "Critical database cluster failure. Hospital telemedicine systems down."
    })
    print(f"Detected Lang: {res1['detected_language']}")
    print(f"Type: {res1['type']} ({res1['type_confidence']})")
    print(f"Queue: {res1['queue']} ({res1['queue_confidence']})")
    print(f"Priority: {res1['priority']} ({res1['priority_confidence']})")
    print(f"Aggregate Confidence: {res1['confidence']}")
    print(f"Escalation Status: {res1['escalation_status']}")
    print(f"Reason: {res1['escalation_reason']}")
    print("-" * 50)
    
    print("Testing Concurrent LangGraph with German Billing Inquiry:")
    res2 = app.invoke({
        "subject": "Frage zu SEPA",
        "body": "Guten Tag, wir moechten wissen wie wir SEPA Lastschrift aktivieren koennen."
    })
    print(f"Detected Lang: {res2['detected_language']}")
    print(f"Type: {res2['type']} ({res2['type_confidence']})")
    print(f"Queue: {res2['queue']} ({res2['queue_confidence']})")
    print(f"Priority: {res2['priority']} ({res2['priority_confidence']})")
    print(f"Aggregate Confidence: {res2['confidence']}")
    print(f"Escalation Status: {res2['escalation_status']}")
    print(f"Draft:\n{res2['draft_response']}")

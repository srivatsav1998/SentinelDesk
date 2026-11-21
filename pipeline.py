"""
SentinelDesk Pipeline Interface
Unified entry point connecting Classifier, Vector Search (RAG), Escalation Gate, and Generator.
This allows the UI developer to build and test the full UI without being blocked by individual model training.
"""

import time
import os
import re
from typing import List, Dict, Any, Optional

# Basic heuristic language detector (English vs German)
GERMAN_STOPWORDS = {
    "und", "der", "die", "das", "wir", "nicht", "bitte", "problem", "hilfe",
    "sehr", "geehrte", "dringend", "konto", "zugang", "danke", "ist", "mit", "für"
}

def detect_language(text: str) -> str:
    """Detects whether text is German ('de') or English ('en')."""
    words = re.findall(r'\b\w+\b', text.lower())
    german_matches = sum(1 for w in words if w in GERMAN_STOPWORDS)
    if german_matches >= 2:
        return "de"
    return "en"

def process_ticket(
    subject: Optional[str],
    body: str,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Main triage and resolution pipeline.
    
    Args:
        subject: Ticket subject line (nullable)
        body: Full ticket content
        top_k: Number of historical precedent tickets to retrieve
        
    Returns:
        Dict containing:
            - type: str
            - queue: str
            - priority: str
            - confidence: float
            - detected_language: str
            - latency_ms: float
            - escalation_status: 'AUTO_DRAFT' | 'ESCALATE'
            - escalation_reason: str
            - retrieved_precedents: List[Dict]
            - draft_response: str
    """
    start_time = time.perf_counter()
    
    safe_subject = (subject or "").strip()
    full_text = f"{safe_subject}\n{body}".strip()
    detected_lang = detect_language(full_text)
    
    # ---------------------------------------------------------
    # 1. TRIAGE CLASSIFICATION (Hook for Person 1)
    # Defaulting to robust rule/heuristic until ML model file is present
    # ---------------------------------------------------------
    text_lower = full_text.lower()
    
    # Priority Heuristic
    if any(k in text_lower for k in ["critical", "outage", "disrupting", "dringend", "emergency", "breach"]):
        pred_priority = "high"
        priority_conf = 0.94
    elif any(k in text_lower for k in ["inquiry", "how to", "frage", "pricing", "plans"]):
        pred_priority = "low"
        priority_conf = 0.88
    else:
        pred_priority = "medium"
        priority_conf = 0.81
        
    # Type Heuristic
    if any(k in text_lower for k in ["outage", "broken", "down", "fails", "persists", "breach"]):
        pred_type = "Incident"
        type_conf = 0.92
    elif any(k in text_lower for k in ["how to", "inquire", "request", "need access", "upgrade"]):
        pred_type = "Request"
        type_conf = 0.89
    elif any(k in text_lower for k in ["problem", "bug", "error"]):
        pred_type = "Problem"
        type_conf = 0.84
    else:
        pred_type = "Incident"
        type_conf = 0.75
        
    # Queue Heuristic
    if any(k in text_lower for k in ["billing", "pricing", "payment", "invoice", "refund", "rechnung"]):
        pred_queue = "Billing and Payments"
    elif any(k in text_lower for k in ["server", "platform", "outage", "service", "infrastructure"]):
        pred_queue = "Technical Support"
    elif any(k in text_lower for k in ["password", "login", "vpn", "access", "account"]):
        pred_queue = "IT Support"
    else:
        pred_queue = "Customer Service"
        
    confidence = round(min(priority_conf, type_conf), 2)
    
    # ---------------------------------------------------------
    # 2. ESCALATION & SAFETY GATE (Hook for Person 3)
    # Rules specified in problem.pdf:
    # - Immediate escalation if priority == high and type == Incident
    # - Escalation if confidence is below calibrated threshold (< 0.70)
    # ---------------------------------------------------------
    escalate = False
    escalation_reason = ""
    
    if pred_priority == "high" and pred_type == "Incident":
        escalate = True
        escalation_reason = "Immediate Human Escalation: High-Priority Incident detected (Service disruption / outage risk)."
    elif confidence < 0.70:
        escalate = True
        escalation_reason = "Human Review Required: Model confidence below safety threshold (confidence < 0.70)."
    else:
        escalation_reason = "Auto-Draft Ready: Safe for AI resolution with historical precedent grounding."
        
    # ---------------------------------------------------------
    # 3. RETRIEVAL & HISTORICAL PRECEDENTS (Hook for Person 2)
    # Simulated realistic precedents from the 80% KB
    # ---------------------------------------------------------
    if pred_queue == "Billing and Payments":
        retrieved_precedents = [
            {
                "ticket_id": "TKT-08142",
                "similarity": 0.89,
                "subject": "Enterprise Subscription & Billing Plans Overview",
                "historical_answer": "Our SaaS platform offers three core billing tiers: Starter ($29/mo), Professional ($99/mo), and Enterprise (custom invoice). Security compliance includes SOC2 Type II and HIPAA encryption. Annual commitments receive a 15% discount."
            },
            {
                "ticket_id": "TKT-02194",
                "similarity": 0.83,
                "subject": "Invoice Breakdown and Payment Integration Inquiry",
                "historical_answer": "We support Stripe, ACH transfer, and direct SEPA invoicing. Integration can be configured in Account Settings > Billing > Webhooks."
            }
        ]
    elif pred_priority == "high":
        retrieved_precedents = [
            {
                "ticket_id": "TKT-01055",
                "similarity": 0.93,
                "subject": "Critical Cluster Outage on US-East Node",
                "historical_answer": "Incident managed by On-Call Infrastructure Team. Service was restored by rolling back kernel update v4.2.1 and performing graceful pod restarts. Post-mortem documented in Jira INC-882."
            },
            {
                "ticket_id": "TKT-03481",
                "similarity": 0.85,
                "subject": "Disrupted Access After Scheduled Patching",
                "historical_answer": "Traffic was rerouted to secondary gateway while database pool connections were flushed. Escalated directly to Level 3 site reliability engineers."
            }
        ]
    else:
        retrieved_precedents = [
            {
                "ticket_id": "TKT-04419",
                "similarity": 0.86,
                "subject": "Standard In-Scope Support Procedure",
                "historical_answer": "Agent provided guided walkthrough from documentation section 4.1. Verified resolution with customer after credential refresh."
            }
        ]

    # ---------------------------------------------------------
    # 4. GROUNDED DRAFT GENERATION (Hook for Person 3)
    # Multi-lingual generation referencing specific Ticket IDs
    # ---------------------------------------------------------
    primary_ref = retrieved_precedents[0]["ticket_id"]
    if escalate:
        if detected_lang == "de":
            draft_response = (
                f"Sehr geehrte Kundin, sehr geehrter Kunde,\n\n"
                f"wir haben Ihre Meldung mit hoher Priorität aufgenommen. Da es sich um einen kritischen Vorfall handelt, "
                f"wurde Ihr Ticket sofort an unser Bereitschafts-Ingenieurteam weitergeleitet (Referenzvorfall: {primary_ref}). "
                f"Ein Techniker wird sich schnellstmöglich mit Ihnen in Verbindung setzen.\n\n"
                f"Mit freundlichen Grüßen,\nIhr SentinelDesk Support-Team"
            )
        else:
            draft_response = (
                f"Hello,\n\n"
                f"Thank you for contacting technical operations. Because this involves a critical incident requiring immediate on-call intervention, "
                f"your ticket has been escalated directly to our Level 2 Engineering Team based on incident protocol {primary_ref}.\n\n"
                f"An engineer has been paged and will investigate immediately.\n\n"
                f"Best regards,\nSentinelDesk Operations Team"
            )
    else:
        if detected_lang == "de":
            draft_response = (
                f"Sehr geehrte Kundin, sehr geehrter Kunde,\n\n"
                f"vielen Dank für Ihre Anfrage. Basierend auf unseren dokumentierten Richtlinien (Referenz: {primary_ref}):\n"
                f"{retrieved_precedents[0]['historical_answer']}\n\n"
                f"Bitte lassen Sie uns wissen, falls Sie weitere Unterstützung benötigen.\n\n"
                f"Mit freundlichen Grüßen,\nIhr SentinelDesk Support-Team"
            )
        else:
            draft_response = (
                f"Hello,\n\n"
                f"Thank you for reaching out to SentinelDesk. Based on our precedent resolutions (specifically referencing {primary_ref}):\n\n"
                f"{retrieved_precedents[0]['historical_answer']}\n\n"
                f"Please let us know if you require any additional assistance or clarification.\n\n"
                f"Best regards,\nSentinelDesk Support Team"
            )
            
    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
    
    return {
        "type": pred_type,
        "queue": pred_queue,
        "priority": pred_priority,
        "confidence": confidence,
        "detected_language": detected_lang,
        "escalation_status": "ESCALATE" if escalate else "AUTO_DRAFT",
        "escalation_reason": escalation_reason,
        "retrieved_precedents": retrieved_precedents,
        "draft_response": draft_response,
        "latency_ms": latency_ms
    }

if __name__ == "__main__":
    test_subject = "Urgent Support Required for Project Management Platform"
    test_body = "Our project management platform is facing a critical outage. All services down."
    res = process_ticket(test_subject, test_body)
    import pprint
    pprint.pprint(res)

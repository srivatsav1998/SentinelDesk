# Integration Contracts for Team Members

This document defines the exact functions and data formats the other teammates need to produce so that their work directly slots into the **LangGraph workflow** and the **UI console**.

---

## 1. Teammate 1: Triage Classification Engineer
**File to produce**: `triage_engine.py`  
**Purpose**: Predict type, queue, and priority with a calibrated confidence score.

### Required Function Signature:
```python
def predict_triage(subject: str, body: str) -> dict:
    """
    Args:
        subject: Ticket subject (can be None or empty string)
        body: Ticket body text
        
    Returns:
        dict:
            "type": str ("Incident", "Request", "Problem", or "Change")
            "queue": str ("Technical Support", "Billing and Payments", etc.)
            "priority": str ("high", "medium", or "low")
            "confidence": float (between 0.0 and 1.0)
    """
```

---

## 2. Teammate 2: RAG & Vector Precedent Engineer
**File to produce**: `retriever.py`  
**Purpose**: Search the 80% KB (`data/kb_train.parquet`) for top-k similar historical tickets.

### Required Function Signature:
```python
def retrieve_precedents(query_text: str, top_k: int = 3) -> list[dict]:
    """
    Args:
        query_text: Combined ticket subject and body
        top_k: Number of historical tickets to return (default: 3)
        
    Returns:
        list of dicts, each containing:
            "ticket_id": str (e.g. "TKT-00123")
            "similarity": float (cosine similarity between 0.0 and 1.0)
            "historical_answer": str (the past resolution text from 'answer' column)
    """
```

---

## 3. Teammate 3: Resolution & Draft Drafter Engineer
**File to produce**: `drafter.py`  
**Purpose**: Generate a polite draft response in the user's detected language citing precedent ticket IDs.

### Required Function Signature:
```python
def generate_draft_reply(
    ticket_text: str,
    language: str,
    precedents: list[dict]
) -> str:
    """
    Args:
        ticket_text: The user's incoming ticket
        language: "en" or "de"
        precedents: The list of top-k retrieved tickets from retriever.py
        
    Returns:
        str: A drafted response strictly citing the retrieved ticket_ids
             (e.g. "Based on past resolution TKT-01055...").
    """
```

---

## 4. Integration Point in LangGraph (`graph.py`)
Once your teammates provide these 3 functions, they replace the temporary placeholders in `graph.py`:
1. `triage_node` calls `predict_triage(subject, body)`
2. `retrieval_node` calls `retrieve_precedents(text, top_k)`
3. `drafter_node` calls `generate_draft_reply(text, lang, precedents)`

Everything else (state passing, escalation condition, UI) will just work without any rewrites.

# Integration Contracts for Team Members

This document defines the exact functions and data formats the other teammates need to produce so that their work directly slots into the **LangGraph workflow** and the **UI console**.

---

## 1. Teammate 1: Triage Classification Engineer
**File to produce**: `triage_engine.py`  
**Purpose**: Predict type, queue, and priority with calibrated confidence scores (using 3 concurrent fine-tuned BERT models).

### Option A: Three Separate Predictors (Plugs directly into concurrent LangGraph nodes)
```python
def predict_type(text: str, *, subject: str | None = None) -> tuple[str, float]:
    """Returns (predicted_type, confidence_score)"""
    # e.g. ("Incident", 0.94)

def predict_queue(text: str, *, subject: str | None = None) -> tuple[str, float]:
    """Returns (predicted_queue, confidence_score)"""
    # e.g. ("Technical Support", 0.92)

def predict_priority(text: str, *, subject: str | None = None) -> tuple[str, float]:
    """Returns (predicted_priority, confidence_score)"""
    # e.g. ("high", 0.96)
```

Pass the body as `text` and the subject separately when available. All three
functions also accept a single text argument. A missing checkpoint raises
`ModelUnavailableError` so the graph can route to human review.

### Option B: Unified Predictor
```python
def predict_triage(subject: str, body: str) -> dict:
    """
    Returns:
        dict:
            "type": str ("Incident", "Request", "Problem", or "Change")
            "type_confidence": float
            "queue": str ("Technical Support", "Billing and Payments", etc.)
            "queue_confidence": float
            "priority": str ("high", "medium", or "low")
            "priority_confidence": float
            "confidence": float (aggregate calibrated score)
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
1. The three concurrent classification nodes call `predict_type`, `predict_queue`,
   and `predict_priority` with the body and optional subject.
2. `retrieval_node` calls `retrieve_precedents(text, top_k)`
3. `drafter_node` calls `generate_draft_reply(text, lang, precedents)`

Everything else (state passing, escalation condition, UI) will just work without any rewrites.

---

## 5. Implementation Status

| Contract | File | Notes |
| :--- | :--- | :--- |
| `predict_type` / `predict_priority` | `triage_engine.py` | Fine-tuned mBERT checkpoints (Git LFS) |
| `predict_queue` | `triage_engine.py` | TF-IDF + logistic regression (`models/queue/queue_classifier.joblib`, trained by `train_queue.py`); a transformer checkpoint in `models/queue/` takes precedence |
| `retrieve_precedents` | `retriever.py` | Also returns `subject`, `language`, `queue`, `type`, `priority`; never returns tickets without an answer; model and index load once (`warm_up()`) |
| `generate_draft_reply` | `drafter.py` | Returns the text only. The graph calls `generate_grounded_draft(...)`, which also returns the source model, verifier result, rejected drafts and PII counts |
| Escalation rules | `safety_gate.py` | `evaluate_safety_gate(..., ticket_text=, precedents=, retrieval_error=)` adds the similarity, unsupported-request and prompt-injection rules |

"""SentinelDesk operator console server.

Run:   python server.py            -> http://127.0.0.1:8000
       SENTINEL_HOST=0.0.0.0 SENTINEL_PORT=8000 python server.py   (to reach it from another device)

Serves the single-page console in static/ and a small JSON API around the LangGraph
pipeline. The Gemini key stays on the server (.env); the browser never sees it.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from burst import BurstDetector  # noqa: E402
from drafter import plain_chatbot_reply, verify_draft  # noqa: E402
from explain import highlight_spans, queue_top_terms  # noqa: E402
from guardrails import mask_pii  # noqa: E402
from safety_gate import CONFIDENCE_THRESHOLD, SIMILARITY_THRESHOLD  # noqa: E402
from text_utils import ticket_text  # noqa: E402

STATIC = ROOT / "static"
PRESETS_FILE = ROOT / "data" / "sample_ui_presets.json"
HOLDOUT_FILE = ROOT / "data" / "test_holdout.parquet"
FEEDBACK_FILE = Path(os.environ.get("SENTINEL_FEEDBACK_FILE", ROOT / "data" / "feedback.jsonl"))
FINDINGS = ROOT / "findings"
MAX_TEXT = 8000

STATE: dict = {"graph": None, "error": None, "started": time.time()}
STATS = {"processed": 0, "escalated": 0, "latencies": []}
BURST = BurstDetector()
_LOCK = threading.Lock()


def _load_pipeline() -> None:
    """Build the graph and warm every model once (runs in a background thread)."""
    try:
        import graph
        import retriever

        app = graph.build_sentinel_graph()
        retriever.warm_up()
        app.invoke({"subject": "Warm-up", "body": "Login to the customer portal fails after password reset."},
                   config={"configurable": {"gemini_api_key": ""}})
        queue_top_terms("Warm-up", "invoice", "Billing and Payments")
        STATE["graph"] = app
    except Exception as error:  # surfaced through /api/health
        STATE["error"] = f"{type(error).__name__}: {error}"


@asynccontextmanager
async def lifespan(app):
    if not os.environ.get("SENTINEL_SKIP_WARMUP"):
        threading.Thread(target=_load_pipeline, daemon=True).start()
    yield


def _gemini_model() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite").split(",")[0].strip()


def _feedback_count() -> int:
    if not FEEDBACK_FILE.is_file():
        return 0
    with FEEDBACK_FILE.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
async def index(request: Request):
    return FileResponse(STATIC / "index.html")


def health(request: Request):
    latencies = STATS["latencies"]
    return JSONResponse({
        "ready": STATE["graph"] is not None,
        "error": STATE["error"],
        "gemini": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
        "model": _gemini_model(),
        "thresholds": {"confidence": CONFIDENCE_THRESHOLD, "similarity": SIMILARITY_THRESHOLD},
        "top_k": int(os.environ.get("SENTINEL_TOP_K", 3)),
        "feedback": _feedback_count(),
        "session": {
            "processed": STATS["processed"],
            "escalated": STATS["escalated"],
            "median_ms": round(statistics.median(latencies), 1) if latencies else None,
        },
    })


def presets(request: Request):
    data = json.loads(PRESETS_FILE.read_text(encoding="utf-8")) if PRESETS_FILE.is_file() else []
    return JSONResponse([
        {"name": p["name"], "subject": p.get("subject", ""), "body": p.get("body", "").replace("\\n", "\n"),
         "notes": p.get("notes", ""), "expected": p.get("expected_escalation", "")}
        for p in data
    ])


def _public_result(result: dict, wall_ms: float, subject: str, body: str) -> dict:
    precedents = [
        {k: p.get(k) for k in ("ticket_id", "similarity", "historical_answer", "subject", "language", "queue")}
        for p in result.get("retrieved_precedents", [])
    ]
    return {
        "type": result.get("type"), "type_confidence": result.get("type_confidence"),
        "priority": result.get("priority"), "priority_confidence": result.get("priority_confidence"),
        "queue": result.get("queue"), "queue_confidence": result.get("queue_confidence"),
        "confidence": result.get("confidence"),
        "language": result.get("detected_language"),
        "status": result.get("escalation_status"),
        "reason": result.get("escalation_reason"),
        "triggers": result.get("escalation_triggers") or [],
        "top_similarity": result.get("top_similarity"),
        "unsupported": result.get("unsupported_requests") or [],
        "precedents": precedents,
        "retrieval_error": result.get("retrieval_error"),
        "draft": result.get("draft_response", ""),
        "draft_source": result.get("draft_source", ""),
        "draft_verified": bool(result.get("draft_verified")),
        "draft_issues": result.get("draft_issues") or [],
        "draft_rejected": [{"model": r.get("model"), "issues": r.get("issues"), "text": r.get("text")}
                           for r in result.get("draft_rejected") or []],
        "llm_error": result.get("llm_error") or None,
        "pii_masked": result.get("pii_masked") or {},
        "handoff_note": result.get("handoff_note", ""),
        "model_errors": {t: result[f"{t}_model_error"] for t in ("type", "queue", "priority")
                         if result.get(f"{t}_model_error")},
        "timings": {
            "type": result.get("type_ms"), "priority": result.get("priority_ms"),
            "queue": result.get("queue_ms"), "retrieval": result.get("retrieval_ms"),
            "draft": result.get("draft_ms"), "graph": result.get("latency_ms"), "wall": round(wall_ms, 1),
        },
        "highlights": {"subject": highlight_spans(subject), "body": highlight_spans(body)},
    }


async def triage(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Send a JSON body with subject and body."}, status_code=400)
    subject = str(payload.get("subject") or "")[:500]
    body = str(payload.get("body") or "")[:MAX_TEXT]
    if not body.strip():
        return JSONResponse({"error": "Add the customer's message before triaging."}, status_code=400)
    if STATE["graph"] is None:
        status = 503 if not STATE["error"] else 500
        return JSONResponse({"error": STATE["error"] or "Models are still loading. Try again in a few seconds."},
                            status_code=status)
    from starlette.concurrency import run_in_threadpool

    return JSONResponse(await run_in_threadpool(
        _run_triage, subject, body, bool(payload.get("use_llm", True)), bool(payload.get("compare", False))))


def _run_triage(subject: str, body: str, use_llm: bool, compare: bool) -> dict:
    config = {"configurable": {"gemini_api_key": None if use_llm else ""}}
    start = time.perf_counter()
    result = STATE["graph"].invoke({"subject": subject, "body": body}, config=config)
    wall_ms = (time.perf_counter() - start) * 1000
    public = _public_result(result, wall_ms, subject, body)
    public["queue_terms"] = queue_top_terms(subject, body, public["queue"] or "")

    try:
        from retriever import embed_query

        alert = BURST.add(f"{public['type']} / {public['queue']}", subject or body[:60],
                          embed_query(ticket_text(subject, body)))
    except Exception:
        alert = None
    public["outage"] = ({"count": alert.count, "window_minutes": alert.window_minutes, "members": alert.members}
                        if alert else None)

    if compare and use_llm:
        text, error = plain_chatbot_reply(subject, body, public["language"] or "en")
        issues = (verify_draft(text, result.get("retrieved_precedents", []), ticket_text(subject, body),
                               public["language"] or "en") if text else [])
        public["plain_chatbot"] = {"text": text, "issues": issues, "error": error}

    with _LOCK:
        STATS["processed"] += 1
        STATS["escalated"] += public["status"] == "ESCALATE"
        STATS["latencies"] = (STATS["latencies"] + [public["timings"]["wall"]])[-500:]
    return public


async def feedback(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Send a JSON body."}, status_code=400)
    action = payload.get("action")
    if action not in {"approve", "edit", "escalate", "correct"}:
        return JSONResponse({"error": "action must be approve, edit, escalate or correct"}, status_code=400)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "action": action,
        # Stored masked: feedback is meant to become training data.
        "subject": mask_pii(str(payload.get("subject") or "")[:500])[0],
        "body": mask_pii(str(payload.get("body") or "")[:MAX_TEXT])[0],
        "predicted": {k: payload.get("predicted", {}).get(k) for k in ("type", "priority", "queue", "status")},
        "correction": {k: v for k, v in (payload.get("correction") or {}).items()
                       if k in {"type", "priority", "queue"} and isinstance(v, str)},
        "draft": mask_pii(str(payload.get("draft") or "")[:MAX_TEXT])[0],
    }
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, FEEDBACK_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return JSONResponse({"ok": True, "count": _feedback_count()})


def stream(request: Request):
    """Random holdout tickets for the live-inbox demo: subject and body only, never labels or answers."""
    if not HOLDOUT_FILE.is_file():
        return JSONResponse({"error": "Run python data_loader.py to create the holdout split."}, status_code=404)
    import pandas as pd

    try:
        n = max(1, min(60, int(request.query_params.get("n", 24))))
    except ValueError:
        n = 24
    frame = pd.read_parquet(HOLDOUT_FILE, columns=["ticket_id", "subject", "body"])
    rows = frame.sample(n=n, random_state=random.randint(0, 10_000))
    return JSONResponse([
        {"id": r.ticket_id, "subject": r.subject if isinstance(r.subject, str) else "", "body": r.body}
        for r in rows.itertuples()
    ])


def reset_outage(request: Request):
    BURST.reset()
    return JSONResponse({"ok": True})


def metrics(request: Request):
    def load(name: str):
        path = FINDINGS / name
        return json.loads(path.read_text()) if path.is_file() else None

    return JSONResponse({
        "holdout": load("holdout_metrics.json"),
        "tradeoff": load("threshold_tradeoff.json"),
        "queue_comparison": load("queue_model_comparison.json"),
        "thresholds": {"confidence": CONFIDENCE_THRESHOLD, "similarity": SIMILARITY_THRESHOLD},
        "feedback": _feedback_count(),
    })


routes = [
    Route("/", index),
    Route("/api/health", health),
    Route("/api/presets", presets),
    Route("/api/triage", triage, methods=["POST"]),
    Route("/api/feedback", feedback, methods=["POST"]),
    Route("/api/stream", stream),
    Route("/api/outage/reset", reset_outage, methods=["POST"]),
    Route("/api/metrics", metrics),
    Mount("/static", StaticFiles(directory=STATIC), name="static"),
]
app = Starlette(routes=routes, lifespan=lifespan)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("SENTINEL_HOST", "127.0.0.1")
    port = int(os.environ.get("SENTINEL_PORT", 8000))
    print(f"SentinelDesk console: http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")

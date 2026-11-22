"""
SentinelDesk holdout evaluation (20% unseen split, data/test_holdout.parquet).

Zero-leakage protocol
  * The holdout `answer` column is dropped as soon as the file is read.
  * Models receive only `subject` + `body`; labels are used afterwards for scoring.
  * Before scoring, the Chroma index is checked: no holdout ticket_id may be indexed.

Reports
  1. Type / Priority / Queue macro-F1 (+ per-class F1)
  2. High-Priority Incident Recall
       - strict: ticket predicted as Incident AND high
       - escalation: ticket routed to a human by the triage gate rules (what matters operationally)
  3. Escalation rate and triage latency on the evaluated tickets
  4. End-to-end pipeline sample (retrieval + gate + draft): latency p50/p95, gate trigger
     counts, escalation recall, draft verification rate (offline template unless --llm N)

Usage
  python evaluate.py                 # 2,000-ticket sample + 100-ticket end-to-end sample
  python evaluate.py --full          # all 12,353 holdout tickets
  python evaluate.py --e2e 200 --llm 20
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from safety_gate import evaluate_safety_gate
from triage_engine import predict_batch

ROOT = Path(__file__).resolve().parent
HOLDOUT_PATH = ROOT / "data" / "test_holdout.parquet"
CHROMA_SQLITE = ROOT / "chroma_dbs" / "chroma.sqlite3"
RESULTS_JSON = ROOT / "findings" / "holdout_metrics.json"
RESULTS_MD = ROOT / "findings" / "02_holdout_evaluation.md"
TRADEOFF_JSON = ROOT / "findings" / "threshold_tradeoff.json"

TYPE_CLASSES = ["Incident", "Request", "Problem", "Change"]
PRIORITY_CLASSES = ["high", "medium", "low"]
PRIORITY_MAP = {"critical": "high", "high": "high", "medium": "medium", "low": "low", "very_low": "low"}
QUEUE_CLASSES = [
    "Technical Support", "Product Support", "Customer Service", "IT Support", "Billing and Payments",
    "Returns and Exchanges", "Service Outages and Maintenance", "Sales and Pre-Sales", "Human Resources",
    "General Inquiry",
]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def leakage_check(holdout_ids: set[str]) -> dict:
    con = sqlite3.connect(f"file:{CHROMA_SQLITE}?mode=ro", uri=True)
    indexed = {row[0] for row in con.execute(
        "select string_value from embedding_metadata where key = 'ticket_id'")}
    con.close()
    overlap = holdout_ids & indexed
    if overlap:
        raise SystemExit(f"LEAKAGE: {len(overlap)} holdout tickets are in the retrieval index, e.g. {sorted(overlap)[:5]}")
    return {"indexed_tickets": len(indexed), "holdout_tickets": len(holdout_ids), "overlap": 0}


def scored(true: list, pred: list, classes: list[str]) -> dict:
    pairs = [(t, p) for t, p in zip(true, pred) if t in classes]
    if not pairs:
        return {"rows": 0}
    t, p = zip(*pairs)
    per_class = f1_score(t, p, labels=classes, average=None, zero_division=0)
    return {
        "rows": len(pairs),
        "macro_f1": round(float(f1_score(t, p, labels=classes, average="macro", zero_division=0)), 4),
        "accuracy": round(sum(a == b for a, b in pairs) / len(pairs), 4),
        "per_class_f1": {c: round(float(f), 4) for c, f in zip(classes, per_class)},
    }


def run_triage(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    subjects = [s if isinstance(s, str) else "" for s in df["subject"]]
    bodies = [b if isinstance(b, str) else "" for b in df["body"]]
    timings, preds = {}, {}
    for task in ("type", "priority", "queue"):
        start = time.perf_counter()
        preds[task] = predict_batch(task, bodies, subjects)
        timings[task] = round((time.perf_counter() - start) * 1000 / len(df), 2)
        print(f"  {task:<8} predicted {len(df)} tickets ({timings[task]} ms/ticket batched)")

    out = df[["ticket_id"]].copy()
    for task in preds:
        out[f"pred_{task}"] = [label for label, _ in preds[task]]
        out[f"conf_{task}"] = [conf for _, conf in preds[task]]
    # Same aggregate as graph.merge_triage_node
    out["confidence"] = (0.40 * out["conf_type"] + 0.35 * out["conf_queue"] + 0.25 * out["conf_priority"]).round(2)
    decisions = [
        evaluate_safety_gate(ticket_type=t, priority=p, confidence=c)
        for t, p, c in zip(out["pred_type"], out["pred_priority"], out["confidence"])
    ]
    out["gate_escalate"] = [d.escalate for d in decisions]
    out["gate_triggers"] = [list(d.triggers) for d in decisions]
    return timings, out


def threshold_tradeoff(df: pd.DataFrame, hpi: pd.Series) -> dict:
    """Escalation rate vs high-priority-incident escalation recall for a range of
    confidence thresholds (triage rules only). Reported for transparency; the shipped
    default (0.65) was fixed before this curve was computed."""
    points = []
    for step in range(30, 96, 5):
        threshold = step / 100
        escalate = [
            evaluate_safety_gate(ticket_type=t, priority=p, confidence=c, threshold=threshold).escalate
            for t, p, c in zip(df["pred_type"], df["pred_priority"], df["confidence"])
        ]
        escalate = pd.Series(escalate, index=df.index)
        points.append({
            "threshold": threshold,
            "escalation_rate": round(float(escalate.mean()), 4),
            "hpi_escalation_recall": round(float(escalate[hpi].mean()), 4) if hpi.any() else None,
        })
    return {"tickets": len(df), "high_priority_incidents": int(hpi.sum()), "points": points}


def run_end_to_end(df: pd.DataFrame, n: int, n_llm: int) -> dict:
    """Full LangGraph on a sample: retrieval + gate + draft (template unless n_llm > 0)."""
    import graph

    app = graph.build_sentinel_graph()
    import retriever

    retriever.warm_up()
    sample = df.sample(n=min(n, len(df)), random_state=7)
    latencies, triggers, verified, sources = [], collections.Counter(), [], collections.Counter()
    hpi_total = hpi_escalated = 0
    for i, row in enumerate(sample.itertuples()):
        # None -> server key (live Gemini); "" -> offline extractive draft
        config = {"configurable": {"gemini_api_key": None if i < n_llm else ""}}
        start = time.perf_counter()
        result = app.invoke({"subject": row.subject if isinstance(row.subject, str) else "", "body": row.body},
                            config=config)
        latencies.append((time.perf_counter() - start) * 1000)
        triggers.update(result.get("escalation_triggers", []))
        if result.get("escalation_status") == "AUTO_DRAFT":
            verified.append(bool(result.get("draft_verified")))
            sources[result.get("draft_source", "?").split(":")[0]] += 1
        if row.type == "Incident" and PRIORITY_MAP.get(str(row.priority)) == "high":
            hpi_total += 1
            hpi_escalated += result.get("escalation_status") == "ESCALATE"
    escalated = len(sample) - len(verified)
    return {
        "tickets": len(sample),
        "llm_tickets": min(n_llm, len(sample)),
        "latency_ms_p50": round(percentile(latencies, 50), 1),
        "latency_ms_p95": round(percentile(latencies, 95), 1),
        "escalation_rate": round(escalated / len(sample), 4),
        "trigger_counts": dict(triggers),
        "auto_drafts": len(verified),
        "draft_verified_rate": round(sum(verified) / len(verified), 4) if verified else None,
        "draft_sources": dict(sources),
        "high_priority_incidents": hpi_total,
        "high_priority_incident_escalation_recall": round(hpi_escalated / hpi_total, 4) if hpi_total else None,
    }


def write_markdown(r: dict) -> None:
    t, p, q, h = r["type"], r["priority"], r["queue"], r["high_priority_incident"]
    lines = [
        "# Findings 02: Holdout Evaluation",
        "",
        f"Generated by `python evaluate.py{' --full' if r['full'] else ''}` on {r['evaluated_tickets']:,} of "
        f"{r['holdout_tickets']:,} holdout tickets. Inputs: subject + body only; answers dropped at load; "
        f"retrieval index checked for holdout IDs (overlap = {r['leakage']['overlap']}).",
        "",
        "| Target | Metric | Score | Rows scored |",
        "| :--- | :--- | ---: | ---: |",
        f"| Type (4 classes) | Macro-F1 | **{t['macro_f1']:.3f}** | {t['rows']:,} |",
        f"| Priority (3 classes) | Macro-F1 | **{p['macro_f1']:.3f}** | {p['rows']:,} |",
        f"| Queue (10 classes) | Macro-F1 | **{q['macro_f1']:.3f}** | {q['rows']:,} |",
        f"| High-Priority Incident | Strict recall (predicted Incident + high) | **{h['strict_recall']:.1%}** | {h['count']:,} |",
        f"| High-Priority Incident | Escalation recall (routed to a human by the gate) | **{h['escalation_recall']:.1%}** | {h['count']:,} |",
        f"| All tickets | Triage-gate escalation rate | {r['triage_escalation_rate']:.1%} | {r['evaluated_tickets']:,} |",
        "",
        "Per-class F1:",
        "",
        f"- Type: {', '.join(f'{k} {v:.2f}' for k, v in t['per_class_f1'].items())}",
        f"- Priority: {', '.join(f'{k} {v:.2f}' for k, v in p['per_class_f1'].items())}",
        f"- Queue: {', '.join(f'{k} {v:.2f}' for k, v in q['per_class_f1'].items())}",
        "",
        f"Triage cost (batched, CPU): {r['triage_ms_per_ticket']}",
    ]
    e = r.get("end_to_end")
    if e:
        lines += [
            "",
            f"## End-to-end pipeline sample ({e['tickets']} tickets, {e['llm_tickets']} with live Gemini)",
            "",
            f"- Latency per ticket: p50 {e['latency_ms_p50']:.0f} ms, p95 {e['latency_ms_p95']:.0f} ms",
            f"- Escalation rate: {e['escalation_rate']:.1%}; gate triggers: {e['trigger_counts']}",
            f"- High-priority incidents escalated: {e['high_priority_incident_escalation_recall']} "
            f"({e['high_priority_incidents']} in sample)",
            f"- Auto-drafts: {e['auto_drafts']} (sources {e['draft_sources']}), "
            f"passing the grounding verifier: {e['draft_verified_rate']}",
        ]
    RESULTS_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SentinelDesk on the holdout split")
    parser.add_argument("--sample", type=int, default=2000, help="Tickets to score (default 2000)")
    parser.add_argument("--full", action="store_true", help="Score all 12,353 holdout tickets")
    parser.add_argument("--e2e", type=int, default=100, help="Tickets for the end-to-end pipeline sample (0 = skip)")
    parser.add_argument("--llm", type=int, default=0, help="How many end-to-end tickets use live Gemini drafting")
    args = parser.parse_args()

    if not HOLDOUT_PATH.is_file():
        raise SystemExit(f"{HOLDOUT_PATH} not found. Run python data_loader.py first.")
    holdout = pd.read_parquet(HOLDOUT_PATH).drop(columns=["answer"])  # never touch holdout answers
    leakage = leakage_check(set(holdout["ticket_id"]))
    print(f"Leakage check passed: 0 of {leakage['holdout_tickets']:,} holdout tickets in the index "
          f"({leakage['indexed_tickets']:,} indexed).")

    df = holdout if args.full or args.sample >= len(holdout) else holdout.sample(n=args.sample, random_state=42)
    print(f"Scoring {len(df):,} holdout tickets...")
    timings, preds = run_triage(df)
    df = df.merge(preds, on="ticket_id")

    true_priority = df["priority"].map(PRIORITY_MAP)
    hpi = (df["type"] == "Incident") & (true_priority == "high")
    results = {
        "full": bool(args.full),
        "holdout_tickets": len(holdout),
        "evaluated_tickets": len(df),
        "leakage": leakage,
        "type": scored(df["type"].tolist(), df["pred_type"].tolist(), TYPE_CLASSES),
        "priority": scored(true_priority.tolist(), df["pred_priority"].tolist(), PRIORITY_CLASSES),
        "queue": scored(df["queue"].tolist(), df["pred_queue"].tolist(), QUEUE_CLASSES),
        "high_priority_incident": {
            "count": int(hpi.sum()),
            "strict_recall": round(float(((df["pred_type"] == "Incident") & (df["pred_priority"] == "high"))[hpi].mean()), 4),
            "escalation_recall": round(float(df.loc[hpi, "gate_escalate"].mean()), 4),
            "escalated_by_rule": dict(collections.Counter(
                t for triggers in df.loc[hpi, "gate_triggers"] for t in triggers)),
        },
        "triage_escalation_rate": round(float(df["gate_escalate"].mean()), 4),
        "triage_ms_per_ticket": timings,
    }
    TRADEOFF_JSON.write_text(json.dumps(threshold_tradeoff(df, hpi), indent=2))
    if args.e2e:
        print(f"Running the full pipeline on {args.e2e} tickets ({args.llm} with live Gemini)...")
        results["end_to_end"] = run_end_to_end(df, args.e2e, args.llm)

    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    write_markdown(results)
    print(json.dumps({k: results[k] for k in ("type", "priority", "queue", "high_priority_incident",
                                               "triage_escalation_rate")}, indent=2))
    if "end_to_end" in results:
        print(json.dumps(results["end_to_end"], indent=2))
    print(f"\nSaved {RESULTS_JSON.relative_to(ROOT)} and {RESULTS_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

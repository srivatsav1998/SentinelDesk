"""
Quantitative Holdout Evaluation Script for SentinelDesk
Evaluates trained triage models on the 20% unseen test partition (data/test_holdout.parquet).
Calculates:
  1. Incident Type Macro-F1 (Incident, Request, Problem, Change)
  2. Priority Macro-F1 (high, medium, low)
  3. High-Priority Incident Recall (critical safety SLA metric)
  4. Concurrent Inference Latency (ms)
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from triage_engine import predict_priority, predict_type

HOLDOUT_PATH = Path(__file__).resolve().parent / "data" / "test_holdout.parquet"

PRIO_MAP = {
    "critical": "high",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "very_low": "low",
}


def compute_macro_f1(true_labels: List[str], pred_labels: List[str], classes: List[str]) -> Tuple[float, Dict[str, float]]:
    f1_per_class = {}
    for c in classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == c and p == c)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != c and p == c)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == c and p != c)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_per_class[c] = f1
        
    macro_f1 = sum(f1_per_class.values()) / len(f1_per_class) if f1_per_class else 0.0
    return macro_f1, f1_per_class


def run_evaluation(sample_size: int | None = None) -> None:
    if not HOLDOUT_PATH.is_file():
        raise FileNotFoundError(f"Holdout file not found at {HOLDOUT_PATH}. Run python data_loader.py first.")

    print(f"Loading holdout test dataset from {HOLDOUT_PATH}...")
    df = pd.read_parquet(HOLDOUT_PATH)
    total_holdout = len(df)
    
    if sample_size and sample_size < total_holdout:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(df)} records for rapid evaluation (use --full for all {total_holdout}).")
    else:
        print(f"Evaluating on all {len(df)} unseen holdout records.")

    true_types: List[str] = []
    pred_types: List[str] = []
    true_prios: List[str] = []
    pred_prios: List[str] = []
    latencies: List[float] = []

    print("Running multi-task inference across test tickets...")
    for idx, row in df.iterrows():
        subject = row.get("subject") if pd.notna(row.get("subject")) else ""
        body = row.get("body") if pd.notna(row.get("body")) else ""
        
        # Ground truth
        raw_type = row.get("type")
        raw_prio = row.get("priority")
        
        # Inference & timing
        t0 = time.perf_counter()
        p_type, _ = predict_type(body, subject=subject)
        p_prio, _ = predict_priority(body, subject=subject)
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)

        # Track valid types
        if pd.notna(raw_type) and str(raw_type).strip() in {"Incident", "Request", "Problem", "Change"}:
            true_types.append(str(raw_type).strip())
            pred_types.append(p_type)

        # Track valid priorities
        if pd.notna(raw_prio) and str(raw_prio).strip() in PRIO_MAP:
            true_prios.append(PRIO_MAP[str(raw_prio).strip()])
            pred_prios.append(p_prio)

    # 1. Type Macro-F1
    type_classes = ["Incident", "Request", "Problem", "Change"]
    type_macro_f1, type_f1s = compute_macro_f1(true_types, pred_types, type_classes)

    # 2. Priority Macro-F1
    prio_classes = ["high", "medium", "low"]
    prio_macro_f1, prio_f1s = compute_macro_f1(true_prios, pred_prios, prio_classes)

    # 3. High-Priority Incident Recall
    true_high_incidents = 0
    detected_high_incidents = 0
    for idx, row in df.iterrows():
        raw_type = str(row.get("type") or "").strip()
        raw_prio = PRIO_MAP.get(str(row.get("priority") or "").strip(), "")
        if raw_type == "Incident" and raw_prio == "high":
            true_high_incidents += 1
            # Check model predictions
            subject = row.get("subject") if pd.notna(row.get("subject")) else ""
            body = row.get("body") if pd.notna(row.get("body")) else ""
            p_type, _ = predict_type(body, subject=subject)
            p_prio, _ = predict_priority(body, subject=subject)
            if p_type == "Incident" and p_prio == "high":
                detected_high_incidents += 1

    high_incident_recall = (
        detected_high_incidents / true_high_incidents if true_high_incidents > 0 else 0.0
    )

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    print("\n" + "=" * 65)
    print("SENTINELDESK HOLDOUT EVALUATION BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Evaluated Samples: {len(df)} holdout tickets")
    print(f"Mean Concurrent Latency: {avg_latency:.2f} ms")
    print("-" * 65)
    print("1. Incident Type Classification:")
    print(f"   Macro-F1: {type_macro_f1:.4f}")
    for c, f1 in type_f1s.items():
        print(f"   - {c}: F1 = {f1:.4f}")
    print("-" * 65)
    print("2. Priority Classification:")
    print(f"   Macro-F1: {prio_macro_f1:.4f}")
    for c, f1 in prio_f1s.items():
        print(f"   - {c}: F1 = {f1:.4f}")
    print("-" * 65)
    print("3. High-Priority Incident Safety SLA:")
    print(f"   True High-Priority Incidents: {true_high_incidents}")
    print(f"   Correctly Detected: {detected_high_incidents}")
    print(f"   Recall: {high_incident_recall * 100:.2f}%")
    print("=" * 65)
    print("\nMarkdown Table for README.md:")
    print(
        f"| Target Dimension | Metric | Score | Target SLA |\n"
        f"| :--- | :--- | :--- | :--- |\n"
        f"| **Incident Type** (4 classes) | Macro-F1 | `{type_macro_f1:.3f}` | > 0.85 |\n"
        f"| **Priority Level** (3 classes) | Macro-F1 | `{prio_macro_f1:.3f}` | > 0.82 |\n"
        f"| **High-Priority Incident Recall** | Recall | `{high_incident_recall*100:.1f}%` | > 95.0% |\n"
        f"| **Concurrent Triage Latency** | Latency | `{avg_latency:.1f} ms` | < 100 ms |"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SentinelDesk models on holdout test set")
    parser.add_argument("--sample", type=int, default=100, help="Number of records to evaluate (default: 100)")
    parser.add_argument("--full", action="store_true", help="Evaluate full holdout split (12,353 records)")
    args = parser.parse_args()

    sample_size = None if args.full else args.sample
    run_evaluation(sample_size=sample_size)

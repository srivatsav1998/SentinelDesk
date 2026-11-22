"""Compare the two queue classifiers on historical tickets that NEITHER model saw.

* TF-IDF + logistic regression (train_queue.py, models/queue/queue_classifier.joblib):
  trained on fit+validation slices of the 38,819 clean historical tickets, so its
  calibration slice is unseen.
* mBERT checkpoint (notebooks/train_queue_classifier.ipynb, models/queue/sentineldesk-queue.zip):
  trained on the fit slice of all 49,412 historical rows (52 queue labels), so its
  validation + calibration slices are unseen.

Both splits are reproduced exactly (stratified, seed 42) and the intersection of
the unseen rows is scored on the 10 real support queues. The holdout is not used.

Usage: python compare_queue_models.py   -> findings/03_queue_model_comparison.md
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import triage_engine

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "findings" / "03_queue_model_comparison.md"
OUT_JSON = ROOT / "findings" / "queue_model_comparison.json"
SEED = 42


def unseen_by_both(kb: pd.DataFrame) -> pd.DataFrame:
    # mBERT notebook split: every historical row with queue + body, stratified on 52 labels
    nb = kb.dropna(subset=["queue", "body"]).copy()
    _, nb_rest = train_test_split(nb, test_size=0.2, random_state=SEED, stratify=nb["queue"])
    # train_queue.py split: clean rows only; final model trained on fit + validation
    clean = kb[kb["type"].notna() & kb["queue"].notna() & kb["body"].notna()].reset_index(drop=True)
    labels = clean["queue"].astype(str).to_numpy()
    idx = np.arange(len(clean))
    _, rest = train_test_split(idx, test_size=0.2, stratify=labels, random_state=SEED)
    _, cal = train_test_split(rest, test_size=0.5, stratify=labels[rest], random_state=SEED)
    keep = set(nb_rest["ticket_id"]) & set(clean.loc[cal, "ticket_id"])
    return clean[clean["ticket_id"].isin(keep)]


def main() -> None:
    kb = pd.read_parquet(ROOT / "data" / "kb_train.parquet")
    rows = unseen_by_both(kb)
    queues = sorted(rows["queue"].unique())
    subjects = [s if isinstance(s, str) else "" for s in rows["subject"]]
    bodies = rows["body"].tolist()

    results = {}
    for name, backend in [("TF-IDF + logistic regression", "sklearn"), ("mBERT (52-label checkpoint)", "transformer")]:
        os.environ["SENTINEL_QUEUE_BACKEND"] = backend
        try:
            preds = [label for label, _ in triage_engine.predict_batch("queue", bodies, subjects)]
        except triage_engine.ModelUnavailableError as error:
            print(f"{name}: unavailable ({error})")
            continue
        results[name] = {
            "macro_f1": f1_score(rows["queue"], preds, labels=queues, average="macro", zero_division=0),
            "accuracy": float(np.mean(np.array(preds) == rows["queue"].to_numpy())),
            "foreign": sum(p not in queues for p in preds),
            "top": collections.Counter(preds).most_common(3),
        }
        print(f"{name:<30} macro-F1={results[name]['macro_f1']:.4f}  accuracy={results[name]['accuracy']:.4f}")
    os.environ.pop("SENTINEL_QUEUE_BACKEND", None)

    lines = [
        "# Findings 03: Queue Model Comparison",
        "",
        f"Scored on {len(rows):,} historical tickets that neither model was trained on "
        "(reproduced splits, seed 42; the holdout is not used). Macro-F1 over the 10 support queues.",
        "",
        "| Model | Macro-F1 | Accuracy | Predictions outside the 10 queues | Most frequent predictions |",
        "| :--- | ---: | ---: | ---: | :--- |",
    ]
    for name, r in results.items():
        top = ", ".join(f"{label} ({count})" for label, count in r["top"])
        lines.append(f"| {name} | **{r['macro_f1']:.3f}** | {r['accuracy']:.3f} | {r['foreign']} | {top} |")
    lines += [
        "",
        "The mBERT checkpoint was trained for one epoch on all historical rows, including ~10.6k rows from the "
        "dataset's third CSV whose queue labels belong to a different taxonomy (52 labels in total), and it "
        "collapses towards `Technical Support`. `triage_engine.py` therefore prefers the TF-IDF model; set "
        "`SENTINEL_QUEUE_BACKEND=transformer` to use the checkpoint instead.",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    OUT_JSON.write_text(json.dumps({
        "tickets": len(rows),
        "models": [{"name": name, "macro_f1": round(r["macro_f1"], 4), "accuracy": round(r["accuracy"], 4),
                    "foreign_predictions": r["foreign"]} for name, r in results.items()],
    }, indent=2))
    print(f"saved {OUT.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

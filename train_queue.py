"""Train the SentinelDesk queue (department) classifier.

Model: TF-IDF (word 1-2 grams + character 3-5 grams) -> class-balanced logistic
regression, with a temperature fitted on a separate calibration slice so the
reported confidence is a calibrated probability.

Data rules (zero leakage):
  * Uses ONLY the historical 80% split (data/kb_train.parquet).
  * Drops rows from the dataset's third CSV (no `type` label, foreign queue
    taxonomy such as "Food & Drink/Groceries"). What remains are the 10 real
    support queues.
  * The 80% split is divided into fit / validation / calibration slices
    (80/10/10, stratified, seed 42). The holdout is never touched here.

Output: models/queue/queue_classifier.joblib + models/queue/queue_classifier.json

Usage:
    python train_queue.py              # pick C on validation, refit, calibrate, save
    python train_queue.py --C 4        # skip the sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "data" / "kb_train.parquet"
OUT_DIR = ROOT / "models" / "queue"
MODEL_FILE = OUT_DIR / "queue_classifier.joblib"
META_FILE = OUT_DIR / "queue_classifier.json"
SEED = 42
DATASET_REVISION = "ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb"


def ticket_text(subject: object, body: object) -> str:
    """Same text format used at inference time (see triage_engine.py)."""
    subject = subject if isinstance(subject, str) else ""
    body = body if isinstance(body, str) else ""
    return f"{subject}\n{body}".strip()


def load_clean_kb() -> tuple[list[str], np.ndarray]:
    df = pd.read_parquet(KB_PATH)
    df = df[df["type"].notna() & df["queue"].notna() & df["body"].notna()]
    texts = [ticket_text(s, b) for s, b in zip(df["subject"], df["body"])]
    return texts, df["queue"].astype(str).to_numpy()


def build_model(C: float) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=150_000,
                                 sublinear_tf=True, dtype=np.float32)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=150_000, sublinear_tf=True, dtype=np.float32)),
    ])
    classifier = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", classifier)])


def fit_temperature(logits: np.ndarray, y_idx: np.ndarray) -> float:
    """Temperature scaling: minimise NLL of softmax(logits / T) on the calibration slice."""
    def nll(log_t: float) -> float:
        log_probs = log_softmax(logits / np.exp(log_t), axis=1)
        return float(-log_probs[np.arange(len(y_idx)), y_idx].mean())
    result = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(result.x))


def expected_calibration_error(probs: np.ndarray, y_idx: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_idx
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Sweep on validation (macro-F1): C=2 0.570, 8 0.631, 32 0.648, 64 0.651, 128 0.647, 256 0.645
    parser.add_argument("--C", type=float, nargs="*", default=[64.0],
                        help="Inverse regularisation strength(s) to try on the validation slice")
    args = parser.parse_args()

    texts, labels = load_clean_kb()
    idx = np.arange(len(texts))
    fit_idx, rest_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=SEED)
    val_idx, cal_idx = train_test_split(rest_idx, test_size=0.5, stratify=labels[rest_idx], random_state=SEED)
    pick = lambda ids: [texts[i] for i in ids]  # noqa: E731
    print(f"clean historical rows={len(texts)}  fit={len(fit_idx)}  val={len(val_idx)}  cal={len(cal_idx)}")

    # 1. Choose C on the validation slice
    best_C, best_f1 = None, -1.0
    for C in args.C:
        t0 = time.time()
        model = build_model(C).fit(pick(fit_idx), labels[fit_idx])
        f1 = f1_score(labels[val_idx], model.predict(pick(val_idx)), average="macro")
        print(f"  C={C:<6} val macro-F1={f1:.4f}  ({time.time() - t0:.0f}s)")
        if f1 > best_f1:
            best_C, best_f1, best_model = C, f1, model

    val_pred = best_model.predict(pick(val_idx))
    print(f"\nBest C={best_C}  validation macro-F1={best_f1:.4f}")
    print(classification_report(labels[val_idx], val_pred, digits=3))

    # 2. Refit on fit + validation, then calibrate the temperature on the untouched slice
    train_idx = np.concatenate([fit_idx, val_idx])
    model = build_model(best_C).fit(pick(train_idx), labels[train_idx])
    classes = list(model.classes_)
    class_pos = {c: i for i, c in enumerate(classes)}
    cal_logits = model.decision_function(pick(cal_idx))
    cal_y = np.array([class_pos[c] for c in labels[cal_idx]])
    temperature = fit_temperature(cal_logits, cal_y)
    ece_raw = expected_calibration_error(softmax(cal_logits), cal_y)
    ece_cal = expected_calibration_error(softmax(cal_logits / temperature), cal_y)
    cal_f1 = f1_score(labels[cal_idx], model.predict(pick(cal_idx)), average="macro")
    print(f"temperature={temperature:.3f}  ECE raw={ece_raw:.4f} -> calibrated={ece_cal:.4f}  "
          f"calibration-slice macro-F1={cal_f1:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE, compress=3)
    META_FILE.write_text(json.dumps({
        "backend": "sklearn",
        "model_file": MODEL_FILE.name,
        "features": "tfidf word 1-2 + char_wb 3-5 (150k each), logistic regression (class_weight=balanced)",
        "input_format": "subject\\nbody",
        "temperature": temperature,
        "C": best_C,
        "classes": classes,
        "validation": {"macro_f1": round(best_f1, 4), "calibration_slice_macro_f1": round(cal_f1, 4),
                       "ece_raw": round(ece_raw, 4), "ece_calibrated": round(ece_cal, 4)},
        "fit_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "calibration_rows": int(len(cal_idx)),
        "dataset_revision": DATASET_REVISION,
        "seed": SEED,
    }, indent=2))
    size_mb = MODEL_FILE.stat().st_size / 1e6
    print(f"saved {MODEL_FILE.relative_to(ROOT)} ({size_mb:.1f} MB) and {META_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

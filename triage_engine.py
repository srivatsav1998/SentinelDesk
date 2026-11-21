"""Run the local SentinelDesk classification checkpoints."""
from __future__ import annotations

import json
import math
import os
import shutil
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import NamedTuple
from zipfile import BadZipFile, ZipFile

MODELS = Path(__file__).resolve().parent / "models"
LOAD_LOCK = Lock()
ARCHIVE_FILES = {
    "config.json", "model.safetensors", "pytorch_model.bin",
    "tokenizer.json", "tokenizer_config.json", "vocab.txt",
    "special_tokens_map.json", "added_tokens.json", "triage_metadata.json",
}
EXPECTED_LABELS = {
    "type": {"Incident", "Request", "Problem", "Change"},
    "priority": {"high", "medium", "low"},
}


class ModelUnavailableError(RuntimeError):
    """The requested classifier cannot be loaded."""


class Classifier(NamedTuple):
    model: object
    tokenizer: object
    temperature: float | None
    max_tokens: int
    input_format: str
    temporary_directory: TemporaryDirectory | None


def _checkpoint_path(task: str) -> tuple[Path, TemporaryDirectory | None]:
    directory = MODELS / task
    weights = ("model.safetensors", "model.safetensors.index.json",
               "pytorch_model.bin", "pytorch_model.bin.index.json")
    if (directory / "config.json").is_file() and any(
        (directory / name).is_file() for name in weights
    ):
        return directory, None

    archive_path = directory / f"sentineldesk-{task}.zip"
    if not archive_path.is_file():
        raise ModelUnavailableError(f"{task} checkpoint is missing from models/{task}/")

    temporary = TemporaryDirectory(prefix=f"sentineldesk-{task}-")
    unpacked = Path(temporary.name)
    try:
        with ZipFile(archive_path) as archive:
            for name in ARCHIVE_FILES.intersection(archive.namelist()):
                with archive.open(name) as source, (unpacked / name).open("wb") as target:
                    shutil.copyfileobj(source, target)
    except BadZipFile as error:
        temporary.cleanup()
        raise ModelUnavailableError(
            f"{archive_path.name} is not a complete ZIP; run git lfs pull"
        ) from error
    except OSError as error:
        temporary.cleanup()
        raise ModelUnavailableError(f"could not open {archive_path.name}: {error}") from error
    return unpacked, temporary


@lru_cache(maxsize=3)
def _load_classifier(task: str) -> Classifier:
    directory, temporary = _checkpoint_path(task)
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        metadata_path = directory / "triage_metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        temperature = metadata.get("temperature")
        if temperature is not None:
            temperature = float(temperature)
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("calibration temperature must be finite and positive")
        max_tokens = int(metadata.get("max_tokens", 128))
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        input_format = metadata.get("input_format", "single" if task == "queue" else "paired")
        if input_format not in {"single", "paired"}:
            raise ValueError("input_format must be 'single' or 'paired'")

        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True
        ).eval()
        labels = set(model.config.id2label.values())
        if task in EXPECTED_LABELS and labels != EXPECTED_LABELS[task]:
            raise ValueError(f"expected labels {sorted(EXPECTED_LABELS[task])}")
        if not labels or any(not label or label.startswith("LABEL_") for label in labels):
            raise ValueError("config.json must provide the trained class names in id2label")
    except (OSError, ValueError, TypeError, AttributeError, ImportError, RuntimeError) as error:
        if temporary is not None:
            temporary.cleanup()
        raise ModelUnavailableError(f"could not load the {task} checkpoint: {error}") from error
    return Classifier(model, tokenizer, temperature, max_tokens, input_format, temporary)


def _sklearn_model_path(task: str) -> Path:
    return MODELS / task / f"{task}_classifier.joblib"


@lru_cache(maxsize=3)
def _load_sklearn(task: str) -> tuple[object, float]:
    """Linear text classifier produced by train_<task>.py (e.g. train_queue.py).

    joblib uses pickle, so only load files this repo produced itself: the path is
    fixed to models/<task>/ and the artifact is written by our training script,
    never downloaded from user input.
    """
    path = _sklearn_model_path(task)
    try:
        import joblib

        model = joblib.load(path)
        meta_path = path.with_suffix(".json")
        metadata = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        temperature = float(metadata.get("temperature", 1.0))
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("calibration temperature must be finite and positive")
    except (OSError, ValueError, ImportError, EOFError) as error:
        raise ModelUnavailableError(f"could not load {path.name}: {error}") from error
    return model, temperature


def _predict_sklearn(task: str, text: str, subject: str | None) -> tuple[str, float]:
    with LOAD_LOCK:
        model, temperature = _load_sklearn(task)
    import numpy as np

    logits = np.asarray(model.decision_function([f"{subject or ''}\n{text}".strip()])[0], dtype=float) / temperature
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    index = int(probabilities.argmax())
    return str(model.classes_[index]), float(probabilities[index])


def _use_sklearn(task: str) -> bool:
    """Choose the backend for a task.

    SENTINEL_<TASK>_BACKEND=sklearn|transformer forces one. By default the linear
    model from train_<task>.py is preferred when present: for the queue, it scored
    0.661 macro-F1 vs 0.230 for the 52-class mBERT checkpoint on 776 historical
    tickets that neither model was trained on (findings/03_queue_model_comparison.md).
    """
    choice = os.environ.get(f"SENTINEL_{task.upper()}_BACKEND", "").strip().lower()
    if choice == "transformer":
        return False
    if choice == "sklearn":
        return True
    return _sklearn_model_path(task).is_file()


def _predict(task: str, text: str, subject: str | None) -> tuple[str, float]:
    if _use_sklearn(task):
        return _predict_sklearn(task, text, subject)
    # Prevent concurrent requests from loading the same weights more than once.
    try:
        with LOAD_LOCK:
            classifier = _load_classifier(task)
    except ModelUnavailableError:
        # Transformer checkpoint unusable: fall back to the trained linear model if present.
        if _sklearn_model_path(task).is_file():
            return _predict_sklearn(task, text, subject)
        raise

    import torch

    subject = subject or ""
    options = {"return_tensors": "pt", "truncation": True, "max_length": classifier.max_tokens}
    if classifier.input_format == "paired":
        inputs = classifier.tokenizer(subject, text, **options)
    else:
        inputs = classifier.tokenizer(f"{subject} {text}".strip(), **options)
    with torch.inference_mode():
        logits = classifier.model(**inputs).logits[0]
        if not torch.isfinite(logits).all():
            raise ModelUnavailableError(f"{task} classifier returned invalid scores")
        index = int(logits.argmax().item())
        confidence = (
            float((logits / classifier.temperature).softmax(dim=-1)[index].item())
            if classifier.temperature is not None else 0.0
        )
    return classifier.model.config.id2label[index], confidence


def predict_batch(
    task: str, texts: list[str], subjects: list[str | None], batch_size: int = 32
) -> list[tuple[str, float]]:
    """Batched predictions for offline evaluation (same models and calibration as `_predict`)."""
    classifier = None
    if not _use_sklearn(task):
        try:
            with LOAD_LOCK:
                classifier = _load_classifier(task)
        except ModelUnavailableError:
            if not _sklearn_model_path(task).is_file():
                raise
    if classifier is None:
        with LOAD_LOCK:
            model, temperature = _load_sklearn(task)
        import numpy as np

        joined = [f"{s or ''}\n{t}".strip() for s, t in zip(subjects, texts)]
        logits = np.asarray(model.decision_function(joined), dtype=float) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        indices = probabilities.argmax(axis=1)
        return [(str(model.classes_[i]), float(probabilities[row, i])) for row, i in enumerate(indices)]

    import torch

    results: list[tuple[str, float]] = []
    options = {"return_tensors": "pt", "truncation": True, "max_length": classifier.max_tokens, "padding": True}
    for start in range(0, len(texts), batch_size):
        chunk_text = texts[start:start + batch_size]
        chunk_subject = [s or "" for s in subjects[start:start + batch_size]]
        if classifier.input_format == "paired":
            inputs = classifier.tokenizer(chunk_subject, chunk_text, **options)
        else:
            inputs = classifier.tokenizer([f"{s} {t}".strip() for s, t in zip(chunk_subject, chunk_text)], **options)
        with torch.inference_mode():
            logits = classifier.model(**inputs).logits
            scale = classifier.temperature if classifier.temperature is not None else 1.0
            probabilities = (logits / scale).softmax(dim=-1)
            confidence, index = probabilities.max(dim=-1)
        for i, c in zip(index.tolist(), confidence.tolist()):
            results.append((classifier.model.config.id2label[i], c if classifier.temperature is not None else 0.0))
    return results


def predict_type(text: str, *, subject: str | None = None) -> tuple[str, float]:
    """Return the ticket type and calibrated confidence."""
    return _predict("type", text, subject)


def predict_queue(text: str, *, subject: str | None = None) -> tuple[str, float]:
    """Return the routing queue and confidence when calibration is available."""
    return _predict("queue", text, subject)


def predict_priority(text: str, *, subject: str | None = None) -> tuple[str, float]:
    """Return the ticket priority and calibrated confidence."""
    return _predict("priority", text, subject)

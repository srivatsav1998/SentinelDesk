"""Run the local SentinelDesk classification checkpoints."""
from __future__ import annotations

import json
import math
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


def _predict(task: str, text: str, subject: str | None) -> tuple[str, float]:
    # Prevent concurrent requests from loading the same weights more than once.
    with LOAD_LOCK:
        classifier = _load_classifier(task)

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


def predict_type(text: str, *, subject: str | None = None) -> tuple[str, float]:
    """Return the ticket type and calibrated confidence."""
    return _predict("type", text, subject)


def predict_queue(text: str, *, subject: str | None = None) -> tuple[str, float]:
    """Return the routing queue and confidence when calibration is available."""
    return _predict("queue", text, subject)


def predict_priority(text: str, *, subject: str | None = None) -> tuple[str, float]:
    """Return the ticket priority and calibrated confidence."""
    return _predict("priority", text, subject)

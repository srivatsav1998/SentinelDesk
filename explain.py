"""Explanations shown in the operator console.

* highlight_spans(text): character spans in a ticket that triggered a guardrail
  (sensitive request, prompt injection) or will be masked as personal data.
* queue_top_terms(subject, body, label): the words that pushed the TF-IDF queue
  model towards its predicted department (positive feature contributions).
"""

from __future__ import annotations

from functools import lru_cache

from guardrails import _INJECTION, _PII_PATTERNS, SENSITIVE_PATTERNS

_STOPWORDS = set("""
a an and are as at be been but by can cannot could do does for from has have i if in is it its me
my no not of on or our please since so than that the their them there these this to too us was we
were what when which will with would you your yours since also just only still any all some
der die das und ist nicht ein eine einen einem einer es wir ich sie mit für von zu im in den dem
des auf bei bitte unser unsere unseren uns sich auch noch nur aber oder wie was wenn kann können
""".split())

SENSITIVE_LABELS = {
    "refund": "refund request",
    "compensation": "compensation request",
    "legal": "legal threat",
    "guarantee": "guarantee request",
}


def highlight_spans(text: str) -> list[dict]:
    """Non-overlapping spans: {"start", "end", "kind", "label"}; kind is
    "injection", "sensitive" or "pii". Earlier, longer matches win overlaps."""
    text = text or ""
    found: list[dict] = []
    for match in _INJECTION.finditer(text):
        found.append({"start": match.start(), "end": match.end(), "kind": "injection",
                      "label": "tries to override instructions"})
    for name, pattern in SENSITIVE_PATTERNS.items():
        for match in pattern.finditer(text):
            found.append({"start": match.start(), "end": match.end(), "kind": "sensitive",
                          "label": SENSITIVE_LABELS.get(name, name)})
    for label, pattern in _PII_PATTERNS:
        for match in pattern.finditer(text):
            found.append({"start": match.start(), "end": match.end(), "kind": "pii",
                          "label": f"{label.lower()} masked before the AI sees it"})
    found.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
    spans: list[dict] = []
    for span in found:
        if not spans or span["start"] >= spans[-1]["end"]:
            spans.append(span)
    return spans


@lru_cache(maxsize=1)
def _queue_model():
    import triage_engine

    if not triage_engine._sklearn_model_path("queue").is_file():
        return None
    model, _ = triage_engine._load_sklearn("queue")
    features = model.named_steps["features"]
    names = features.get_feature_names_out()
    return model, features, names


def queue_top_terms(subject: str | None, body: str, label: str, k: int = 6) -> list[str]:
    """Word features with the largest positive contribution to `label`."""
    loaded = _queue_model()
    if loaded is None:
        return []
    model, features, names = loaded
    classifier = model.named_steps["clf"]
    classes = list(classifier.classes_)
    if label not in classes:
        return []
    row = features.transform([f"{subject or ''}\n{body}".strip()]).tocoo()
    weights = classifier.coef_[classes.index(label)]
    scored = []
    for column, value in zip(row.col, row.data):
        name = names[column]
        if not name.startswith("word__"):
            continue  # character n-grams are not readable
        contribution = value * weights[column]
        if contribution > 0:
            scored.append((contribution, name.split("__", 1)[1]))
    scored.sort(reverse=True)
    terms: list[str] = []
    for _, term in scored:
        if all(word in _STOPWORDS for word in term.split()):
            continue
        if not any(term in kept or kept in term for kept in terms):
            terms.append(term)
        if len(terms) == k:
            break
    return terms

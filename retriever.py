"""Precedent retrieval over the historical 80% knowledge base (Chroma + BAAI/bge-m3).

The Chroma index in chroma_dbs/ holds exactly the 49,412 historical tickets
(TKT-00000 .. TKT-49411). No holdout ticket is indexed.

Public API (see contracts/team_contracts.md):
    retrieve_precedents(query_text, top_k=3) -> list[dict]
    embed_query(text) -> np.ndarray          (cached; shared with other components)
    warm_up()                                (load model + index ahead of the first ticket)
"""

from __future__ import annotations

import hashlib
import os
import shutil
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np

CHROMA_DIR = Path(__file__).resolve().parent / "chroma_dbs"
CACHE_DIR = Path(os.environ.get("SENTINEL_CACHE_DIR", Path.home() / ".cache" / "sentineldesk"))
COLLECTION = "my_data"
EMBED_MODEL = os.environ.get("SENTINEL_EMBED_MODEL", "BAAI/bge-m3")
MAX_SEQ_LENGTH = 512
_LOCK = Lock()


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _embedder():
    import torch
    from sentence_transformers import SentenceTransformer

    device = _device()
    model = SentenceTransformer(
        EMBED_MODEL,
        device=device,
        model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32},
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def _working_index_dir() -> Path:
    """Chroma writes to chroma.sqlite3 whenever it opens an index. Work on a private
    copy (keyed by the tracked file's size and first MB) so the git-tracked index
    in chroma_dbs/ is never modified."""
    sqlite_file = CHROMA_DIR / "chroma.sqlite3"
    if not sqlite_file.is_file():
        raise FileNotFoundError(f"Chroma index not found at {CHROMA_DIR} (run git lfs pull)")
    with sqlite_file.open("rb") as handle:
        head = handle.read(1 << 20)
    if head.startswith(b"version https://git-lfs"):
        raise FileNotFoundError("chroma_dbs/ holds Git LFS pointers, not the index (run git lfs pull)")
    stamp = hashlib.sha1(head + str(sqlite_file.stat().st_size).encode()).hexdigest()[:12]
    target = CACHE_DIR / f"chroma-{stamp}"
    if not (target / ".complete").is_file():
        staging = CACHE_DIR / f"chroma-{stamp}.tmp"
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(CHROMA_DIR, staging)
        (staging / ".complete").touch()
        shutil.rmtree(target, ignore_errors=True)
        staging.rename(target)
    return target


@lru_cache(maxsize=1)
def _collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(_working_index_dir()))
    return client.get_collection(COLLECTION)


def warm_up() -> None:
    """Load the embedding model and open the index once (call at app start)."""
    with _LOCK:
        _embedder()
        _collection()
    embed_query("warm-up")


@lru_cache(maxsize=512)
def _embed_cached(text: str) -> tuple[float, ...]:
    with _LOCK:
        model = _embedder()
    vector = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
    return tuple(float(x) for x in vector)


def embed_query(text: str) -> np.ndarray:
    """Normalised bge-m3 embedding of a ticket text (cached per text)."""
    return np.asarray(_embed_cached(text.strip()), dtype=np.float32)


def _clean(value: object) -> str:
    # Some dataset answers contain literal "\n" escape sequences instead of newlines.
    text = "" if value is None else str(value).replace("\\n", "\n").strip()
    return "" if text.lower() in {"nan", "none"} else text


def retrieve_precedents(query_text: str, top_k: int = 3) -> list[dict]:
    """
    Args:
        query_text: Combined ticket subject and body
        top_k: Number of historical tickets to return (default: 3)

    Returns:
        list of dicts (highest similarity first), each containing:
            "ticket_id": str (e.g. "TKT-00123")
            "similarity": float (cosine similarity between 0.0 and 1.0)
            "historical_answer": str (the past resolution text from 'answer' column)
        plus display fields: "subject", "language", "queue", "type", "priority".
        Tickets without a recorded answer are never returned.
    """
    query_text = (query_text or "").strip()
    if not query_text or top_k <= 0:
        return []

    vector = embed_query(query_text)
    with _LOCK:
        collection = _collection()
    # ~10.6k indexed rows (from the dataset's third CSV) have no answer. Search only
    # tickets with a resolution; over-fetch a little and re-check as a safety net.
    response = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=top_k + 3,
        where={"answer": {"$ne": ""}},
        include=["metadatas", "distances"],
    )

    results: list[dict] = []
    for meta, distance in zip(response["metadatas"][0], response["distances"][0]):
        answer = _clean(meta.get("answer"))
        if not answer:
            continue
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))  # Chroma cosine distance = 1 - cosine similarity
        results.append({
            "ticket_id": str(meta["ticket_id"]),
            "similarity": round(similarity, 4),
            "historical_answer": answer,
            "subject": _clean(meta.get("subject")),
            "language": _clean(meta.get("language")),
            "queue": _clean(meta.get("queue")),
            "type": _clean(meta.get("type")),
            "priority": _clean(meta.get("priority")),
        })
        if len(results) == top_k:
            break
    return results


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    warm_up()
    print(f"warm-up {time.perf_counter() - t0:.1f}s")
    for query in [
        "VPN connection keeps dropping, hospital staff cannot reach the EMR system",
        "Wir können uns nicht mehr in das Kundenportal einloggen, Passwort-Reset funktioniert nicht",
    ]:
        t0 = time.perf_counter()
        hits = retrieve_precedents(query, top_k=3)
        print(f"\n{query[:70]}  ({(time.perf_counter() - t0) * 1000:.0f} ms)")
        for hit in hits:
            print(f"  {hit['ticket_id']}  sim={hit['similarity']:.3f}  [{hit['language']}] {hit['subject'][:60]}")

"""Live outage / burst detection.

Keeps a sliding window of recently processed tickets (their bge-m3 embeddings,
reused from retrieval) and raises a Major Incident alert when at least
`min_cluster` tickets in the window describe the same failure (pairwise cosine
similarity >= `similarity` with the newest ticket).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import numpy as np

from guardrails import mask_pii

DUPLICATE_SIMILARITY = 0.999


@dataclass
class BurstAlert:
    count: int
    window_minutes: int
    members: list[dict] = field(default_factory=list)  # [{"label", "subject", "similarity", "age_s"}]


class BurstDetector:
    def __init__(self, window_s: int = 1800, similarity: float = 0.72, min_cluster: int = 3, max_items: int = 500):
        self.window_s = window_s
        self.similarity = similarity
        self.min_cluster = min_cluster
        self._items: deque = deque(maxlen=max_items)
        self._lock = Lock()

    def add(self, label: str, subject: str, embedding: np.ndarray, now: Optional[float] = None) -> Optional[BurstAlert]:
        now = time.time() if now is None else now
        # The window is shared by every operator session, so keep only masked subjects.
        subject = mask_pii(subject or "")[0][:80]
        vector = np.asarray(embedding, dtype=np.float32)
        vector = vector / (np.linalg.norm(vector) or 1.0)
        with self._lock:
            while self._items and now - self._items[0]["ts"] > self.window_s:
                self._items.popleft()
            members = [{"label": label, "subject": subject, "similarity": 1.0, "age_s": 0}]
            duplicate = None
            for item in self._items:
                score = float(item["vec"] @ vector)
                if score >= DUPLICATE_SIMILARITY:  # same ticket resubmitted: not a new report
                    duplicate = item
                elif score >= self.similarity:
                    members.append({"label": item["label"], "subject": item["subject"],
                                    "similarity": round(score, 3), "age_s": int(now - item["ts"])})
            if duplicate is not None:
                duplicate["ts"] = now
            else:
                self._items.append({"ts": now, "vec": vector, "label": label, "subject": subject})
        if len(members) >= self.min_cluster:
            return BurstAlert(count=len(members), window_minutes=self.window_s // 60, members=members)
        return None

    def reset(self) -> None:
        with self._lock:
            self._items.clear()

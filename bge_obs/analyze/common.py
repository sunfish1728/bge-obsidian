from __future__ import annotations

import numpy as np


def snippet(text: str, n: int = 160) -> str:
    s = " ".join(text.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def r(x: float) -> float:
    return round(float(x), 4)


def note_chunk_ranges(chunks: list[dict]) -> dict[str, tuple[int, int]]:
    """Map note path -> (start, end) row range in the chunk matrix (chunks are sorted by note)."""
    out: dict[str, tuple[int, int]] = {}
    for i, c in enumerate(chunks):
        s, _ = out.get(c["note"], (i, i))
        out[c["note"]] = (s, i + 1)
    return out


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    if k <= 0:
        return np.zeros(0, dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]

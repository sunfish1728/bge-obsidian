"""Nearest notes for a document, a text query, or an image."""
from __future__ import annotations

import numpy as np

from .common import note_chunk_ranges, r, snippet, topk


def similar(data: dict, vec: np.ndarray, k: int = 10, self_path: str | None = None,
            query_chunks: np.ndarray | None = None) -> list[dict]:
    notes = data["notes"]
    if not notes:
        return []
    sims = data["note_vecs"] @ vec
    if self_path is not None:
        for i, n in enumerate(notes):
            if n["path"] == self_path:
                sims[i] = -1.0
    ranges = note_chunk_ranges(data["chunks"])
    q = query_chunks if query_chunks is not None and len(query_chunks) else vec[None, :]
    out = []
    for i in topk(sims, k):
        if sims[i] < 0:
            continue
        n = notes[i]
        s, e = ranges.get(n["path"], (0, 0))
        item = {"path": n["path"], "title": n["title"], "score": r(sims[i])}
        if e > s:
            cs = (q @ data["chunk_vecs"][s:e].T).max(axis=0)
            j = int(cs.argmax())
            c = data["chunks"][s + j]
            item["best_chunk"] = {"heading": c["heading"], "text": snippet(c["text"]), "score": r(cs[j])}
        out.append(item)
    return out


def similar_images(data: dict, vec: np.ndarray, k: int = 10) -> list[dict]:
    imgs = data["images"]
    if not imgs:
        return []
    sims = data["image_vecs"] @ vec
    return [{"path": imgs[i]["path"], "score": r(sims[i]), "referenced_by": imgs[i]["referenced_by"]}
            for i in topk(sims, k)]

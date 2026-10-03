"""Folder classification by folder centroids."""
from __future__ import annotations

import numpy as np

from ..embed.base import Embedder, normalize
from .common import r

MIN_NOTES = 3


def folder_centroids(data: dict, embedder: Embedder, folder_desc: dict[str, str],
                     exclude_path: str | None = None, skip: set[str] = frozenset()) -> tuple[list[str], np.ndarray, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, n in enumerate(data["notes"]):
        if n["path"] == exclude_path or not n["folder"] or any(n["folder"].split("/")[0] == s for s in skip):
            continue
        groups.setdefault(n["folder"], []).append(i)
    for f in folder_desc:
        groups.setdefault(f, [])
    if not groups:
        return [], np.zeros((0, 0), np.float32), []
    names = sorted(groups)
    # Folders with few notes are anchored by their name/description.
    weak = [f for f in names if len(groups[f]) < MIN_NOTES]
    desc_vecs = dict(zip(weak, embedder.embed_texts(
        [f"{f.replace('/', ' / ')}: {folder_desc.get(f, '')}".strip(": ") for f in weak]))) if weak else {}
    cents = []
    for f in names:
        idx = groups[f]
        v = data["note_vecs"][idx].sum(axis=0) if idx else np.zeros(data["note_vecs"].shape[1] or 1024, np.float32)
        v = normalize(v) if idx else v
        if f in desc_vecs:
            v = v + desc_vecs[f]
        cents.append(normalize(v))
    return names, np.vstack(cents), [len(groups[f]) for f in names]


def classify(data: dict, vec: np.ndarray, embedder: Embedder, cfg, self_path: str | None = None) -> dict:
    th = cfg.thresholds
    names, cents, counts = folder_centroids(data, embedder, cfg.get("folders") or {}, self_path,
                                            skip={cfg["inbox"], cfg["review_dir"]})
    if not names:
        return {"top": [], "margin": 0.0, "confident": False, "reason": "no folders in index"}
    sims = cents @ vec
    order = np.argsort(-sims)[:3]
    top = [{"folder": names[i], "score": r(sims[i]), "notes": counts[i]} for i in order]
    margin = float(sims[order[0]] - sims[order[1]]) if len(order) > 1 else 1.0
    confident = top[0]["score"] >= th["classify_min"] and margin >= th["classify_margin"]
    out = {"top": top, "margin": r(margin), "confident": confident}
    if not confident:
        out["reason"] = "low_score" if top[0]["score"] < th["classify_min"] else "ambiguous"
    return out

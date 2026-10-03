"""Tag suggestion from a curated tag list, and tag-list proposal by clustering."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from ..embed.base import Embedder, normalize
from .common import r

MIN_TAG_NOTES = 3


def load_tag_list(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    return {str(k).lstrip("#"): str(v or "") for k, v in d.items()}


def suggest_tags(data: dict, vec: np.ndarray, embedder: Embedder, cfg, existing: list[str],
                 self_path: str | None = None, k: int = 5) -> dict:
    tag_list = load_tag_list(cfg.tags_path)
    usage: dict[str, list[int]] = {}
    for i, n in enumerate(data["notes"]):
        if n["path"] != self_path:
            for t in n["tags"]:
                usage.setdefault(t, []).append(i)

    if tag_list is None:  # fall back to tags already used in the vault
        candidates = {t: "" for t, idx in usage.items() if len(idx) >= MIN_TAG_NOTES}
    else:
        candidates = tag_list
    if not candidates:
        return {"tags": [], "existing": existing, "tags_file": str(cfg.tags_path),
                "tags_file_missing": tag_list is None, "reason": "no candidate tags"}

    names = sorted(candidates)
    desc = embedder.embed_texts([f"{t}: {candidates[t]}" if candidates[t] else t for t in names])
    scores = desc @ vec
    sources = ["description"] * len(names)
    for j, t in enumerate(names):
        idx = usage.get(t, [])
        if len(idx) >= MIN_TAG_NOTES:
            s = float(normalize(data["note_vecs"][idx].sum(axis=0)) @ vec)
            if s > scores[j]:
                scores[j], sources[j] = s, "usage"
    order = np.argsort(-scores)
    out = [{"tag": names[j], "score": r(scores[j]), "source": sources[j]}
           for j in order[:k] if scores[j] >= cfg.thresholds["tag_min"]]
    return {"tags": out, "existing": existing, "tags_file": str(cfg.tags_path),
            "tags_file_missing": tag_list is None}


def propose_tag_list(data: dict, k: int | None = None, distance: float = 0.35, min_size: int = 3) -> dict:
    """Cluster notes; the agent names each cluster to build tags.yaml."""
    from sklearn.cluster import AgglomerativeClustering

    notes, vecs = data["notes"], data["note_vecs"]
    if len(notes) < 2:
        return {"clusters": [], "reason": "need at least 2 notes"}
    model = AgglomerativeClustering(
        n_clusters=k, distance_threshold=None if k else distance, metric="cosine", linkage="average")
    labels = model.fit_predict(vecs)
    clusters = []
    for lab in set(labels.tolist()):
        idx = np.nonzero(labels == lab)[0]
        if len(idx) < min_size:
            continue
        cent = normalize(vecs[idx].sum(axis=0))
        sims = vecs[idx] @ cent
        rep = idx[np.argsort(-sims)[:5]]
        tags = Counter(t for i in idx for t in notes[i]["tags"]).most_common(5)
        folders = Counter(notes[i]["folder"] or "(root)" for i in idx).most_common(3)
        clusters.append({"size": int(len(idx)), "cohesion": r(sims.mean()),
                         "representative": [notes[i]["title"] or notes[i]["path"] for i in rep],
                         "common_tags": [t for t, _ in tags], "folders": [f for f, _ in folders]})
    clusters.sort(key=lambda c: -c["size"])
    clustered = sum(c["size"] for c in clusters)
    return {"clusters": clusters, "notes": len(notes), "unclustered": len(notes) - clustered}

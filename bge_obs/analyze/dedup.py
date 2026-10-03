"""Duplicate detection: exact, note-level, containment, images."""
from __future__ import annotations

import numpy as np

from ..index import Embedded, text_hash
from .common import note_chunk_ranges, r, snippet

KIND_ORDER = {"exact": 0, "duplicate": 1, "contained_in": 2, "contains": 3, "near_duplicate": 4}


def _evidence(a_chunks, a_vecs, b_chunks, b_vecs, n: int = 3) -> list[dict]:
    if len(a_vecs) == 0 or len(b_vecs) == 0:
        return []
    sim = a_vecs @ b_vecs.T
    flat = np.argsort(-sim, axis=None)[: n * 3]
    out, used_a = [], set()
    for f in flat:
        i, j = divmod(int(f), sim.shape[1])
        if i in used_a:
            continue
        used_a.add(i)
        out.append({"src": snippet(a_chunks[i]["text"]), "src_heading": a_chunks[i]["heading"],
                    "dst": snippet(b_chunks[j]["text"]), "dst_heading": b_chunks[j]["heading"],
                    "score": r(sim[i, j])})
        if len(out) == n:
            break
    return out


def _containment(sim: np.ndarray, th: float) -> tuple[float, float]:
    """(fraction of A chunks matched in B, fraction of B chunks matched in A)."""
    if sim.size == 0:
        return 0.0, 0.0
    m = sim >= th
    return float(m.any(axis=1).mean()), float(m.any(axis=0).mean())


def _classify(note_sim: float, a_in_b: float, b_in_a: float, th: dict) -> str | None:
    if note_sim >= th["dup"]:
        return "duplicate"
    if a_in_b >= th["contained_ratio"] and b_in_a < th["contained_ratio"]:
        return "contained_in"  # the checked doc is a subset of the target
    if b_in_a >= th["contained_ratio"] and a_in_b < th["contained_ratio"]:
        return "contains"
    if a_in_b >= th["contained_ratio"] and b_in_a >= th["contained_ratio"]:
        return "duplicate"
    if note_sim >= th["near_dup"]:
        return "near_duplicate"
    return None


def dedup_doc(data: dict, emb: Embedded, fhash: str | None, th: dict, self_path: str | None = None) -> list[dict]:
    """Compare one embedded document against the whole index."""
    notes, chunks, cvecs = data["notes"], data["chunks"], data["chunk_vecs"]
    a_chunks = [{"text": c.text, "heading": c.heading_path} for c in emb.chunks]
    thash = text_hash(emb.doc.text)
    note_sims = data["note_vecs"] @ emb.vec if len(notes) else np.zeros(0)
    chunk_sims = emb.chunk_vecs @ cvecs.T if len(cvecs) and len(emb.chunk_vecs) else None
    ranges = note_chunk_ranges(chunks)
    out = []
    for i, n in enumerate(notes):
        if n["path"] == self_path:
            continue
        if (fhash and n["hash"] == fhash) or n["text_hash"] == thash:
            out.append({"kind": "exact", "target": n["path"], "score": 1.0, "evidence": []})
            continue
        s, e = ranges.get(n["path"], (0, 0))
        sim = chunk_sims[:, s:e] if chunk_sims is not None else np.zeros((0, 0))
        if note_sims[i] < th["near_dup"] and (sim.size == 0 or sim.max() < th["chunk_match"]):
            continue
        a_in_b, b_in_a = _containment(sim, th["chunk_match"])
        kind = _classify(note_sims[i], a_in_b, b_in_a, th)
        if kind:
            out.append({"kind": kind, "target": n["path"], "score": r(note_sims[i]),
                        "coverage": {"doc_in_target": r(a_in_b), "target_in_doc": r(b_in_a)},
                        "evidence": _evidence(a_chunks, emb.chunk_vecs, chunks[s:e], cvecs[s:e])})
    out.sort(key=lambda d: (KIND_ORDER[d["kind"]], -d["score"]))
    return out


def dedup_images(data: dict, vecs: np.ndarray, names: list[str], hashes: list[str | None], th: dict,
                 self_paths: set[str] = frozenset()) -> list[dict]:
    imgs, ivecs = data["images"], data["image_vecs"]
    if len(imgs) == 0 or len(vecs) == 0:
        return []
    sims = vecs @ ivecs.T
    out = []
    for a in range(len(names)):
        for b in np.nonzero(sims[a] >= th["image_dup"])[0]:
            t = imgs[b]
            if t["path"] in self_paths:
                continue
            exact = hashes[a] is not None and hashes[a] == t["hash"]
            out.append({"kind": "image_exact" if exact else "image_duplicate", "image": names[a],
                        "target": t["path"], "score": 1.0 if exact else r(sims[a, b]),
                        "referenced_by": t["referenced_by"]})
    return out


def dedup_all(data: dict, th: dict) -> dict:
    """All duplicate pairs inside the index (each unordered pair reported once)."""
    notes, chunks, cvecs, nvecs = data["notes"], data["chunks"], data["chunk_vecs"], data["note_vecs"]
    ranges = note_chunk_ranges(chunks)
    pairs = []
    by_hash: dict[str, list[str]] = {}
    for n in notes:
        by_hash.setdefault(n["text_hash"], []).append(n["path"])
    exact_groups = [g for g in by_hash.values() if len(g) > 1]
    exact_set = {p for g in exact_groups for p in g}

    for i, a in enumerate(notes):
        sa, ea = ranges.get(a["path"], (0, 0))
        if ea == sa:
            continue
        cs = cvecs[sa:ea] @ cvecs.T  # this note's chunks vs every chunk
        hit_cols = np.nonzero((cs >= th["chunk_match"]).any(axis=0))[0]
        cand = {chunks[c]["note"] for c in hit_cols}
        cand |= {notes[j]["path"] for j in np.nonzero(nvecs @ nvecs[i] >= th["near_dup"])[0]}
        for j, b in enumerate(notes):
            if j <= i or b["path"] not in cand:
                continue
            if a["path"] in exact_set and b["path"] in exact_set and a["text_hash"] == b["text_hash"]:
                continue
            sb, eb = ranges.get(b["path"], (0, 0))
            sim = cs[:, sb:eb]
            a_in_b, b_in_a = _containment(sim, th["chunk_match"])
            note_sim = float(nvecs[i] @ nvecs[j])
            kind = _classify(note_sim, a_in_b, b_in_a, th)
            if kind:
                pairs.append({"kind": kind, "a": a["path"], "b": b["path"], "score": r(note_sim),
                              "coverage": {"a_in_b": r(a_in_b), "b_in_a": r(b_in_a)},
                              "evidence": _evidence(chunks[sa:ea], cvecs[sa:ea], chunks[sb:eb], cvecs[sb:eb])})
    # Express containment from a's point of view: "contained_in" means a ⊂ b.
    pairs.sort(key=lambda d: (KIND_ORDER[d["kind"]], -d["score"]))

    img_pairs = []
    imgs, ivecs = data["images"], data["image_vecs"]
    if len(imgs) > 1:
        sims = ivecs @ ivecs.T
        for a, b in zip(*np.nonzero(np.triu(sims >= th["image_dup"], k=1))):
            exact = imgs[a]["hash"] == imgs[b]["hash"]
            img_pairs.append({"kind": "image_exact" if exact else "image_duplicate", "a": imgs[a]["path"],
                              "b": imgs[b]["path"], "score": 1.0 if exact else r(sims[a, b]),
                              "referenced_by": {"a": imgs[a]["referenced_by"], "b": imgs[b]["referenced_by"]}})
    return {"exact_groups": exact_groups, "pairs": pairs, "images": img_pairs,
            "counts": {"exact_groups": len(exact_groups), "pairs": len(pairs), "images": len(img_pairs)}}

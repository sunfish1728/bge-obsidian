"""Redundant passages: repeated inside the document, or already present elsewhere in the vault."""
from __future__ import annotations

import numpy as np

from ..index import Embedded
from .common import r, snippet


def redundant(data: dict, emb: Embedded, th: dict, self_path: str | None = None) -> dict:
    t = th["chunk_match"]
    cv = emb.chunk_vecs
    internal = []
    if len(cv) > 1:
        sim = cv @ cv.T
        for i, j in zip(*np.nonzero(np.triu(sim >= t, k=1))):
            internal.append({"chunk": int(i), "duplicate_of_chunk": int(j), "score": r(sim[i, j]),
                             "heading": emb.chunks[i].heading_path, "text": snippet(emb.chunks[i].text),
                             "other_heading": emb.chunks[j].heading_path,
                             "other_text": snippet(emb.chunks[j].text)})

    cross = []
    chunks, cvecs = data["chunks"], data["chunk_vecs"]
    if len(cv) and len(cvecs):
        keep = np.array([c["note"] != self_path for c in chunks])
        sim = cv @ cvecs.T
        sim[:, ~keep] = -1.0
        for i in range(len(cv)):
            j = int(sim[i].argmax())
            if sim[i, j] >= t:
                c = chunks[j]
                cross.append({"chunk": i, "score": r(sim[i, j]), "heading": emb.chunks[i].heading_path,
                              "text": snippet(emb.chunks[i].text), "found_in": c["note"],
                              "found_heading": c["heading"], "found_text": snippet(c["text"])})

    total_tokens = sum(c.tokens for c in emb.chunks) or 1
    red_chunks = {d["chunk"] for d in internal} | {d["chunk"] for d in cross}
    red_tokens = sum(emb.chunks[i].tokens for i in red_chunks)
    return {"chunks": len(emb.chunks), "internal": internal, "cross_vault": cross,
            "redundant_ratio": r(red_tokens / total_tokens)}

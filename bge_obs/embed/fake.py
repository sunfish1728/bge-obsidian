"""Deterministic test backend: hashed character n-grams. No model, no GPU.

Texts sharing many n-grams get high cosine similarity, which is enough to test the pipeline.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .base import DIM, Embedder, normalize


def _bucket(s: str) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little") % DIM


class FakeEmbedder(Embedder):
    name = "fake"
    supports_images = True

    def __init__(self, opts: dict | None = None):
        pass

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            t = "".join(t.lower().split())
            for n in (2, 3):
                for j in range(len(t) - n + 1):
                    out[i, _bucket(t[j : j + n])] += 1.0
            if not t:
                out[i, 0] = 1.0
        return normalize(out)

    def embed_images(self, images: list[Path | bytes]) -> np.ndarray:
        blobs = [p if isinstance(p, bytes) else Path(p).read_bytes() for p in images]
        return self.embed_texts([b.hex()[:4096] for b in blobs])

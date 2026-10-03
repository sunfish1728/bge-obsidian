"""Embedder interface shared by all backends.

All vectors are float32, L2-normalized, shape (n, DIM). Cosine similarity is a dot product.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

DIM = 1024
MODEL_FAMILY = "bge-m3"

# Fixed sentences used to verify that different backends produce compatible vectors.
PROBE_TEXTS = [
    "Obsidian 是一個以 Markdown 為基礎的個人知識管理工具。",
    "Transformer models use self-attention to process sequences in parallel.",
    "今天天氣很好，我們去公園散步吧。",
]


class EmbedderError(Exception):
    pass


class Embedder(ABC):
    name: str = "base"
    supports_images: bool = False

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return (len(texts), DIM) normalized float32 vectors."""

    def embed_images(self, images: list[Path | bytes]) -> np.ndarray:
        """Each item is an image file path or raw image bytes."""
        raise EmbedderError(f"backend '{self.name}' does not support image embedding")

    def close(self) -> None:
        pass


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def make_embedder(cfg) -> Embedder:
    backend = cfg["backend"]
    if backend == "local":
        from .local_visual import LocalVisualEmbedder

        return LocalVisualEmbedder(cfg["local"])
    if backend == "api":
        from .api import ApiEmbedder

        return ApiEmbedder(cfg["api"])
    if backend == "fake":
        from .fake import FakeEmbedder

        return FakeEmbedder()
    raise EmbedderError(f"unknown backend: {backend}")

"""API backend: any OpenAI-compatible /embeddings endpoint serving bge-m3 (text only)."""
from __future__ import annotations

import os
import time

import httpx
import numpy as np

from .base import DIM, Embedder, EmbedderError, normalize


class ApiEmbedder(Embedder):
    name = "api"
    supports_images = False

    def __init__(self, opts: dict):
        key_env = opts.get("api_key_env", "BGE_OBS_API_KEY")
        key = os.environ.get(key_env)
        if not key:
            raise EmbedderError(f"API key not found in environment variable {key_env}")
        self.url = opts["base_url"].rstrip("/") + "/embeddings"
        self.model = opts["model"]
        self.batch_size = int(opts.get("batch_size", 32))
        self.min_interval = 1.0 / float(opts.get("max_rps", 5))
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {key}"},
            timeout=float(opts.get("timeout", 60)),
        )
        self._last = 0.0

    def _post(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(5):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.client.post(self.url, json={"model": self.model, "input": batch})
            except httpx.TransportError as e:
                err = str(e)
            else:
                if r.status_code == 200:
                    data = sorted(r.json()["data"], key=lambda d: d["index"])
                    return [d["embedding"] for d in data]
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise EmbedderError(f"API error {r.status_code}: {r.text[:300]}")
                err = f"HTTP {r.status_code}"
            time.sleep(min(2**attempt, 20))
        raise EmbedderError(f"API request failed after retries: {err}")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        vecs: list[list[float]] = []
        for s in range(0, len(texts), self.batch_size):
            vecs.extend(self._post([t or " " for t in texts[s : s + self.batch_size]]))
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.shape[1] != DIM:
            raise EmbedderError(f"API returned dim {arr.shape[1]}, expected {DIM}")
        return normalize(arr)

    def close(self) -> None:
        self.client.close()

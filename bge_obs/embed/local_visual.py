"""Local backend: BAAI Visualized_m3 (bge-m3 text tower + EVA02-CLIP-L-14 vision tower).

The text tower is frozen in Visualized BGE training, so text vectors match plain bge-m3 dense vectors.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

from ..config import PROJECT_ROOT
from .base import DIM, Embedder, EmbedderError, normalize

VISUAL_BGE_SRC = PROJECT_ROOT / "vendor" / "FlagEmbedding" / "research" / "visual_bge"


class LocalVisualEmbedder(Embedder):
    name = "local"
    supports_images = True

    def __init__(self, opts: dict):
        try:
            import torch
        except ImportError as e:  # pragma: no cover
            raise EmbedderError("torch is not installed; install the 'local' extras") from e

        text_model = Path(opts["text_model"])
        weight = Path(opts["visual_weight"])
        if not weight.exists():
            raise EmbedderError(f"Visualized_m3 weight not found: {weight} (run scripts/download_model.py)")
        if not (text_model / "config.json").exists():
            raise EmbedderError(f"bge-m3 config/tokenizer not found in {text_model}")
        if not VISUAL_BGE_SRC.exists():
            raise EmbedderError(f"visual_bge source not found: {VISUAL_BGE_SRC}")
        if str(VISUAL_BGE_SRC) not in sys.path:
            sys.path.insert(0, str(VISUAL_BGE_SRC))
        from visual_bge.modeling import Visualized_BGE

        want_cuda = opts.get("device", "cuda") == "cuda"
        if want_cuda and not torch.cuda.is_available():
            raise EmbedderError("device=cuda requested but CUDA is not available")

        model = Visualized_BGE(model_name_bge=str(text_model), model_weight=str(weight))
        # Visualized_BGE moves itself to CUDA when available; honour device=cpu explicitly.
        device = torch.device("cuda" if want_cuda else "cpu")
        model.to(device)
        model.device = device
        if opts.get("fp16", True) and device.type == "cuda":
            model.half()
            model.dtype = torch.float16  # used to build attention masks
        model.eval()

        self._torch = torch
        self.model = model
        self.device = device
        self.batch_size = int(opts.get("batch_size", 16))
        self.max_tokens = int(opts.get("max_tokens", 1024))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        torch = self._torch
        tok = self.model.tokenizer
        # Sort by length so each batch pads to a similar size.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        with torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idx = order[s : s + self.batch_size]
                batch = tok(
                    [texts[i] or " " for i in idx],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_tokens,
                ).to(self.device)
                vec = self.model.encode_text(batch).float().cpu().numpy()
                out[idx] = vec
        return normalize(out)

    def embed_images(self, images: list[Path | bytes]) -> np.ndarray:
        if not images:
            return np.zeros((0, DIM), dtype=np.float32)
        torch = self._torch
        from PIL import Image

        out = np.zeros((len(images), DIM), dtype=np.float32)
        bs = max(1, self.batch_size // 2)
        with torch.inference_mode():
            for s in range(0, len(images), bs):
                chunk = images[s : s + bs]
                imgs = []
                for p in chunk:
                    with Image.open(io.BytesIO(p) if isinstance(p, bytes) else p) as im:
                        imgs.append(self.model.preprocess_val(im.convert("RGB")))
                pix = torch.stack(imgs).to(self.device)
                if self.model.dtype == torch.float16:
                    pix = pix.half()
                out[s : s + len(chunk)] = self.model.encode_image(pix).float().cpu().numpy()
        return normalize(out)

    def close(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

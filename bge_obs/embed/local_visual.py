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


def resolve_device(requested: str, cuda_available: bool) -> str:
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda":
        if not cuda_available:
            raise EmbedderError("CUDA is unavailable; set device: auto or device: cpu")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    raise EmbedderError(f"Unknown device: {requested}; use auto, cuda or cpu")


def resolve_precision(fp16_opt, device: str) -> bool:
    if fp16_opt == "auto":
        return device == "cuda"
    if not isinstance(fp16_opt, bool):
        raise EmbedderError("fp16 must be auto, true or false")
    return fp16_opt and device == "cuda"


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

        self.device_name = resolve_device(opts.get("device", "auto"), torch.cuda.is_available())
        use_fp16 = resolve_precision(opts.get("fp16", "auto"), self.device_name)
        self.precision = "fp16" if use_fp16 else "fp32"
        threads = opts.get("threads")
        if threads is not None and self.device_name == "cpu":
            torch.set_num_threads(int(threads))

        # Map checkpoint pages instead of allocating a second full CPU copy.
        class MappedVisualBGE(Visualized_BGE):
            def load_model(self, model_weight):
                self.load_state_dict(torch.load(model_weight, map_location="cpu", mmap=True))

        model = MappedVisualBGE(model_name_bge=str(text_model), model_weight=str(weight))
        # Visualized_BGE moves itself to CUDA when available; honour device=cpu explicitly.
        device = torch.device(self.device_name)
        dtype = torch.float16 if use_fp16 else torch.float32
        model.to(device=device, dtype=dtype)
        model.device = device
        model.dtype = dtype  # used to build attention masks on either device
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

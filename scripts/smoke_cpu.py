"""Offline CPU smoke test using existing project models (no installation)."""
import io
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
os.environ.update(HF_HOME=str(ROOT / "models" / ".hf-cache"),
                  PIP_CACHE_DIR=str(ROOT / ".pip-cache"),
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
sys.path.insert(0, str(ROOT))

from PIL import Image
from bge_obs.config import load_config
from bge_obs.embed.base import make_embedder


def main():
    started = time.perf_counter()
    cfg = load_config(overrides={"vault": str(ROOT / "state" / "demo-vault"),
                                 "local": {"device": "cpu"}})
    emb = make_embedder(cfg)
    loaded = time.perf_counter()
    texts = emb.embed_texts(["今天天氣很好，我們去公園散步。", "天氣晴朗，適合到公園走走。"])
    text_done = time.perf_counter()
    png = io.BytesIO()
    Image.new("RGB", (64, 64), (80, 160, 220)).save(png, format="PNG")
    images = emb.embed_images([png.getvalue()])
    finished = time.perf_counter()
    print(f"device={emb.device_name} precision={emb.precision}")
    print(f"text_shape={texts.shape} image_shape={images.shape}")
    print(f"cosine={float(texts[0] @ texts[1]):.6f}")
    print(f"load_s={loaded-started:.3f} text_s={text_done-loaded:.3f} "
          f"image_s={finished-text_done:.3f} total_s={finished-started:.3f}")
    emb.close()


if __name__ == "__main__":
    main()

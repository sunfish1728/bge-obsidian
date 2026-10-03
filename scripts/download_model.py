"""Download Visualized_m3 weights and the bge-m3 config/tokenizer into <project>/models (no global cache)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
os.environ.setdefault("HF_HOME", str(MODELS / ".hf-cache"))

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    print(hf_hub_download("BAAI/bge-visualized", "Visualized_m3.pth", local_dir=MODELS))
    print(snapshot_download("BAAI/bge-m3", local_dir=MODELS / "bge-m3",
                            allow_patterns=["config.json", "tokenizer*", "sentencepiece*", "special_tokens_map.json"]))


if __name__ == "__main__":
    main()

"""Download Visualized_m3 weights and the bge-m3 config/tokenizer into <project>/models (no global cache)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
os.environ["HF_HOME"] = str(MODELS / ".hf-cache")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    weight = MODELS / "Visualized_m3.pth"
    if weight.is_file():
        print(f"Keep existing {weight}")
    else:
        print(hf_hub_download("BAAI/bge-visualized", weight.name, local_dir=MODELS))
    # Discover matching config/tokenizer files, then download only missing ones.
    import fnmatch

    patterns = ["config.json", "tokenizer*", "sentencepiece*", "special_tokens_map.json"]
    directory = MODELS / "bge-m3"
    required = ["config.json", "tokenizer_config.json", "sentencepiece.bpe.model"]
    if all((directory / name).is_file() for name in required):
        print(f"Keep existing {directory}")
        return
    for name in HfApi().list_repo_files("BAAI/bge-m3"):
        if not any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            continue
        if (directory / name).is_file():
            print(f"Keep existing {directory / name}")
        else:
            print(hf_hub_download("BAAI/bge-m3", name, local_dir=directory))


if __name__ == "__main__":
    main()

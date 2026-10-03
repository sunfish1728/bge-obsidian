"""Configuration loading.

Lookup order: explicit path > $BGE_OBS_CONFIG > <project>/config.yaml.
User values are deep-merged over DEFAULTS. All state lives inside the project folder.
"""
from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_ROOT = PROJECT_ROOT / "state"

# Keep Hugging Face caches inside the project, never in the user profile.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / ".hf-cache"))

DEFAULTS: dict[str, Any] = {
    "vault": None,
    "inbox": "_inbox",
    "review_dir": "_review",
    "exclude": [".obsidian", ".trash", ".claude", ".agents", "_review", "templates"],
    "extensions": [".md", ".pdf", ".txt"],
    "image_extensions": [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"],
    "tags_file": None,  # defaults to <project>/state/<vault>/tags.yaml
    "backend": "local",
    "local": {
        "text_model": str(PROJECT_ROOT / "models" / "bge-m3"),
        "visual_weight": str(PROJECT_ROOT / "models" / "Visualized_m3.pth"),
        "device": "auto",
        "fp16": "auto",
        "threads": None,
        "batch_size": 16,
        "max_tokens": 1024,
    },
    "api": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "BAAI/bge-m3",
        "api_key_env": "BGE_OBS_API_KEY",
        "batch_size": 32,
        "max_rps": 5,
        "timeout": 60,
    },
    "chunk": {"min_tokens": 300, "max_tokens": 800, "hard_max": 1024},
    "thresholds": {
        "dup": 0.95,
        "near_dup": 0.85,
        "chunk_match": 0.92,
        "contained_ratio": 0.80,
        "image_dup": 0.97,
        "classify_min": 0.55,
        "classify_margin": 0.03,
        "tag_min": 0.55,
        "probe_min": 0.999,
    },
    "folders": {},
    "server": {"idle_minutes": 30, "start_timeout": 300},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def default_config_path() -> Path:
    env = os.environ.get("BGE_OBS_CONFIG")
    if env:
        return Path(env)
    return PROJECT_ROOT / "config.yaml"


@dataclass
class Config:
    data: dict[str, Any]
    path: Path | None

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @property
    def vault(self) -> Path:
        v = self.data.get("vault")
        if not v:
            raise ConfigError(f"'vault' is not set in config ({self.path})")
        return Path(v).expanduser().resolve()

    @property
    def state_dir(self) -> Path:
        """Per-vault state directory inside the project (index, tags)."""
        v = self.vault
        digest = hashlib.sha1(str(v).lower().encode("utf-8")).hexdigest()[:8]
        return STATE_ROOT / f"{v.name}-{digest}"

    @property
    def index_path(self) -> Path:
        return self.state_dir / "index.sqlite"

    @property
    def tags_path(self) -> Path:
        t = self.data.get("tags_file")
        return Path(t).expanduser() if t else self.state_dir / "tags.yaml"

    @property
    def thresholds(self) -> dict[str, float]:
        return self.data["thresholds"]


class ConfigError(Exception):
    pass


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    p = Path(path) if path else default_config_path()
    user: dict = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    elif path:
        raise ConfigError(f"config file not found: {p}")
    data = _deep_merge(DEFAULTS, user)
    if overrides:
        data = _deep_merge(data, overrides)
    return Config(data=data, path=p if p.exists() else None)

import os
from pathlib import Path
import subprocess
import sys

import pytest

from bge_obs.embed.base import EmbedderError
from bge_obs.embed.local_visual import resolve_device, resolve_precision


@pytest.mark.parametrize("requested,available,expected", [
    ("auto", True, "cuda"), ("auto", False, "cpu"),
    ("cuda", True, "cuda"), ("cpu", True, "cpu"), ("cpu", False, "cpu"),
])
def test_device(requested, available, expected):
    assert resolve_device(requested, available) == expected


def test_cuda_unavailable():
    with pytest.raises(EmbedderError, match="device: auto or device: cpu"):
        resolve_device("cuda", False)


def test_invalid_device():
    with pytest.raises(EmbedderError, match="Unknown device"):
        resolve_device("invalid", True)


@pytest.mark.parametrize("option,device,expected", [
    ("auto", "cuda", True), ("auto", "cpu", False),
    (True, "cuda", True), (True, "cpu", False),
    (False, "cuda", False), (False, "cpu", False),
])
def test_precision(option, device, expected):
    assert resolve_precision(option, device) is expected


def test_invalid_precision():
    with pytest.raises(EmbedderError, match="fp16 must"):
        resolve_precision("true", "cpu")


def test_setup_dry_run():
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env.update(PIP_CACHE_DIR=str(root / ".pip-cache"),
               HF_HOME=str(root / "models" / ".hf-cache"))
    watched = [root / ".venv" / "pyvenv.cfg", root / "config.yaml"]
    before = [(p.read_bytes(), p.stat().st_mtime_ns) if p.exists() else None for p in watched]
    result = subprocess.run([sys.executable, "scripts/setup_env.py", "--dry-run", "--cpu", "--skip-models"],
                            cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "https://download.pytorch.org/whl/cpu" in result.stdout
    assert before == [(p.read_bytes(), p.stat().st_mtime_ns) if p.exists() else None for p in watched]

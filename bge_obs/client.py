"""Find or start the resident server and send it a command."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .config import PROJECT_ROOT, STATE_ROOT
from .server import server_file

START_TIMEOUT = 180  # model load can take ~20s cold, longer on first CUDA init


def _read(sf: Path) -> dict | None:
    try:
        return json.loads(sf.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _alive(info: dict | None) -> bool:
    if not info:
        return False
    try:
        r = httpx.get(f"http://127.0.0.1:{info['port']}/", headers={"X-Token": info["token"]}, timeout=2)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def _spawn(config_path: Path | None) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    log = open(STATE_ROOT / "server.log", "ab")
    args = [sys.executable, "-m", "bge_obs.server"]
    if config_path:
        args += ["--config", str(config_path)]
    kw: dict = {"cwd": str(PROJECT_ROOT), "stdout": log, "stderr": log, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kw["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP \
            | subprocess.CREATE_NO_WINDOW
    else:
        kw["start_new_session"] = True
    subprocess.Popen(args, **kw)


def ensure_server(config_path: Path | None) -> dict:
    sf = server_file(config_path)
    info = _read(sf)
    if _alive(info):
        return info
    lock = STATE_ROOT / (sf.stem + ".lock")
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        owner = True
    except FileExistsError:  # someone else is starting it; a stale lock expires
        owner = time.time() - lock.stat().st_mtime > START_TIMEOUT
        if owner:
            lock.touch()
    try:
        if owner:
            if sf.exists():
                sf.unlink()
            _spawn(config_path)
        deadline = time.time() + START_TIMEOUT
        while time.time() < deadline:
            info = _read(sf)
            if _alive(info):
                return info
            time.sleep(0.5)
        raise RuntimeError(f"server did not start within {START_TIMEOUT}s; see {STATE_ROOT / 'server.log'}")
    finally:
        if owner:
            lock.unlink(missing_ok=True)


def call(config_path: Path | None, cmd: str, args: dict, timeout: float = 3600) -> dict:
    info = ensure_server(config_path)
    r = httpx.post(f"http://127.0.0.1:{info['port']}/", headers={"X-Token": info["token"]},
                   json={"cmd": cmd, "args": args}, timeout=timeout)
    return r.json()


def stop(config_path: Path | None) -> dict:
    info = _read(server_file(config_path))
    if not _alive(info):
        return {"ok": True, "running": False}
    httpx.post(f"http://127.0.0.1:{info['port']}/", headers={"X-Token": info["token"]},
               json={"cmd": "stop"}, timeout=10)
    return {"ok": True, "stopped": True}

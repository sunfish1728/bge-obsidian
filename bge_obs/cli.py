"""bge-obs command line. Every command prints one JSON object to stdout."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, load_config


def _abs(p: str | None) -> str | None:
    """Resolve paths on the caller's side: the server runs with a different cwd."""
    if p is None:
        return None
    path = Path(p)
    return str(path.resolve()) if path.exists() else p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="bge-obs", description="Embedding-based preprocessing for Obsidian")
    ap.add_argument("--config", help="config.yaml (default: <project>/config.yaml or $BGE_OBS_CONFIG)")
    ap.add_argument("--no-server", action="store_true", help="run in-process instead of the resident server")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="server/index status and backend compatibility")
    p = sub.add_parser("index", help="incrementally update the vault index")
    p.add_argument("--rebuild", action="store_true", help="drop and rebuild the whole index")
    p = sub.add_parser("ingest", help="extract + dedup + classify + tags + similar for new material")
    p.add_argument("files", nargs="+")
    p.add_argument("--k", type=int, default=5, help="number of similar notes")
    p = sub.add_parser("dedup", help="duplicates of one file, or all pairs in the vault")
    p.add_argument("file", nargs="?")
    p.add_argument("--all", action="store_true")
    p = sub.add_parser("redundant", help="passages repeated inside the file or already in the vault")
    p.add_argument("file")
    p = sub.add_parser("classify", help="folder candidates")
    p.add_argument("file")
    p = sub.add_parser("tags", help="tag candidates from tags.yaml / vault usage")
    p.add_argument("file")
    p.add_argument("--k", type=int, default=5)
    p = sub.add_parser("suggest-tags", help="cluster the vault to propose a tag list")
    p.add_argument("--k", type=int, default=None, help="fixed number of clusters (default: automatic)")
    p = sub.add_parser("similar", help="nearest notes for a file, text, or image")
    p.add_argument("file", nargs="?")
    p.add_argument("--text")
    p.add_argument("--image")
    p.add_argument("--k", type=int, default=10)
    p = sub.add_parser("extract", help="cleaned Markdown text of a file (no model needed)")
    p.add_argument("file")
    p.add_argument("--max-chars", type=int, default=None)
    sub.add_parser("serve", help="run the resident server in the foreground")
    sub.add_parser("stop", help="stop the resident server")
    return ap


def to_call(a: argparse.Namespace) -> tuple[str, dict]:
    c = a.cmd
    if c == "status":
        return c, {}
    if c == "index":
        return c, {"rebuild": a.rebuild}
    if c == "ingest":
        return c, {"files": [_abs(f) for f in a.files], "k": a.k}
    if c == "dedup":
        return c, {"file": _abs(a.file), "all_": a.all}
    if c in ("redundant", "classify"):
        return c, {"file": _abs(a.file)}
    if c == "tags":
        return c, {"file": _abs(a.file), "k": a.k}
    if c == "suggest-tags":
        return c, {"k": a.k}
    if c == "similar":
        return c, {"file": _abs(a.file), "text": a.text, "image": _abs(a.image), "k": a.k}
    if c == "extract":
        return c, {"file": _abs(a.file), "max_chars": a.max_chars}
    raise ValueError(c)


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    a = build_parser().parse_args(argv)
    try:
        cfg = load_config(a.config)
        cfg.vault  # fail early with a clear message
    except ConfigError as e:
        emit({"ok": False, "error": {"code": "config", "message": str(e)}})
        return 2

    if a.cmd == "serve":
        from .server import run

        run(str(cfg.path) if cfg.path else None)
        return 0
    if a.cmd == "stop":
        from .client import stop

        emit(stop(cfg.path))
        return 0

    cmd, args = to_call(a)
    # extract needs no model: always run in-process.
    if a.no_server or cmd == "extract":
        from .service import CommandError, Service

        svc = Service(cfg)
        try:
            res = svc.dispatch(cmd, args)
            res = res if "ok" in res else {"ok": True, **res}
        except CommandError as e:
            res = {"ok": False, "error": {"code": e.code, "message": e.message, **e.extra}}
        finally:
            svc.close()
    else:
        from .client import call

        try:
            res = call(cfg.path, cmd, args)
        except Exception as e:
            res = {"ok": False, "error": {"code": "server", "message": str(e)}}
    emit(res)
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

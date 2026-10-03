"""Resident local server: keeps the model loaded between CLI calls.

Binds 127.0.0.1 on a random port, requires a per-run token, exits after `server.idle_minutes` idle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import STATE_ROOT, load_config
from .service import CommandError, Service


def server_file(config_path: Path | None) -> Path:
    key = hashlib.sha1(str(config_path or "default").lower().encode("utf-8")).hexdigest()[:8]
    return STATE_ROOT / f"server-{key}.json"


def run(config_path: str | None) -> None:
    cfg = load_config(config_path)
    svc = Service(cfg)
    lock = threading.Lock()  # one GPU job at a time
    token = secrets.token_hex(16)
    last = [time.time()]
    idle = float(cfg["server"]["idle_minutes"]) * 60

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep stderr quiet
            pass

        def _send(self, code: int, body: dict) -> None:
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.headers.get("X-Token") != token:
                return self._send(403, {"ok": False})
            self._send(200, {"ok": True, "pid": os.getpid(), "ready": True})

        def do_POST(self):
            if self.headers.get("X-Token") != token:
                return self._send(403, {"ok": False, "error": {"code": "forbidden"}})
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if req.get("cmd") == "stop":
                self._send(200, {"ok": True, "stopped": True})
                threading.Thread(target=httpd.shutdown, daemon=True).start()
                return
            with lock:
                last[0] = time.time()
                try:
                    result = svc.dispatch(req.get("cmd", ""), req.get("args") or {})
                    body = result if "ok" in result else {"ok": True, **result}
                    code = 200
                except CommandError as e:
                    body, code = {"ok": False, "error": {"code": e.code, "message": e.message, **e.extra}}, 200
                except Exception as e:
                    body = {"ok": False, "error": {"code": type(e).__name__, "message": str(e),
                                                   "trace": traceback.format_exc(limit=5)}}
                    code = 500
                last[0] = time.time()
            self._send(code, body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]

    # Load the model before announcing readiness so the first call is fast.
    try:
        svc.embedder
    except Exception as e:
        print(f"failed to load embedder: {e}", file=sys.stderr)
        raise

    sf = server_file(cfg.path)
    sf.parent.mkdir(parents=True, exist_ok=True)
    sf.write_text(json.dumps({"port": port, "pid": os.getpid(), "token": token, "started": time.time()}),
                  encoding="utf-8")

    def watchdog():
        while True:
            time.sleep(15)
            if time.time() - last[0] > idle and not lock.locked():
                httpd.shutdown()
                return

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        httpd.serve_forever()
    finally:
        try:
            if json.loads(sf.read_text(encoding="utf-8")).get("pid") == os.getpid():
                sf.unlink()
        except (OSError, ValueError):
            pass
        svc.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    a = ap.parse_args()
    run(a.config)


if __name__ == "__main__":
    main()

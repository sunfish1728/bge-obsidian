"""All commands as plain functions returning JSON-serialisable dicts. Used by the server and --no-server."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from .analyze.classify import classify
from .analyze.dedup import dedup_all, dedup_doc, dedup_images
from .analyze.redundant import redundant
from .analyze.similar import similar, similar_images
from .analyze.tags import propose_tag_list, suggest_tags
from .embed.base import Embedder, make_embedder
from .extract import extract
from .index import Embedded, Index, embed_documents, file_hash


class CommandError(Exception):
    def __init__(self, code: str, message: str, **extra):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra


class Service:
    def __init__(self, cfg):
        self.cfg = cfg
        self._embedder: Embedder | None = None
        self._index: Index | None = None
        self.started = time.time()

    # ---------- lazy resources ----------

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = make_embedder(self.cfg)
        return self._embedder

    @property
    def index(self) -> Index:
        if self._index is None:
            self._index = Index(self.cfg)
        return self._index

    def close(self) -> None:
        if self._embedder:
            self._embedder.close()
        if self._index:
            self._index.close()

    # ---------- helpers ----------

    def _data(self, refresh: bool = True) -> dict:
        idx = self.index
        if idx.count() == 0:
            raise CommandError("index_empty", "index is empty; run `bge-obs index` first")
        if refresh:
            res = idx.update(self.embedder)
            if not res.get("ok"):
                raise CommandError(res.get("error", "index_error"), "index refresh failed", **res)
        return idx.load()

    def _path(self, p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            cand = self.cfg.vault / path
            path = cand if cand.exists() else path.resolve()
        if not path.exists():
            raise CommandError("not_found", f"file not found: {p}")
        return path

    def _self_rel(self, path: Path) -> str | None:
        try:
            return path.resolve().relative_to(self.cfg.vault).as_posix()
        except ValueError:
            return None

    def _embed(self, path: Path) -> tuple[Embedded, str | None]:
        doc = extract(path, vault=self.cfg.vault)
        emb = embed_documents(self.embedder, [doc], self.cfg["chunk"])[0]
        return emb, (file_hash(path) if path.is_file() else None)

    def _doc_images(self, emb: Embedded) -> tuple[np.ndarray, list[str], list[str | None], set[str]]:
        items, names, hashes, own = [], [], [], set()
        for im in emb.doc.images:
            if im.data is not None:
                items.append(im.data)
                hashes.append(hashlib.sha1(im.data).hexdigest())
            elif im.path is not None:
                items.append(im.path)
                hashes.append(file_hash(im.path))
                rel = self._self_rel(im.path)
                if rel:
                    own.add(rel)
            else:
                continue
            names.append(im.name)
        if not items or not self.embedder.supports_images:
            return np.zeros((0, 1024), np.float32), [], [], own
        return self.embedder.embed_images(items), names, hashes, own

    def _extract_summary(self, emb: Embedded) -> dict:
        d = emb.doc
        return {"kind": d.kind, "title": d.title, "chars": len(d.text), "chunks": len(emb.chunks),
                "images": len(d.images), "needs_ocr": d.needs_ocr, "removed": d.removed,
                "existing_tags": d.tags, **{k: v for k, v in d.meta.items() if k != "files"}}

    # ---------- commands ----------

    def status(self) -> dict:
        idx = self.index
        out = {"vault": str(self.cfg.vault), "backend": self.cfg["backend"], "config": str(self.cfg.path),
               "index_path": str(self.cfg.index_path), "notes": idx.count(),
               "chunks": idx.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
               "images": idx.db.execute("SELECT COUNT(*) FROM images").fetchone()[0],
               "tags_file": str(self.cfg.tags_path), "tags_file_exists": self.cfg.tags_path.exists(),
               "uptime_s": round(time.time() - self.started)}
        if idx.count():
            out["probe"] = idx.check_probe(self.embedder)
        if self._embedder is not None or self.cfg["backend"] == "local":
            emb = self.embedder
            out["embedder"] = {"backend": emb.name,
                               "device": getattr(emb, "device_name", None),
                               "precision": getattr(emb, "precision", None)}
        return out

    def index_cmd(self, rebuild: bool = False) -> dict:
        return self.index.update(self.embedder, rebuild=rebuild)

    def ingest(self, files: list[str], k: int = 5) -> dict:
        data = self._data()
        th = self.cfg.thresholds
        results = []
        for f in files:
            try:
                path = self._path(f)
                emb, fh = self._embed(path)
                me = self._self_rel(path)
                ivecs, inames, ihashes, own = self._doc_images(emb)
                red = redundant(data, emb, th, me)
                results.append({
                    "ok": True, "file": str(path), "in_vault": me,
                    "extract": self._extract_summary(emb),
                    "dedup": dedup_doc(data, emb, fh, th, me),
                    "image_dedup": dedup_images(data, ivecs, inames, ihashes, th, own),
                    "redundant": {"internal": len(red["internal"]), "cross_vault": len(red["cross_vault"]),
                                  "redundant_ratio": red["redundant_ratio"],
                                  "top_cross_vault": red["cross_vault"][:5]},
                    "classify": classify(data, emb.vec, self.embedder, self.cfg, me),
                    "tags": suggest_tags(data, emb.vec, self.embedder, self.cfg, emb.doc.tags, me),
                    "similar": similar(data, emb.vec, k, me, emb.chunk_vecs),
                })
            except CommandError as e:
                results.append({"ok": False, "file": f, "error": {"code": e.code, "message": e.message}})
            except Exception as e:
                results.append({"ok": False, "file": f, "error": {"code": type(e).__name__, "message": str(e)}})
        return {"results": results}

    def dedup(self, file: str | None = None, all_: bool = False) -> dict:
        data = self._data()
        if all_ or not file:
            return dedup_all(data, self.cfg.thresholds)
        path = self._path(file)
        emb, fh = self._embed(path)
        me = self._self_rel(path)
        ivecs, inames, ihashes, own = self._doc_images(emb)
        return {"file": str(path), "dedup": dedup_doc(data, emb, fh, self.cfg.thresholds, me),
                "image_dedup": dedup_images(data, ivecs, inames, ihashes, self.cfg.thresholds, own)}

    def redundant(self, file: str) -> dict:
        data = self._data()
        path = self._path(file)
        emb, _ = self._embed(path)
        return {"file": str(path), **redundant(data, emb, self.cfg.thresholds, self._self_rel(path))}

    def classify(self, file: str) -> dict:
        data = self._data()
        path = self._path(file)
        emb, _ = self._embed(path)
        return {"file": str(path), **classify(data, emb.vec, self.embedder, self.cfg, self._self_rel(path))}

    def tags(self, file: str, k: int = 5) -> dict:
        data = self._data()
        path = self._path(file)
        emb, _ = self._embed(path)
        return {"file": str(path),
                **suggest_tags(data, emb.vec, self.embedder, self.cfg, emb.doc.tags, self._self_rel(path), k)}

    def suggest_tags(self, k: int | None = None) -> dict:
        return propose_tag_list(self._data(), k=k)

    def similar(self, file: str | None = None, text: str | None = None, image: str | None = None,
                k: int = 10) -> dict:
        data = self._data()
        if image:
            path = self._path(image)
            v = self.embedder.embed_images([path])[0]
            return {"query": {"image": str(path)}, "images": similar_images(data, v, k)}
        if text:
            v = self.embedder.embed_texts([text])[0]
            return {"query": {"text": text}, "notes": similar(data, v, k)}
        if not file:
            raise CommandError("bad_args", "give a file, --text or --image")
        path = self._path(file)
        emb, _ = self._embed(path)
        return {"query": {"file": str(path)},
                "notes": similar(data, emb.vec, k, self._self_rel(path), emb.chunk_vecs)}

    def extract_cmd(self, file: str, max_chars: int | None = None) -> dict:
        path = self._path(file)
        d = extract(path, vault=self.cfg.vault)
        text = d.text if not max_chars else d.text[:max_chars]
        return {"file": str(path), "kind": d.kind, "title": d.title, "frontmatter": d.frontmatter,
                "tags": d.tags, "removed": d.removed, "needs_ocr": d.needs_ocr, "meta": d.meta,
                "images": [{"name": i.name, "path": str(i.path) if i.path else None, "page": i.page}
                           for i in d.images],
                "truncated": bool(max_chars and len(d.text) > max_chars), "text": text}

    def dispatch(self, cmd: str, args: dict) -> dict:
        table = {
            "status": self.status, "index": self.index_cmd, "ingest": self.ingest, "dedup": self.dedup,
            "redundant": self.redundant, "classify": self.classify, "tags": self.tags,
            "suggest-tags": self.suggest_tags, "similar": self.similar, "extract": self.extract_cmd,
        }
        if cmd not in table:
            raise CommandError("unknown_command", f"unknown command: {cmd}")
        return table[cmd](**args)

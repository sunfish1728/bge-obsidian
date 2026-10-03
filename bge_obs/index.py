"""SQLite vector index of a vault: notes, chunks and images.

Keys are vault-relative POSIX paths. Vectors are float32 BLOBs, loaded into numpy for brute-force search.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunk import chunk_text
from .embed.base import DIM, MODEL_FAMILY, PROBE_TEXTS, Embedder, normalize
from .extract import extract
from .types import Chunk, Extracted

SCHEMA_VERSION = "1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    path TEXT PRIMARY KEY, hash TEXT, text_hash TEXT, mtime REAL, title TEXT, folder TEXT,
    kind TEXT, tags_json TEXT, vec BLOB, indexed_at REAL
);
CREATE TABLE IF NOT EXISTS chunks (
    note_path TEXT, idx INTEGER, heading_path TEXT, text TEXT, tokens INTEGER, vec BLOB,
    PRIMARY KEY (note_path, idx)
);
CREATE TABLE IF NOT EXISTS images (
    path TEXT PRIMARY KEY, hash TEXT, mtime REAL, vec BLOB, referenced_by_json TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def file_hash(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()


def _blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


@dataclass
class Embedded:
    """A document turned into vectors, whether or not it is stored in the index."""

    doc: Extracted
    chunks: list[Chunk]
    chunk_vecs: np.ndarray  # (n_chunks, DIM)
    vec: np.ndarray  # (DIM,) note-level vector


def note_vector(chunk_vecs: np.ndarray, tokens: list[int], head_vec: np.ndarray) -> np.ndarray:
    """Length-weighted mean of chunk vectors, averaged with a title+opening vector."""
    if len(chunk_vecs) == 0:
        return normalize(head_vec)
    w = np.asarray(tokens, dtype=np.float32).clip(min=1)[:, None]
    body = normalize((chunk_vecs * w).sum(axis=0))
    return normalize(body + normalize(head_vec))


def head_text(doc: Extracted) -> str:
    return f"{doc.title}\n{doc.text[:500]}"


def embed_documents(embedder: Embedder, docs: list[Extracted], chunk_opts: dict) -> list[Embedded]:
    """Chunk and embed several documents with one batched embedding call."""
    all_chunks = [chunk_text(d.text, **chunk_opts) for d in docs]
    texts: list[str] = []
    for d, chunks in zip(docs, all_chunks):
        texts.append(head_text(d))
        texts.extend(c.text for c in chunks)
    vecs = embedder.embed_texts(texts) if texts else np.zeros((0, DIM), np.float32)
    out, pos = [], 0
    for d, chunks in zip(docs, all_chunks):
        head = vecs[pos]
        cv = vecs[pos + 1 : pos + 1 + len(chunks)]
        pos += 1 + len(chunks)
        out.append(Embedded(d, chunks, cv, note_vector(cv, [c.tokens for c in chunks], head)))
    return out


class Index:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vault: Path = cfg.vault
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cfg.index_path, check_same_thread=False)  # server serialises calls
        self.db.executescript(SCHEMA)
        self._cache: dict | None = None

    # ---------- meta / probe ----------

    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    def check_probe(self, embedder: Embedder) -> dict:
        """Compare the backend's probe vectors with the ones stored when the index was built."""
        cur = embedder.embed_texts(PROBE_TEXTS)
        stored = self.get_meta("probe_vecs")
        if stored is None:
            self.set_meta("probe_vecs", json.dumps(cur.tolist()))
            self.set_meta("probe_backend", embedder.name)
            self.set_meta("model_family", MODEL_FAMILY)
            self.set_meta("dim", str(DIM))
            self.set_meta("schema_version", SCHEMA_VERSION)
            self.db.commit()
            return {"compatible": True, "min_cos": 1.0, "first_use": True}
        ref = np.asarray(json.loads(stored), dtype=np.float32)
        cos = float((ref * cur).sum(axis=1).min())
        ok = cos >= self.cfg.thresholds["probe_min"]
        return {
            "compatible": ok,
            "min_cos": round(cos, 5),
            "index_backend": self.get_meta("probe_backend"),
            "current_backend": embedder.name,
        }

    # ---------- file discovery ----------

    def _excluded(self, rel: Path) -> bool:
        excl = set(self.cfg["exclude"]) | {self.cfg["inbox"]}
        return any(part in excl for part in rel.parts)

    def scan_files(self) -> tuple[list[Path], list[Path]]:
        """Return (documents, images) under the vault, honouring exclusions."""
        doc_ext = {e.lower() for e in self.cfg["extensions"]}
        img_ext = {e.lower() for e in self.cfg["image_extensions"]}
        docs, imgs = [], []
        for p in self.vault.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(self.vault)
            if self._excluded(rel):
                continue
            ext = p.suffix.lower()
            if ext in doc_ext:
                docs.append(p)
            elif ext in img_ext:
                imgs.append(p)
        return sorted(docs), sorted(imgs)

    def rel(self, p: Path) -> str:
        return p.resolve().relative_to(self.vault).as_posix()

    # ---------- update ----------

    def update(self, embedder: Embedder, rebuild: bool = False, batch_docs: int = 16) -> dict:
        t0 = time.time()
        if rebuild:
            for t in ("notes", "chunks", "images", "meta"):
                self.db.execute(f"DELETE FROM {t}")
            self.db.commit()
        probe = self.check_probe(embedder)
        if not probe["compatible"]:
            return {"ok": False, "error": "backend_incompatible", "probe": probe,
                    "hint": "run `bge-obs index --rebuild` or switch backend"}

        docs, imgs = self.scan_files()
        known = {r[0]: (r[1], r[2]) for r in self.db.execute("SELECT path, mtime, hash FROM notes")}
        seen: set[str] = set()
        todo: list[tuple[Path, str]] = []
        stats = {"added": 0, "updated": 0, "unchanged": 0, "removed": 0, "failed": [], "needs_ocr": []}

        for p in docs:
            rel = self.rel(p)
            seen.add(rel)
            mtime = p.stat().st_mtime
            if rel in known and known[rel][0] == mtime:
                stats["unchanged"] += 1
                continue
            h = file_hash(p)
            if rel in known and known[rel][1] == h:
                self.db.execute("UPDATE notes SET mtime=? WHERE path=?", (mtime, rel))
                stats["unchanged"] += 1
                continue
            todo.append((p, h))

        for rel in set(known) - seen:
            self._delete_note(rel)
            stats["removed"] += 1

        for s in range(0, len(todo), batch_docs):
            batch = todo[s : s + batch_docs]
            extracted = []
            for p, h in batch:
                try:
                    extracted.append((extract(p, vault=self.vault), h))
                except Exception as e:  # keep indexing other files
                    stats["failed"].append({"path": self.rel(p), "error": f"{type(e).__name__}: {e}"})
            embedded = embed_documents(embedder, [d for d, _ in extracted], self.cfg["chunk"])
            for (doc, h), emb in zip(extracted, embedded):
                rel = self.rel(doc.path)
                stats["updated" if rel in known else "added"] += 1
                if doc.needs_ocr:
                    stats["needs_ocr"].append(rel)
                self._store(emb, h)
            self.db.commit()

        # Image references come from every indexed note, so rebuild them from stored docs' links.
        img_stats = self._update_images(embedder, imgs) if embedder.supports_images else "unavailable"
        self.db.commit()
        self._cache = None
        stats.update(ok=True, probe=probe, images=img_stats, notes_total=self.count(),
                     seconds=round(time.time() - t0, 1))
        return stats

    def _delete_note(self, rel: str) -> None:
        self.db.execute("DELETE FROM notes WHERE path=?", (rel,))
        self.db.execute("DELETE FROM chunks WHERE note_path=?", (rel,))

    def _store(self, emb: Embedded, h: str) -> None:
        d = emb.doc
        rel = self.rel(d.path)
        self._delete_note(rel)
        folder = Path(rel).parent.as_posix()
        self.db.execute(
            "INSERT INTO notes VALUES (?,?,?,?,?,?,?,?,?,?)",
            (rel, h, text_hash(d.text), d.path.stat().st_mtime, d.title, "" if folder == "." else folder,
             d.kind, json.dumps(d.tags, ensure_ascii=False), _blob(emb.vec), time.time()),
        )
        self.db.executemany(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
            [(rel, c.idx, c.heading_path, c.text, c.tokens, _blob(v)) for c, v in zip(emb.chunks, emb.chunk_vecs)],
        )
        refs = [self.rel(i.path) for i in d.images if i.path is not None and self._inside_vault(i.path)]
        self.db.execute("DELETE FROM meta WHERE key=?", (f"refs:{rel}",))
        if refs:
            self.set_meta(f"refs:{rel}", json.dumps(refs, ensure_ascii=False))

    def _inside_vault(self, p: Path) -> bool:
        try:
            p.resolve().relative_to(self.vault)
            return True
        except ValueError:
            return False

    def _update_images(self, embedder: Embedder, imgs: list[Path]) -> dict:
        known = {r[0]: (r[1], r[2]) for r in self.db.execute("SELECT path, mtime, hash FROM images")}
        referenced: dict[str, list[str]] = {}
        for key, val in self.db.execute("SELECT key, value FROM meta WHERE key LIKE 'refs:%'"):
            for img in json.loads(val):
                referenced.setdefault(img, []).append(key[5:])
        seen, todo = set(), []
        for p in imgs:
            rel = self.rel(p)
            seen.add(rel)
            mtime = p.stat().st_mtime
            if rel in known and known[rel][0] == mtime:
                continue
            todo.append((p, rel, mtime, file_hash(p)))
        for rel in set(known) - seen:
            self.db.execute("DELETE FROM images WHERE path=?", (rel,))
        failed = []
        for s in range(0, len(todo), 32):
            batch = todo[s : s + 32]
            try:
                vecs = embedder.embed_images([p for p, *_ in batch])
            except Exception:
                vecs = []
                for p, *_ in batch:  # fall back to one-by-one to isolate broken images
                    try:
                        vecs.append(embedder.embed_images([p])[0])
                    except Exception as e:
                        vecs.append(None)
                        failed.append({"path": self.rel(p), "error": f"{type(e).__name__}: {e}"})
            for (p, rel, mtime, h), v in zip(batch, vecs):
                if v is None:
                    continue
                self.db.execute("INSERT OR REPLACE INTO images VALUES (?,?,?,?,?)",
                                (rel, h, mtime, _blob(v), "[]"))
        for rel in seen:
            self.db.execute("UPDATE images SET referenced_by_json=? WHERE path=?",
                            (json.dumps(referenced.get(rel, []), ensure_ascii=False), rel))
        return {"embedded": len(todo) - len(failed), "total": len(seen), "failed": failed}

    # ---------- read access for analysis ----------

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]

    def load(self) -> dict:
        """Load all vectors into memory (cached until the next update)."""
        if self._cache is not None:
            return self._cache
        notes = self.db.execute(
            "SELECT path, title, folder, kind, tags_json, hash, text_hash, vec FROM notes ORDER BY path").fetchall()
        chunks = self.db.execute(
            "SELECT note_path, idx, heading_path, text, tokens, vec FROM chunks ORDER BY note_path, idx").fetchall()
        images = self.db.execute("SELECT path, hash, referenced_by_json, vec FROM images ORDER BY path").fetchall()

        def mat(rows, col):
            return np.vstack([_vec(r[col]) for r in rows]) if rows else np.zeros((0, DIM), np.float32)

        self._cache = {
            "notes": [
                {"path": r[0], "title": r[1], "folder": r[2], "kind": r[3], "tags": json.loads(r[4]),
                 "hash": r[5], "text_hash": r[6]} for r in notes
            ],
            "note_vecs": mat(notes, 7),
            "chunks": [
                {"note": r[0], "idx": r[1], "heading": r[2], "text": r[3], "tokens": r[4]} for r in chunks
            ],
            "chunk_vecs": mat(chunks, 5),
            "images": [{"path": r[0], "hash": r[1], "referenced_by": json.loads(r[2])} for r in images],
            "image_vecs": mat(images, 3),
        }
        return self._cache

    def close(self) -> None:
        self.db.commit()
        self.db.close()

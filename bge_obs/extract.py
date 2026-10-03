"""Document extraction and mechanical cleanup: Markdown (Obsidian), PDF, TXT, Skill folders."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import yaml

from .types import Extracted, ImageRef

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg"}

FM_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.S)
FENCE_RE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.S | re.M)
WIKI_EMBED_RE = re.compile(r"!\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
MD_IMG_RE = re.compile(r"!\[[^\]]*\]\(<?([^)>\s]+)>?(?:\s+\"[^\"]*\")?\)")
WIKI_RE = re.compile(r"\[\[([^\]|#]*)(?:#([^\]|]*))?(?:\|([^\]]*))?\]\]")
TAG_RE = re.compile(r"(?<![\w/#&])#([\w\-/]*[^\W\d][\w\-/]*)", re.U)
ZERO_WIDTH_RE = re.compile("[​‌‍﻿]")
PAGE_NUM_RE = re.compile(
    r"^\s*(?:[-–—]\s*)?(?:page\s*)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?(?:\s*[-–—])?\s*$|^\s*第\s*\d{1,4}\s*頁\s*$",
    re.I,
)
SENT_END = tuple("。！？.!?：:；;」』）)")
CJK_RE = re.compile(r"[぀-ヿ㐀-鿿豈-﫿가-힯]")
CJK_OR_PUNCT_RE = re.compile(r"[　-〿぀-ヿ㐀-鿿豈-﫿가-힯＀-￯]")


# ---------- shared helpers ----------

def _common_cleanup(text: str) -> str:
    text = ZERO_WIDTH_RE.sub("", text).replace(" ", " ").replace("　", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n" if text.strip() else ""


def _split_frontmatter(raw: str, removed: dict) -> tuple[dict, str]:
    m = FM_RE.match(raw.lstrip("﻿"))
    if not m:
        return {}, raw
    try:
        fm = yaml.safe_load(m.group(1)) or {}
        if not isinstance(fm, dict):
            fm = {}
    except yaml.YAMLError:
        fm = {}
        removed["bad_frontmatter"] = 1
    return fm, raw.lstrip("﻿")[m.end():]


def _protect_code(text: str) -> tuple[str, list[str]]:
    """Replace fenced code blocks with placeholders so Obsidian rewrites don't touch them."""
    blocks: list[str] = []

    def keep(m: re.Match) -> str:
        blocks.append(m.group(0))
        return f"\x00CODE{len(blocks) - 1}\x00"

    return FENCE_RE.sub(keep, text), blocks


def _restore_code(text: str, blocks: list[str]) -> str:
    return re.sub(r"\x00CODE(\d+)\x00", lambda m: blocks[int(m.group(1))], text)


def _fm_tags(fm: dict) -> list[str]:
    t = fm.get("tags") or fm.get("tag") or []
    if isinstance(t, str):
        t = re.split(r"[,\s]+", t)
    return [str(x).strip().lstrip("#") for x in t if str(x).strip().lstrip("#")]


def _inline_tags(text: str) -> list[str]:
    no_code = re.sub(r"`[^`\n]*`", " ", text)
    tags = []
    for line in no_code.split("\n"):
        if re.match(r"^\s{0,3}#{1,6}\s", line):  # heading line: only look after the heading marks
            line = re.sub(r"^\s{0,3}#{1,6}\s", "", line)
        tags.extend(TAG_RE.findall(line))
    return tags


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _resolve_image(name: str, base: Path, vault: Path | None) -> Path | None:
    name = name.strip()
    if re.match(r"^[a-z]+://", name, re.I):
        return None
    from urllib.parse import unquote

    name = unquote(name)
    cand = (base / name)
    if cand.is_file():
        return cand.resolve()
    if vault is not None:
        cand = vault / name
        if cand.is_file():
            return cand.resolve()
        hits = list(vault.rglob(Path(name).name))
        if hits:
            return hits[0].resolve()
    return None


def _normalise_obsidian(body: str, base: Path, vault: Path | None) -> tuple[str, list[ImageRef]]:
    text, code = _protect_code(body)
    text = re.sub(r"%%.*?%%", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    images: list[ImageRef] = []

    def wiki_embed(m: re.Match) -> str:
        target = m.group(1).strip()
        if Path(target).suffix.lower() in IMAGE_EXT:
            images.append(ImageRef(name=target, path=_resolve_image(target, base, vault)))
            return ""
        return Path(target).stem if Path(target).suffix.lower() == ".md" else target

    def md_img(m: re.Match) -> str:
        target = m.group(1)
        images.append(ImageRef(name=target, path=_resolve_image(target, base, vault)))
        return ""

    def wiki(m: re.Match) -> str:
        note, heading, alias = m.group(1), m.group(2), m.group(3)
        if alias:
            return alias
        return note.strip() or (heading or "").strip()

    text = WIKI_EMBED_RE.sub(wiki_embed, text)
    text = MD_IMG_RE.sub(md_img, text)
    text = WIKI_RE.sub(wiki, text)
    return _restore_code(text, code), images


def _first_h1(text: str) -> str | None:
    t, _ = _protect_code(text)
    m = re.search(r"^#\s+(.+?)\s*#*\s*$", t, re.M)
    return m.group(1).strip() if m else None


# ---------- per-kind extractors ----------

def _extract_md(path: Path, vault: Path | None, kind: str = "md") -> Extracted:
    raw = path.read_text(encoding="utf-8", errors="replace")
    removed: dict[str, int] = {}
    fm, body = _split_frontmatter(raw, removed)
    text, images = _normalise_obsidian(body, path.parent, vault)
    tags = _dedupe(_fm_tags(fm) + _inline_tags(_protect_code(body)[0]))
    text = _common_cleanup(text)
    title = str(fm.get("title") or _first_h1(text) or path.stem)
    return Extracted(path=path, kind=kind, title=title, text=text, frontmatter=fm, tags=tags,
                     images=images, removed=removed, meta={"raw_chars": len(raw)})


def _extract_skill(path: Path, vault: Path | None) -> Extracted:
    folder = path if path.is_dir() else path.parent
    skill_md = folder / "SKILL.md"
    doc = _extract_md(skill_md, vault, kind="skill")
    doc.path = path
    doc.title = str(doc.frontmatter.get("name") or folder.name)
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*")
                   if p.is_file() and p != skill_md)
    doc.meta["files"] = files[:200]
    return doc


def _extract_txt(path: Path) -> Extracted:
    from charset_normalizer import from_bytes

    data = path.read_bytes()
    try:
        raw = data.decode("utf-8")
    except UnicodeDecodeError:
        best = from_bytes(data).best()
        raw = str(best) if best is not None else data.decode("utf-8", errors="replace")
    return Extracted(path=path, kind="txt", title=path.stem, text=_common_cleanup(raw),
                     meta={"raw_chars": len(raw)})


def _norm_line(s: str) -> str:
    return re.sub(r"\d+", "#", s.strip())


def _clean_pdf_pages(pages: list[str], removed: dict, headers: Counter | None = None) -> str:
    split = [[ln.rstrip() for ln in p.split("\n")] for p in pages]
    # Header/footer: the same normalised line at the same edge position (top 2 / bottom 2)
    # on >=50% of pages (and at least 3 pages).
    n = len(split)
    if n >= 3:
        counts: Counter = Counter()
        for lines in split:
            ne = [ln for ln in lines if ln.strip()]
            keys = {(f"t{k}", _norm_line(ln)) for k, ln in enumerate(ne[:2])}
            keys |= {(f"b{k}", _norm_line(ln)) for k, ln in enumerate(reversed(ne[-2:]))}
            counts.update(keys)
        repeated = {k for k, c in counts.items() if k[1] and c >= max(3, n * 0.5)}
        if repeated:
            for lines in split:
                ne_idx = [i for i, ln in enumerate(lines) if ln.strip()]
                edge = [(f"t{k}", i) for k, i in enumerate(ne_idx[:2])]
                edge += [(f"b{k}", i) for k, i in enumerate(reversed(ne_idx[-2:]))]
                for pos, i in edge:
                    if lines[i] is not None and (pos, _norm_line(lines[i])) in repeated:
                        if headers is not None and pos.startswith("t") and not re.search(r"\d", lines[i]):
                            headers[lines[i].strip()] += 1
                        lines[i] = None
                        removed["header_footer"] = removed.get("header_footer", 0) + 1
    out_pages = []
    for lines in split:
        kept = []
        for ln in lines:
            if ln is None:
                continue
            if PAGE_NUM_RE.match(ln):
                removed["page_numbers"] = removed.get("page_numbers", 0) + 1
                continue
            kept.append(ln)
        out_pages.append(_join_lines(kept, removed))
    return "\n\n".join(p for p in out_pages if p.strip())


def _join_lines(lines: list[str], removed: dict) -> str:
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            if out and out[-1] != "":
                out.append("")
            continue
        if out and out[-1]:
            prev = out[-1]
            if re.search(r"[A-Za-z]-$", prev) and re.match(r"[a-z]", s):
                out[-1] = prev[:-1] + s
                removed["hyphen_joins"] = removed.get("hyphen_joins", 0) + 1
                continue
            if not prev.endswith(SENT_END) and (re.match(r"[a-z]", s) or CJK_RE.match(s)) \
                    and not re.match(r"^\s*([-*•●▪]|\d+[.)、])\s", s):
                sep = "" if CJK_OR_PUNCT_RE.match(prev[-1]) and CJK_OR_PUNCT_RE.match(s) else " "
                out[-1] = prev + sep + s
                removed["line_joins"] = removed.get("line_joins", 0) + 1
                continue
        out.append(s)
    return "\n".join(out).strip()


def _pdf_title(doc, text: str, stem: str, header: str | None = None) -> str:
    t = (doc.metadata or {}).get("title", "") or ""
    t = t.strip()
    if t and not re.match(r"^(untitled|microsoft (word|powerpoint)\b)", t, re.I) and not t.lower().endswith((".doc", ".docx", ".pdf")):
        return t
    if header and len(header) < 80:  # a running header is usually the document title
        return header
    for ln in text.split("\n"):
        if ln.strip():
            return ln.strip() if len(ln.strip()) < 80 else stem
    return stem


def _extract_pdf(path: Path) -> Extracted:
    import pymupdf

    removed: dict[str, int] = {}
    images: list[ImageRef] = []
    with pymupdf.open(path) as doc:
        pages = [p.get_text("text") for p in doc]
        raw_chars = sum(len(p) for p in pages)
        seen: set[int] = set()
        for pno, page in enumerate(doc, start=1):
            for info in page.get_images(full=True):
                xref = info[0]
                if xref in seen:
                    continue
                seen.add(xref)
                try:
                    img = doc.extract_image(xref)
                except Exception:
                    continue
                if min(img.get("width", 0), img.get("height", 0)) < 128:
                    continue
                images.append(ImageRef(name=f"p{pno}-{xref}.{img['ext']}", data=img["image"], page=pno))
        headers: Counter = Counter()
        text = _common_cleanup(_clean_pdf_pages(pages, removed, headers))
        n = len(pages)
        title = _pdf_title(doc, text, path.stem, headers.most_common(1)[0][0] if headers else None)
    needs_ocr = n > 0 and len("".join(pages).strip()) < 50 * n
    return Extracted(path=path, kind="pdf", title=title, text=text, images=images, removed=removed,
                     needs_ocr=needs_ocr, meta={"pages": n, "raw_chars": raw_chars})


def extract(path: Path, vault: Path | None = None) -> Extracted:
    path = Path(path)
    vault = Path(vault).resolve() if vault else None
    if path.is_dir():
        if (path / "SKILL.md").is_file():
            return _extract_skill(path, vault)
        raise ValueError(f"directory without SKILL.md: {path}")
    if path.name == "SKILL.md":
        return _extract_skill(path, vault)
    ext = path.suffix.lower()
    if ext == ".md":
        return _extract_md(path, vault)
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".txt":
        return _extract_txt(path)
    raise ValueError(f"unsupported file type: {path.suffix}")

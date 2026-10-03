"""Heading-aware chunking with a tokenizer-free token estimate."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .types import Chunk

CJK_CHAR = r"぀-ヿ㐀-鿿豈-﫿가-힯"
TOKEN_RE = re.compile(rf"[{CJK_CHAR}]|[A-Za-z0-9_]+|[^\s{CJK_CHAR}A-Za-z0-9_]")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def estimate_tokens(text: str) -> int:
    n = 0.0
    for t in TOKEN_RE.findall(text):
        if len(t) == 1 and re.match(rf"[{CJK_CHAR}]", t):
            n += 1
        elif t[0].isalnum() or t[0] == "_":
            n += 1.3 * max(1, len(t) / 6)  # long words split into several sub-word tokens
        else:
            n += 0.5
    return math.ceil(n)


@dataclass
class _Block:
    text: str
    start: int
    end: int
    atomic: bool = False  # fenced code: never split if it fits hard_max


@dataclass
class _Section:
    path: list[str]
    heading: str  # heading line ("" for preamble)
    blocks: list[_Block]


def _sections(text: str) -> list[_Section]:
    """Split text into heading sections; each section's blocks are paragraphs or code fences."""
    sections: list[_Section] = [_Section([], "", [])]
    stack: list[tuple[int, str]] = []
    pos, in_fence, para, para_start, fence_buf, fence_start = 0, None, [], 0, [], 0

    def flush_para(end: int) -> None:
        nonlocal para
        if para and "".join(para).strip():
            sections[-1].blocks.append(_Block("".join(para).strip("\n"), para_start, end))
        para = []

    for line in text.splitlines(keepends=True):
        start = pos
        pos += len(line)
        if in_fence:
            fence_buf.append(line)
            if line.strip().startswith(in_fence):
                sections[-1].blocks.append(_Block("".join(fence_buf).strip("\n"), fence_start, pos, True))
                in_fence, fence_buf = None, []
            continue
        m_f = FENCE_RE.match(line)
        if m_f:
            flush_para(start)
            in_fence, fence_buf, fence_start = m_f.group(1), [line], start
            continue
        m_h = HEADING_RE.match(line.rstrip("\n"))
        if m_h:
            flush_para(start)
            level, title = len(m_h.group(1)), m_h.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sections.append(_Section([t for _, t in stack], line.strip(), []))
            continue
        if not line.strip():
            flush_para(start)
            continue
        if not para:
            para_start = start
        para.append(line)
    if in_fence:  # unterminated fence: keep as atomic block
        sections[-1].blocks.append(_Block("".join(fence_buf).strip("\n"), fence_start, pos, True))
    flush_para(pos)
    return [s for s in sections if s.blocks or s.heading]


def _split_long(block: _Block, max_tokens: int, hard_max: int) -> list[_Block]:
    tok = estimate_tokens(block.text)
    if tok <= max_tokens or (block.atomic and tok <= hard_max):
        return [block]
    pieces, cur, cur_start, off = [], "", block.start, block.start
    for sent in re.split(r"(?<=[。！？!?；;])|(?<=\.)(?=\s)|(?<=\n)", block.text):
        if not sent:
            continue
        if cur and estimate_tokens(cur + sent) > max_tokens:
            pieces.append(_Block(cur, cur_start, off))
            cur, cur_start = "", off
        cur += sent
        off += len(sent)
    if cur:
        pieces.append(_Block(cur, cur_start, off))
    out = []
    for p in pieces:  # still too long (no sentence boundaries): hard split by characters
        if estimate_tokens(p.text) <= hard_max:
            out.append(p)
            continue
        step = max(1, int(len(p.text) * max_tokens / estimate_tokens(p.text)))
        for i in range(0, len(p.text), step):
            out.append(_Block(p.text[i : i + step], p.start + i, p.start + min(i + step, len(p.text))))
    return out


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def chunk_text(text: str, min_tokens: int = 300, max_tokens: int = 800, hard_max: int = 1024) -> list[Chunk]:
    if not text.strip():
        return []
    raw: list[dict] = []  # {path, text, start, end, tokens}
    for sec in _sections(text):
        head = sec.heading
        head_tok = estimate_tokens(head) if head else 0
        blocks = [p for b in sec.blocks for p in _split_long(b, max_tokens - head_tok, hard_max)]
        if not blocks:
            continue
        cur: list[_Block] = []

        def emit() -> None:
            body = "\n\n".join(b.text for b in cur)
            t = f"{head}\n\n{body}" if head else body
            raw.append({"path": sec.path, "text": t, "start": cur[0].start, "end": cur[-1].end,
                        "tokens": estimate_tokens(t)})

        for b in blocks:
            if cur and estimate_tokens("\n\n".join(x.text for x in cur + [b])) + head_tok > max_tokens:
                emit()
                cur = []
            cur.append(b)
        if cur:
            emit()

    # Merge small chunks into the previous one (same section or a sibling under a common parent).
    merged: list[dict] = []
    for c in raw:
        if merged and (c["tokens"] < min_tokens or merged[-1]["tokens"] < min_tokens):
            prev = merged[-1]
            joined = prev["text"] + "\n\n" + c["text"]
            if estimate_tokens(joined) <= max_tokens:
                prev.update(text=joined, end=max(prev["end"], c["end"]), start=min(prev["start"], c["start"]),
                            path=_common_prefix(prev["path"], c["path"]), tokens=estimate_tokens(joined))
                continue
        merged.append(dict(c))
    return [Chunk(idx=i, heading_path=" > ".join(c["path"]), text=c["text"], start=c["start"],
                  end=c["end"], tokens=c["tokens"]) for i, c in enumerate(merged)]

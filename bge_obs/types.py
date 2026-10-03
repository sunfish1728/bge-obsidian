"""Data types shared between extract / chunk / index / analyze."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ImageRef:
    """An image belonging to a document: a file on disk, or bytes embedded in a PDF."""

    name: str
    path: Path | None = None
    data: bytes | None = None
    page: int | None = None


@dataclass
class Extracted:
    path: Path
    kind: str  # "md" | "pdf" | "txt" | "skill"
    title: str
    text: str  # cleaned Markdown body (no frontmatter)
    frontmatter: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)  # frontmatter tags + inline #tags, without '#'
    images: list[ImageRef] = field(default_factory=list)
    removed: dict[str, int] = field(default_factory=dict)  # cleanup stats, e.g. {"header_footer": 30}
    needs_ocr: bool = False
    meta: dict[str, Any] = field(default_factory=dict)  # e.g. {"pages": 15, "raw_chars": 41230}


@dataclass
class Chunk:
    idx: int
    heading_path: str  # e.g. "第二章 > 2.1 安裝"; "" when before any heading
    text: str
    start: int  # char offsets into Extracted.text
    end: int
    tokens: int  # estimated token count

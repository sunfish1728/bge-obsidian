from __future__ import annotations

import io

import pymupdf
from PIL import Image

from bge_obs.chunk import chunk_text, estimate_tokens
from bge_obs.extract import extract

from .conftest import write


def _png(size: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (0, 128, 255)).save(buf, "PNG")
    return buf.getvalue()


def test_markdown_obsidian(tmp_path):
    (tmp_path / "att").mkdir()
    (tmp_path / "att/img.png").write_bytes(_png(10))
    p = write(tmp_path / "n.md", "---\ntags: [a, '#b']\n---\n# 標題\n見 [[其他|別名]]、[[X#h]] 與 ![[other note]] #學習/AI #123\n"
                                 "![[img.png|300]] ![x](att/img.png)\n%%hidden%%\n```\n#notatag [[keep]]\n```\n")
    d = extract(p, vault=tmp_path)
    assert d.title == "標題"
    assert d.tags == ["a", "b", "學習/AI"]
    assert "別名" in d.text and "X" in d.text and "other note" in d.text
    assert "hidden" not in d.text and "[[keep]]" in d.text
    assert [i.path is not None for i in d.images] == [True, True]


def test_txt_big5(tmp_path):
    p = tmp_path / "b.txt"
    p.write_bytes("繁體中文的文字檔案，使用大五碼編碼儲存。這是第二句話。".encode("big5"))
    assert "大五碼" in extract(p).text


def test_skill_dir(tmp_path):
    write(tmp_path / "my-skill/SKILL.md", "---\nname: my-skill\ndescription: does things\n---\n# Usage\nRun it.")
    write(tmp_path / "my-skill/scripts/run.py", "print(1)")
    d = extract(tmp_path / "my-skill")
    assert d.kind == "skill" and d.title == "my-skill" and d.meta["files"] == ["scripts/run.py"]


def test_pdf_cleanup(tmp_path):
    doc = pymupdf.open()
    words = ["alpha", "beta", "gamma", "delta"]
    for i, w in enumerate(words, start=1):
        pg = doc.new_page()
        pg.insert_text((50, 40), "My Report 2026")
        pg.insert_text((50, 100), f"Section {w} intro is a para-")
        pg.insert_text((50, 115), "graph that wraps across")
        pg.insert_text((50, 130), f"several lines about {w}.")
        pg.insert_text((50, 160), f"Closing remark on {w}.")
        pg.insert_text((300, 800), f"- {i} -")
        if i == 2:
            pg.insert_image(pymupdf.Rect(50, 200, 250, 400), stream=_png(200))
            pg.insert_image(pymupdf.Rect(50, 450, 60, 460), stream=_png(20))  # too small, skipped
    doc.save(tmp_path / "r.pdf")
    d = extract(tmp_path / "r.pdf")
    assert "My Report" not in d.text
    assert "paragraph that wraps across several lines about beta." in d.text
    assert d.removed["header_footer"] == 8 and d.removed["hyphen_joins"] == 4
    assert len(d.images) == 1 and d.images[0].page == 2 and d.images[0].data
    assert d.meta["pages"] == 4 and not d.needs_ocr


def test_pdf_needs_ocr(tmp_path):
    doc = pymupdf.open()
    for _ in range(2):
        doc.new_page().insert_image(pymupdf.Rect(0, 0, 300, 300), stream=_png(300))
    doc.save(tmp_path / "scan.pdf")
    assert extract(tmp_path / "scan.pdf").needs_ocr


def test_chunk_headings_and_limits():
    text = ("# 標題\n\n前言。\n\n## 第一節\n\n" + "這是一段很長的中文內容。" * 120 +
            "\n\n## 第二節\n\n短段落。\n\n```python\nprint(1)\n```\n\n### 2.1 子節\n\nEnglish words here.")
    chunks = chunk_text(text, min_tokens=50, max_tokens=300, hard_max=400)
    assert [c.idx for c in chunks] == list(range(len(chunks)))
    assert all(c.tokens <= 300 for c in chunks)
    assert any(c.heading_path == "標題 > 第一節" for c in chunks)
    assert all(0 <= c.start < c.end <= len(text) for c in chunks)
    assert "print(1)" in "".join(c.text for c in chunks)


def test_chunk_hard_split_and_empty():
    assert chunk_text("   \n") == []
    chunks = chunk_text("字" * 3000, max_tokens=500, hard_max=600)
    assert len(chunks) >= 5 and all(c.tokens <= 600 for c in chunks)
    assert estimate_tokens("Hello world 你好") == 5

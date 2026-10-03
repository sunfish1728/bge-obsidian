from __future__ import annotations

import io
from pathlib import Path

import pytest

from bge_obs import config as config_mod
from bge_obs.config import load_config
from bge_obs.service import Service

ML = """監督式學習使用標註資料訓練模型，常見任務包含分類與迴歸。模型從輸入和標籤的配對中學習映射函數，
訓練時以損失函數衡量預測誤差，再用梯度下降更新參數。驗證集用來調整超參數並避免過擬合。"""
PY = """Python 的 list comprehension 可以用一行建立列表，例如 [x*x for x in range(10)]。
字典推導式與集合推導式語法類似。生成器表達式則以小括號包住，能節省記憶體。"""
COOK = """番茄炒蛋的做法：先把蛋打散加少許鹽，熱鍋下油炒到半熟後盛起。再炒番茄出汁，
加入糖與鹽調味，最後倒回蛋拌炒均勻即可起鍋。"""


def write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    for i in range(3):
        write(v / "學習/機器學習" / f"ml{i}.md", f"---\ntags: [ml]\n---\n# ML {i}\n\n{ML}\n\n補充 {i}：" + "特徵工程很重要。" * (i + 1))
        write(v / "程式/Python" / f"py{i}.md", f"# Py {i}\n\n{PY}\n\n#python 範例 {i}：" + "使用 enumerate。" * (i + 1))
        write(v / "生活/料理" / f"cook{i}.md", f"# 料理 {i}\n\n{COOK}\n\n心得 {i}：" + "火候要控制好。" * (i + 1))
    write(v / ".obsidian/app.json", "{}")
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(buf, "PNG")
    (v / "attachments").mkdir()
    (v / "attachments/red.png").write_bytes(buf.getvalue())
    write(v / "學習/機器學習/with_img.md", f"# 圖片筆記\n\n![[red.png]]\n\n{ML}")
    return v


@pytest.fixture
def cfg(vault: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config_mod, "STATE_ROOT", tmp_path / "state")
    return load_config(path=None, overrides={"vault": str(vault), "backend": "fake",
                                             "chunk": {"min_tokens": 20, "max_tokens": 200, "hard_max": 300}})


@pytest.fixture
def svc(cfg):
    s = Service(cfg)
    yield s
    s.close()

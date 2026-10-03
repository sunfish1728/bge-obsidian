# bge-obs

給 Claude Code / Codex 用的 Obsidian 前處理工具：用 BAAI Visualized_m3（bge-m3 + 圖片）做去重、冗餘偵測、資料夾分類、標籤建議、相似筆記。工具只輸出 JSON，修改由 Agent 透過 Skill 執行。設計細節見 [PLAN.md](PLAN.md)。

所有檔案都留在本資料夾內：虛擬環境 `.venv/`、模型 `models/`、索引與服務狀態 `state/`、設定 `config.yaml`。

## 安裝

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -e ".[local,dev]"
git clone https://github.com/FlagOpen/FlagEmbedding.git vendor/FlagEmbedding
.venv/Scripts/python scripts/download_model.py
```

複製 `config.example.yaml` 為 `config.yaml`，填入 `vault` 路徑。

## 使用

```bash
.venv/Scripts/bge-obs index                 # 第一次建立索引
.venv/Scripts/bge-obs ingest "_inbox/新資料.pdf"
.venv/Scripts/python scripts/install_skill.py   # 安裝 Skill 到 vault 的 .claude/ 與 .agents/
```

Skill 安裝後，在 vault 中對 Claude Code / Codex 說「把 _inbox 的資料整理進 vault」即可。

## 測試

```bash
.venv/Scripts/python -m pytest tests -q
```

測試使用 `backend: fake`（不需要 GPU 或模型）。

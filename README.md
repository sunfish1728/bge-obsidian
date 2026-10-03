# bge-obs

給 Claude Code / Codex 用的 Obsidian 前處理工具：用 BAAI Visualized_m3（bge-m3 + 圖片）做去重、冗餘偵測、資料夾分類、標籤建議、相似筆記。工具只輸出 JSON，修改由 Agent 透過 Skill 執行。設計細節見 [PLAN.md](PLAN.md)。

所有檔案都留在本資料夾內：虛擬環境 `.venv/`、模型 `models/`、索引與服務狀態 `state/`、設定 `config.yaml`。

## 一鍵安裝（拉取 + 安裝 + 安裝 Skill）

需要 git 與 Python 3.10 以上。會在目前資料夾下建立 `bge-obsidian/`，再次執行即更新。

Windows（PowerShell）：

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/sunfish1728/bge-obsidian/main/install.ps1))) -Vault "D:\MyVault" -Index
```

macOS / Linux：

```bash
curl -fsSL https://raw.githubusercontent.com/sunfish1728/bge-obsidian/main/install.sh | bash -s -- --vault ~/MyVault --index
```

選項：`-Dir` / `--dir` 安裝位置、`-Cpu` / `--cpu` 或 `-Cuda` / `--cuda` 指定 torch 版本（預設依 `nvidia-smi` 自動判斷）、`-SkipModels` / `--skip-models`、`-Index` / `--index` 安裝後立即建立索引、`-DryRun` / `--dry-run` 只列出要執行的指令。

腳本會：clone 或 `git pull` 本倉庫 → 建立專案內 `.venv` 並安裝對應的 torch → 下載 FlagEmbedding 與模型 → 把 vault 寫入 `config.yaml` → 將 Skill 安裝到 vault 的 `.claude/skills/` 與 `.agents/skills/`。所有快取都留在安裝資料夾內。

## 手動安裝

```bash
python scripts/setup_env.py --vault "D:/MyVault"
# 模型已經備妥時：
python scripts/setup_env.py --skip-models
# 只看預計執行的指令，不修改檔案：
python scripts/setup_env.py --dry-run --cpu --skip-models
```

需要 Python 3.10 以上。腳本會建立專案內的 `.venv`，以 `nvidia-smi` 判斷使用 CPU 或 CUDA；也可以指定 `--cpu` 或 `--cuda --cuda-index cu128`。已安裝且種類相符的 torch 會保留。`--recreate` 會直接刪除並重建 `.venv`，請只在確定要重建時使用。pip 快取在 `.pip-cache/`，Hugging Face 快取在 `models/.hf-cache/`，下載時跳過已存在的模型檔案。

腳本會在缺少 `config.yaml` 時從範例複製，接著填入 `vault` 路徑。Windows 使用 `.venv/Scripts/`，macOS/Linux 使用 `.venv/bin/`。

本機後端預設 `local.device: auto`、`local.fp16: auto`：CUDA 可用就用 CUDA 與 fp16，否則用 CPU 與 fp32。設定 `device: cpu` 可強制使用 CPU，CPU 上的 `fp16: true` 也會改用 fp32。`local.threads: null` 使用 torch 預設 CPU 執行緒數，也可填入正整數。`status` 的 `embedder` 欄位會顯示實際使用的 `device` 和 `precision`。

CPU 載入與運算較慢；`server.start_timeout` 預設等候 300 秒，可在 `config.yaml` 調整。既有設定檔不會自動改寫，若原本指定 `device: cuda`，需要自行改為 `auto` 或 `cpu` 才能在沒有 CUDA 的電腦上使用。

## 使用

```bash
.venv/Scripts/bge-obs index                 # 第一次建立索引
.venv/Scripts/bge-obs ingest "_inbox/新資料.pdf"
.venv/Scripts/python scripts/install_skill.py   # 安裝 Skill 到 vault 的 .claude/ 與 .agents/
```

Skill 安裝後，在 vault 中對 Claude Code / Codex 說「把 _inbox 的資料整理進 vault」即可。

## 測試

```bash
.venv/Scripts/python -m pytest tests -q -p no:cacheprovider
# 使用既有模型測試 CPU 文字與圖片向量（離線）：
.venv/Scripts/python scripts/smoke_cpu.py
```

測試使用 `backend: fake`（不需要 GPU 或模型）。

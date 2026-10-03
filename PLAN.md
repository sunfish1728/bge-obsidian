# BGE-Obsidian 完整方案

> 給 Claude Code / Codex 呼叫的 Obsidian 資料前處理工具。
> 以 Skill 驅動：工具負責「量」（嵌入、相似度、比對），Agent 負責「判斷與動手」（改寫、移動、寫入 frontmatter）。

---

## 1. 定位與原則

| 原則 | 說明 |
|---|---|
| **工具唯讀** | CLI 不修改任何筆記，只輸出 JSON。所有寫入由 Agent 透過自己的檔案工具執行。 |
| **不刪除** | 重複、疑似重複一律標記並移到 `_review/`，由使用者決定。 |
| **跨 Agent** | 同一份 `SKILL.md` 同時供 Claude Code 與 Codex 使用。 |
| **本機優先，可切 API** | 預設本機 `Visualized_m3`；可切換 OpenAI 相容的 embeddings API。 |
| **規模** | 目標 300–1000 篇筆記，不引入向量資料庫，SQLite + numpy 暴力搜尋即可。 |

---

## 2. 架構

```
                ┌──────────── Claude Code / Codex ────────────┐
                │  SKILL.md：何時呼叫、如何解讀、如何動手       │
                └───────────────┬─────────────────────────────┘
                                │ bge-obs <cmd> --json
                ┌───────────────▼─────────────────────────────┐
                │ CLI (bge_obs/cli.py)                         │
                │  └─ 自動啟動 / 連線常駐服務 127.0.0.1         │
                └───────────────┬─────────────────────────────┘
                ┌───────────────▼─────────────────────────────┐
                │ 常駐服務 (server.py) — 模型只載入一次          │
                │  extract → chunk → embed → index → analyze   │
                └──────┬───────────────────────┬──────────────┘
                       │                       │
          ┌────────────▼──────────┐   ┌────────▼─────────────┐
          │ Embedder 後端          │   │ index.sqlite          │
          │  local: Visualized_m3 │   │  notes / chunks /     │
          │  api:   bge-m3 (text) │   │  images / meta        │
          └───────────────────────┘   └──────────────────────┘
```

---

## 3. 模型與後端

### 3.1 本機：BAAI Visualized_m3

- 權重：Hugging Face `BAAI/bge-visualized` 內的 `Visualized_m3.pth`，文字骨幹為 `BAAI/bge-m3`。
- 安裝（固定 commit，避免上游變動）：
  ```bash
  git clone https://github.com/FlagOpen/FlagEmbedding.git vendor/FlagEmbedding
  pip install -e vendor/FlagEmbedding/research/visual_bge
  pip install torchvision timm einops ftfy
  ```
- 能力：文字、圖片、圖文混合三種輸入，輸出 **1024 維** dense 向量。
- 顯存：bge-m3（約 568M 參數）FP16 加上 EVA-CLIP 視覺塔，估計約 2GB。本機 RTX 4060 Ti 8GB 足夠，**不需要量化**。
- 關鍵事實：VISTA 論文說明訓練時**文字編碼器完全凍結**，所以純文字向量與原版 bge-m3 dense 向量相同。這是能和 API 混用的前提（見 3.3）。

### 3.2 API：OpenAI 相容 embeddings

```yaml
backend: api
api:
  base_url: https://api.siliconflow.cn/v1   # 任一 OpenAI 相容端點
  model: BAAI/bge-m3
  api_key_env: BGE_OBS_API_KEY              # 金鑰只從環境變數讀取
  batch_size: 32
  max_rps: 5
```

- API 只處理**文字**。圖片嵌入一律走本機；若本機不可用，則跳過圖片功能並在 JSON 中回報 `"images": "unavailable"`。

### 3.3 向量一致性保護

- `meta` 表記錄 `model_family=bge-m3`、`dim=1024`，以及一組**探針句子**的向量。
- 每次啟動或切換後端時，重新嵌入探針並和已存向量比較 cosine：
  - ≥ 0.999：相容，可混用
  - 低於門檻：拒絕寫入，提示執行 `bge-obs index --rebuild`
- 實作第一步就要驗證：本機 Visualized_m3 與 API bge-m3 的探針 cosine 確實接近 1。

---

## 4. 資料處理流程

### 4.1 抽取 `extract.py`

| 類型 | 做法 |
|---|---|
| `.md` | 解析 frontmatter（保留原值），正文保留標題結構；收集 `![[img]]` 和 `![](img)` 引用的圖片 |
| `.pdf` | PyMuPDF 逐頁抽取文字與內嵌圖片（只保留邊長 ≥ 128px 的圖片）；掃描檔若沒有文字層則回報 `needs_ocr: true`，**不在本工具範圍內處理 OCR** |
| `.txt` | 直接讀取，自動偵測編碼（utf-8 / big5 / gbk） |
| Skill 資料夾 | 以 `SKILL.md` 為主體，其餘檔案名稱列為附屬清單 |

### 4.2 機械清理（不需要 LLM）

- PDF：移除在 ≥ 50% 頁面重複出現的頁首和頁尾、獨立的頁碼行；合併被硬斷行切開的句子；處理連字號斷字
- 通用：統一全形和半形空白，把連續 3 個以上的空行壓成 1 個，移除零寬字元
- 輸出 `clean_text` 和 `removed_stats`（移除了多少行、屬於哪一類），讓 Agent 知道清掉了什麼

### 4.3 切塊 `chunk.py`

- 依 Markdown 標題分段；段落過長時依句號切開；目標長度 300–800 tokens，上限 1024
- 每個 chunk 帶 `heading_path`（例如 `第二章 > 2.1 安裝`），方便 Agent 定位
- **筆記向量** = 所有 chunk 向量依長度加權平均後再做 L2 正規化，另外加入「標題 + 前 500 字」的向量，兩者取平均

---

## 5. 索引 `index.py`

SQLite 單一檔案（`.bge-obs/index.sqlite`，放在 vault 根目錄，並加進 `.gitignore`）：

```sql
notes  (path PK, hash, mtime, title, folder, tags_json, vec BLOB, indexed_at)
chunks (id PK, note_path, idx, heading_path, text, vec BLOB)
images (path PK, hash, vec BLOB, referenced_by_json)
meta   (key PK, value)          -- model_family, dim, probe_vec, schema_version
```

- 向量以 float32 BLOB 儲存，啟動時載入成 numpy 矩陣。1000 篇 × 約 20 chunks × 1024 維約 80MB，暴力搜尋只需幾毫秒。
- **增量更新**：mtime 有變再比對 hash，hash 不同才重新嵌入；刪除的檔案同步移出索引。
- 排除路徑寫在設定檔：`.obsidian/`、`.trash/`、`_review/`、`templates/`。

---

## 6. 分析功能與演算法

所有門檻都放在 `config.yaml`，第 8 階段會用真實 vault 校準。

### 6.1 去重 `dedup`

| 層級 | 判定 | 輸出 |
|---|---|---|
| 完全重複 | 檔案 hash 或 `clean_text` hash 相同 | `exact` |
| 筆記級 | 筆記向量 cos ≥ 0.95 | `duplicate` |
|  | 0.85 ≤ cos < 0.95 | `near_duplicate` |
| 包含關係 | A 的 chunk 中有 ≥ 80% 能在 B 找到 cos ≥ 0.92 的對應 | `contained_in`（A 是 B 的子集，例如摘錄或舊版） |
| 圖片 | 圖片向量 cos ≥ 0.97 | `image_duplicate` |

每組配對都附上**證據**：最相似的 3 對 chunk 原文片段，讓 Agent 不必讀完全文就能判斷。

### 6.2 冗餘段落 `redundant`（供 Agent 精簡）

- **文件內**：chunk 兩兩比較，cos ≥ 0.92 的配對視為內容重複
- **跨 vault**：chunk 和其他筆記的 chunk 比對，cos ≥ 0.92 時回報「這段在 `[[X]]` 已經有了」
- Agent 依結果刪除重複段落、改成連結，或合併改寫

### 6.3 分類 `classify`

- 每個資料夾的**中心向量** = 該資料夾內筆記向量的平均（筆記數少於 3 的資料夾，改用資料夾名稱加上 `config.folders` 裡的描述來嵌入）
- 回傳前 3 名和分數，以及第 1 名和第 2 名的差距 `margin`
- 最高分低於 0.45，或 `margin` 低於 0.03 時，標記 `confident: false`，由 Agent 詢問使用者

### 6.4 標籤 `tags`

標籤分數取兩個來源的最大值：

1. **描述向量**：`tags.yaml` 中每個標籤的說明句
2. **使用中心向量**：vault 內已經使用該標籤的筆記的平均向量（至少 3 篇才採用）

回傳分數 ≥ 0.50 的前 5 個；只建議 `tags.yaml` 裡已有的標籤，避免標籤數量失控。

```yaml
# tags.yaml
python:   Python 程式語言、套件、語法與實作範例
llm:      大型語言模型、提示工程、Agent、RAG
skill:    Claude Code 或 Codex 的 Skill 定義與使用方式
paper:    學術論文、研究方法與實驗結果
```

### 6.5 標籤建議 `suggest-tags`

- 對全部筆記向量做 agglomerative clustering（cosine 距離，群數自動決定，也可以用 `--k` 指定）
- 每群回傳：大小、離中心最近的 5 篇標題、群內最常出現的既有標籤
- **由 Agent 命名**，再和使用者確認後寫入 `tags.yaml`

### 6.6 相似 `similar`

- 輸入可以是筆記路徑、一段文字，或圖片路徑（只有本機後端支援圖片）
- 回傳 Top-K 筆記，並附上最相關的 chunk 片段

---

## 7. CLI 規格

所有指令都輸出 JSON 到 stdout，錯誤也用 JSON 格式（`{"ok": false, "error": {...}}`），exit code 不為 0。

| 指令 | 用途 |
|---|---|
| `bge-obs status` | 服務狀態、後端、索引筆數、一致性檢查結果 |
| `bge-obs index [--rebuild]` | 增量或完整重建索引 |
| `bge-obs ingest <file...>` | **主要入口**：抽取、清理、去重、分類、標籤、相似筆記，一次完成 |
| `bge-obs dedup [<file>\|--all]` | 去重報告 |
| `bge-obs redundant <file>` | 冗餘段落報告 |
| `bge-obs classify <file>` | 資料夾候選 |
| `bge-obs tags <file>` | 標籤候選 |
| `bge-obs suggest-tags [--k N]` | 分群後提出候選標籤 |
| `bge-obs similar <file\|--text "..."\|--image p> [--k 10]` | 相似筆記 |
| `bge-obs extract <file>` | 只輸出清理後的 Markdown（PDF 轉筆記時使用） |
| `bge-obs serve / stop` | 手動啟動或停止常駐服務 |

`ingest` 輸出範例：

```json
{
  "ok": true,
  "file": "inbox/attention.pdf",
  "extract": {"pages": 15, "chars": 41230, "images": 6, "needs_ocr": false,
              "removed": {"header_footer": 30, "page_numbers": 15, "hyphen_joins": 42}},
  "dedup": [
    {"kind": "near_duplicate", "target": "論文/Transformer 筆記.md", "score": 0.91,
     "evidence": [{"src": "3.2 Scaled Dot-Product...", "dst": "## 注意力計算 ...", "score": 0.95}]}
  ],
  "redundant": {"internal": 2, "cross_vault": 4, "details_cmd": "bge-obs redundant inbox/attention.pdf"},
  "classify": {"top": [{"folder": "論文/NLP", "score": 0.71}, {"folder": "學習/深度學習", "score": 0.64}],
               "margin": 0.07, "confident": true},
  "tags": [{"tag": "paper", "score": 0.68}, {"tag": "llm", "score": 0.61}],
  "similar": [{"path": "學習/深度學習/Self-Attention.md", "score": 0.83}]
}
```

---

## 8. 常駐服務 `server.py`

- 目的：Visualized_m3 載入需要 5–10 秒，Agent 一個任務可能連續呼叫多次，讓模型只載入一次
- CLI 被呼叫時，先讀取 `.bge-obs/server.json`（port、pid、token）並檢查服務是否存活；沒有存活就在背景啟動（Windows 使用 `DETACHED_PROCESS`），再等待 ready
- 只綁定 `127.0.0.1`，使用隨機 port，加上一次性 token 驗證
- 閒置 30 分鐘後自動結束，釋放顯存
- 用檔案鎖避免同時啟動兩個服務
- `--no-server` 參數可以在單一程序內直接執行（方便除錯和 CI）

---

## 9. Skill 設計

### 9.1 安裝位置

| Agent | 路徑 |
|---|---|
| Claude Code | `~/.claude/skills/bge-obsidian/` |
| Codex | `~/.agents/skills/bge-obsidian/` |

`scripts/install_skill.py` 會把 `skill/` 複製或連結到這兩個位置。

### 9.2 `SKILL.md` 大綱

```markdown
---
name: bge-obsidian
description: Obsidian vault 資料前處理。新資料（PDF、Skill、學習資料）要放進 vault、
  檢查是否重複、決定資料夾、加標籤、找相關筆記、精簡重複內容時使用。
  不用於一般問答或與 vault 無關的檔案。
---

## 核心規則
- 工具只讀；所有修改由你執行。永遠不刪除檔案。
- duplicate / contained_in → 加 frontmatter `duplicate_of`，移到 `_review/`，告知使用者。
- classify.confident == false → 詢問使用者，不要自行決定。
- 標籤只用 tags 指令回傳的值。
- 批次移動超過 10 個檔案前，先列出清單請使用者確認。

## 工作流程
### A. 新資料入庫
1. `bge-obs ingest <file>`
2. 依 dedup 結果決定：新增 / 合併到既有筆記 / 移到 _review
3. PDF：`bge-obs extract` 取得清理後的文字 → 你撰寫成筆記（摘要、重點）
4. 依 redundant 刪除或改寫已存在於 vault 的段落，並改成 [[連結]]
5. 寫入 frontmatter（tags、source、ingested_at），在「相關筆記」區塊加上 similar 前 3 名
6. 移動到 classify 第一名的資料夾

### B. 整理既有 vault
`bge-obs index` → `dedup --all` → 依結果逐組處理

### C. 建立標籤表
`suggest-tags` → 為每群命名 → 和使用者確認 → 寫入 tags.yaml

## 輸出欄位說明
（各指令的 JSON 欄位意義與門檻）
```

### 9.3 Frontmatter 慣例（由 Agent 寫入）

```yaml
---
tags: [paper, llm]
source: inbox/attention.pdf
ingested_at: 2026-10-03
duplicate_of: "[[Transformer 筆記]]"   # 只在有重複時出現
related: ["[[Self-Attention]]", "[[位置編碼]]"]
---
```

---

## 10. 設定檔 `config.yaml`

```yaml
vault: "D:/Obsidian/MyVault"      # 待使用者提供
inbox: "_inbox"
review_dir: "_review"
exclude: [".obsidian", ".trash", "_review", "templates"]

backend: local                     # local | api
local:
  text_model: BAAI/bge-m3
  visual_weight: models/Visualized_m3.pth
  device: cuda
  fp16: true
  batch_size: 16
api: { ... }                       # 見 3.2

chunk: { min_tokens: 300, max_tokens: 800, hard_max: 1024 }

thresholds:
  dup: 0.95
  near_dup: 0.85
  chunk_match: 0.92
  contained_ratio: 0.80
  image_dup: 0.97
  classify_min: 0.45
  classify_margin: 0.03
  tag_min: 0.50

folders:                           # 選填：筆記少的資料夾用描述補強
  "學習/深度學習": 神經網路、訓練方法、模型架構
server: { idle_minutes: 30 }
```

---

## 11. 專案結構

```
BGE Obsidian/
├─ PLAN.md
├─ pyproject.toml              # 進入點：bge-obs
├─ config.example.yaml
├─ tags.example.yaml
├─ bge_obs/
│   ├─ cli.py                  # argparse，輸出 JSON
│   ├─ client.py               # 連線、自動啟動服務
│   ├─ server.py               # 常駐服務（stdlib http.server，不額外引入框架）
│   ├─ config.py
│   ├─ embed/
│   │   ├─ base.py             # Embedder 介面：embed_text / embed_image
│   │   ├─ local_visual.py     # Visualized_m3
│   │   └─ api.py              # OpenAI 相容
│   ├─ extract.py
│   ├─ chunk.py
│   ├─ index.py
│   └─ analyze/
│       ├─ dedup.py
│       ├─ redundant.py
│       ├─ classify.py
│       ├─ tags.py
│       └─ similar.py
├─ skill/
│   └─ SKILL.md
├─ scripts/
│   ├─ install_skill.py
│   └─ download_model.py       # 下載 Visualized_m3.pth 與 bge-m3
├─ tests/                      # 用小型範例 vault 測試
└─ vendor/FlagEmbedding/       # visual_bge（固定 commit）
```

依賴套件：`torch`（已安裝，需確認是 CUDA 版本）、`visual_bge` 及其依賴、`pymupdf`、`numpy`、`scikit-learn`（分群用）、`pyyaml`、`httpx`、`charset-normalizer`。

---

## 12. 實作階段

| 階段 | 內容 | 驗收標準 |
|---|---|---|
| **P0 環境** | 確認 torch 是 CUDA 版、安裝 visual_bge、下載權重 | 本機能嵌入一句文字和一張圖片；顯存用量有實際記錄 |
| **P1 後端** | Embedder 介面、local 與 api 兩個實作、探針一致性檢查 | local 和 API 的探針向量 cos ≥ 0.999（若達不到，改成各後端使用獨立索引） |
| **P2 抽取與切塊** | md / pdf / txt / skill 抽取、清理、切塊 | 用 3 份真實 PDF 檢查，頁首和頁尾都有移除 |
| **P3 索引** | SQLite schema、增量更新、排除規則 | 第二次執行 `index` 時，未變更的檔案 0 次重新嵌入 |
| **P4 分析** | dedup / redundant / classify / tags / similar / suggest-tags | 在測試 vault 上，刻意放入的重複檔案全部被抓到 |
| **P5 CLI 與服務** | JSON 輸出、自動啟動、閒置關閉 | 第二次呼叫延遲 < 1 秒 |
| **P6 Skill** | SKILL.md、安裝腳本 | Claude Code 和 Codex 都能用自然語言觸發並完成「新資料入庫」流程 |
| **P7 校準** | 在真實 vault 上執行 `index`，檢視相似度分布，調整門檻 | 抽查 20 組去重和分類結果，由使用者確認準確度 |

---

## 13. 風險與待確認

| 項目 | 說明 / 對策 |
|---|---|
| visual_bge 在 Windows 的安裝 | 依賴 timm、EVA-CLIP，P0 先驗證；失敗時退回純文字 bge-m3（sentence-transformers），圖片功能暫停 |
| 本機與 API 向量相容性 | 理論上相同（文字編碼器凍結），但必須在 P1 實測 |
| 掃描版 PDF | 不在範圍內；回報 `needs_ocr`，由 Agent 另外處理 |
| 門檻是否適用 | 中文和混合語言的分數分布可能不同，P7 用真實資料校準 |
| **Vault 路徑** | **待使用者提供** |
| **初始標籤表** | **待使用者提供**，或 P7 之後用 `suggest-tags` 產生 |

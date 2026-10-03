---
name: bge-obsidian
description: Obsidian vault 資料前處理（嵌入模型 bge-m3）。當使用者要把新資料（PDF、Skill、學習資料、筆記、txt）放進 vault，或要檢查重複、決定資料夾、加標籤、找相關筆記、精簡和既有筆記重複的內容、整理 vault、建立標籤表時使用。不用於與 vault 無關的一般問答或程式工作。
---

# bge-obsidian

用嵌入向量分析 Obsidian vault。**工具只讀不寫**：它只輸出 JSON，所有修改（改寫、搬移、寫 frontmatter）都由你用自己的檔案工具完成。

## 呼叫方式

```
"{{BGE_OBS}}" <command> [args]
```

- 每個指令都輸出一個 JSON 物件；`"ok": false` 時讀 `error.code` / `error.message`。
- 第一次呼叫會在背景啟動常駐服務並載入模型（約 20 秒），之後每次約 0.5 秒。閒置 30 分鐘自動關閉。
- 路徑可以是絕對路徑，或相對於 vault 根目錄。
- 分析指令會先自動做增量索引；若回傳 `index_empty`，先執行 `index`。

| 指令 | 用途 |
|---|---|
| `status` | 索引筆數、後端、相容性 |
| `index [--rebuild]` | 建立／增量更新索引（第一次對 300–1000 篇約需數分鐘） |
| `ingest <file...> [--k 5]` | **新資料主要入口**：抽取＋去重＋冗餘＋分類＋標籤＋相似筆記 |
| `extract <file> [--max-chars N]` | 清理後的純文字（PDF 去頁首頁尾、接斷行），用來撰寫筆記 |
| `dedup <file>` / `dedup --all` | 單一檔案或整個 vault 的重複 |
| `redundant <file>` | 每段內容是否在文件內重複、或已存在於 vault 的哪篇筆記 |
| `classify <file>` | 資料夾候選 |
| `tags <file>` | 標籤候選 |
| `suggest-tags [--k N]` | 分群提出候選標籤表 |
| `similar <file>` / `--text "..."` / `--image <png>` | 相似筆記（圖片只限本機後端） |

## 核心規則

1. **永不刪除檔案。** 重複的檔案移到 `_review/`，並在 frontmatter 加 `duplicate_of`。
2. `classify.confident == false` 時，列出候選詢問使用者，不要自行決定。
3. 標籤只使用 `tags` 指令回傳的值；`tags_file_missing: true` 時告訴使用者可以用流程 C 建立標籤表。
4. 一次搬移或修改超過 10 個檔案前，先列出清單請使用者確認。
5. `needs_ocr: true` 的 PDF 沒有文字層，告知使用者，不要硬寫內容。
6. 分數只是參考：判斷前先看 `evidence` / `best_chunk` 的原文片段。

## 流程 A：新資料入庫

1. 執行 `ingest <file>`。
2. 依 `dedup` 決定（取第一筆，依 kind）：
   - `exact`：完全相同 → 不新增，告知使用者已存在於 `target`。
   - `duplicate`：內容幾乎相同 → 移到 `_review/`，加 `duplicate_of: "[[target]]"`，告知使用者。
   - `contained_in`：新資料是既有筆記的子集 → 同上，或只把新資料中多出的部分合併進 target。
   - `contains`：新資料涵蓋既有筆記且更完整 → 建議合併；舊筆記加 `duplicate_of` 移到 `_review/`，需使用者確認。
   - `near_duplicate`：主題相近但不同 → 正常入庫，並在相關筆記加互相連結。
3. PDF / txt：用 `extract` 取得清理後的文字，寫成 Markdown 筆記（保留結構與重點；需要時附原檔連結）。
4. 精簡：依 `redundant.top_cross_vault`（需要完整清單時執行 `redundant <file>`），把已存在於其他筆記的段落刪掉或縮成一句，並改成 `[[found_in]]` 連結；文件內重複的段落（`internal`）合併。
5. 寫入 frontmatter：
   ```yaml
   tags: [ ... ]            # tags 指令結果，加上原有標籤
   source: <原始檔名或網址>
   ingested_at: YYYY-MM-DD
   related: ["[[筆記A]]", "[[筆記B]]"]   # similar 前 3 名中分數 >= 0.65 者
   ```
6. 移到 `classify.top[0].folder`（confident 時）。
7. 回報：放到哪裡、加了哪些標籤、精簡了什麼、發現哪些重複。

## 流程 B：整理既有 vault

1. `index`，再執行 `dedup --all`。
2. `exact_groups`：每組保留一份（通常是路徑較合理或較舊的），其餘移到 `_review/`。
3. `pairs`：依 `kind` 與 `evidence` 處理（同流程 A 第 2 步），先列出計畫給使用者確認再動手。
4. `images`：重複圖片只回報，由使用者決定（圖片可能被多篇筆記引用，見 `referenced_by`）。

## 流程 C：建立標籤表

1. `suggest-tags` 取得分群（每群有 `representative` 代表標題、`common_tags`、`folders`）。
2. 為每群取一個簡短標籤名，並寫一句描述（描述會被嵌入，用來比對筆記，寫得具體一點）。
3. 給使用者確認後，寫入 `status` 回傳的 `tags_file` 路徑（YAML：`標籤名: 描述`）。

## 欄位速查

- 相似度都是 cosine（0–1）。bge-m3 不相關內容約 0.3–0.5，同主題約 0.6–0.8，重複 ≥ 0.9。
- `dedup[].coverage.doc_in_target`：新文件有多少比例的段落能在 target 找到。
- `redundant.redundant_ratio`：新文件中與既有內容重複的 token 比例。
- `classify.margin`：第一名和第二名的分數差，越小越模糊。
- `tags[].source`：`description`（依標籤描述）或 `usage`（依 vault 中已使用該標籤的筆記）。

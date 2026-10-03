# bge-obsidian 新手教學

這份教學假設你**沒有寫過程式**，照著做就能用。

## 這是什麼？

一個幫你整理 Obsidian 筆記的小幫手。裝好之後，你只要把新資料（PDF、筆記、文字檔）丟進一個資料夾，然後對 Claude Code 或 Codex 說「幫我整理」，它就會：

- 🔍 **檢查重複**：這份資料是不是已經存過了？
- ✂️ **精簡內容**：已經寫在其他筆記裡的段落，改成連結
- 📁 **自動分類**：放到最適合的資料夾
- 🏷️ **自動加標籤**
- 🔗 **連結相關筆記**

它**不會刪除你的任何檔案**。遇到重複的資料，會移到 `_review` 資料夾讓你自己決定。

---

## 第一部分：安裝（只需要做一次）

整個安裝約需 10 到 30 分鐘，大部分時間是在下載（約 5GB），下載期間不用守著。

### 步驟 1：安裝兩個必要工具

如果你已經裝過，可以跳過。

**① Python**（3.10 以上的版本）

1. 前往 https://www.python.org/downloads/
2. 點黃色的「Download Python」按鈕，下載後執行
3. ⚠️ **重要**：安裝畫面最下方有一個「**Add python.exe to PATH**」，**一定要打勾**
4. 點「Install Now」

**② Git**

1. 前往 https://git-scm.com/download/win
2. 下載後執行，**全部按「Next」使用預設值**即可

> 💡 macOS 使用者：打開「終端機」輸入 `xcode-select --install` 就會裝好 Git；Python 請從上面的網址下載 macOS 版。

### 步驟 2：找到你的 Obsidian 筆記庫（vault）路徑

1. 打開 Obsidian
2. 在左側的檔案清單中，對任一檔案按右鍵 →「**在系統檔案總管中顯示**」
3. 一路往上找到**最外層**的那個資料夾，就是你的 vault（裡面會有一個隱藏的 `.obsidian` 資料夾）
4. 點一下檔案總管上方的網址列，複製路徑，例如：`D:\Obsidian\我的筆記`

先把這個路徑記下來。

### 步驟 3：打開 PowerShell

1. 打開你想安裝工具的資料夾，例如「文件」
2. 在檔案總管的網址列輸入 `powershell`，然後按 Enter
3. 會跳出一個藍色或黑色的視窗，這就是 PowerShell

> 💡 macOS / Linux：打開「終端機」，先用 `cd` 移到想安裝的位置。

### 步驟 4：貼上安裝指令

把下面這行指令**整行複製**，把 `D:\Obsidian\我的筆記` 換成步驟 2 記下的路徑，貼到 PowerShell 後按 Enter：

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/sunfish1728/bge-obsidian/main/install.ps1))) -Vault "D:\Obsidian\我的筆記" -Index
```

macOS / Linux 使用這行（同樣要換成你的路徑）：

```bash
curl -fsSL https://raw.githubusercontent.com/sunfish1728/bge-obsidian/main/install.sh | bash -s -- --vault ~/Obsidian/我的筆記 --index
```

接著畫面會開始跑出很多文字，這是正常的。看到下面這行就代表**完成了**：

```
Done. Open the vault in Claude Code or Codex and ask it to organise _inbox.
```

> 💡 安裝程式會自動判斷你有沒有 NVIDIA 顯示卡：有的話用顯示卡（比較快），沒有的話用 CPU（比較慢，但一樣能用）。

### 步驟 5：在 vault 裡建立收件匣

在你的 vault 裡，新增一個名為 **`_inbox`** 的資料夾（在 Obsidian 左側空白處按右鍵 →「新增資料夾」）。

**安裝完成！** 🎉

---

## 第二部分：日常使用

### 1. 把新資料丟進 `_inbox`

PDF、`.md` 筆記、`.txt` 文字檔都可以，直接拖進 `_inbox` 資料夾就好。

### 2. 在 vault 資料夾裡打開 Claude Code 或 Codex

- **Claude Code 桌面版**：開啟一個新工作，資料夾選擇你的 vault
- **Claude Code 終端機版**：在 vault 資料夾開啟 PowerShell，輸入 `claude`
- **Codex**：在 vault 資料夾開啟 PowerShell，輸入 `codex`

> ⚠️ 一定要在 **vault 資料夾**裡打開，它才找得到這個工具。

### 3. 用一般說話的方式下指令

直接打字就好，例如：

| 你想做的事 | 這樣說 |
|---|---|
| 整理新資料 | `幫我把 _inbox 的資料整理進 vault` |
| 處理某個檔案 | `幫我處理 _inbox 裡的機器學習講義.pdf` |
| 找出重複的筆記 | `檢查我的 vault 有沒有重複的筆記` |
| 找相關筆記 | `找跟「梯度下降」有關的筆記` |
| 建立標籤 | `幫我根據 vault 的內容建立一套標籤` |

它會先告訴你打算怎麼做。如果一次要動超過 10 個檔案，會先列出清單讓你確認。

### ⭐ 建議第一次使用時先說這句

```
幫我根據 vault 的內容建立一套標籤
```

它會分析你所有的筆記，提出一套標籤讓你確認。有了這套標籤，之後自動加的標籤會準確很多。

---

## 常見問題

**Q：第一次下指令時好像卡住了？**
第一次要載入 AI 模型，大約需要 20 秒（沒有顯示卡的話更久）。之後每次只要不到 1 秒。閒置 30 分鐘後，模型會自動關閉以釋放記憶體。

**Q：我新增了很多筆記，需要重新做什麼嗎？**
不用。每次使用時，它都會自動找出新增或修改過的筆記並更新。

**Q：怎麼更新到最新版？**
在當初安裝的位置，**重新執行一次步驟 4 的指令**即可。

**Q：被移到 `_review` 的檔案怎麼辦？**
那些是被判斷為重複的資料。打開來看看：確定不要就自己刪掉，想留就移回原本的位置。工具不會自動刪除任何檔案。

**Q：掃描版的 PDF（圖片做成的 PDF）可以用嗎？**
不行。它只能讀取有文字的 PDF，遇到掃描版會告訴你。

**Q：出現紅色的 `ERROR`？**
- `git is required` → 回到步驟 1 安裝 Git，然後**關掉 PowerShell 重新打開**再試
- `Python 3.10+ is required` → 回到步驟 1 安裝 Python，記得勾選「Add python.exe to PATH」
- `vault folder not found` → 步驟 4 的路徑打錯了，檢查一下，路徑前後要有雙引號 `"`

**Q：想換成另一個 vault？**
在安裝資料夾裡重新執行步驟 4 的指令，把路徑換成新的 vault。

**Q：怎麼解除安裝？**
1. 在安裝資料夾執行：`.venv\Scripts\python scripts\install_skill.py --uninstall`（這會移除 vault 裡的 Skill）
2. 刪除整個 `bge-obsidian` 資料夾

所有檔案都在這個資料夾裡，不會在電腦其他地方留下東西。

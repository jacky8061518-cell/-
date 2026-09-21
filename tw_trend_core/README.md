# TW Trend Core

**TW Trend Core** 是 [AI Trend Core](../ai_trend_core/) 的台股版本：對台灣**全部上市
＋上櫃個股**（約 1,900 多檔，排除 ETF）一次性計算 20 期布林帶與 Z-Score，
找出統計上顯著偏離（|Z-Score| > 2）的異常標的，交給 Claude Code 扮演首席策略官
分析、查新聞、給建議。做法與 `ai_trend_core/daily_scan.py` 的「手動智慧模式」完全相同，
不需要 Anthropic API Key，不會產生 API 費用。

## 為什麼不是重寫一套抓資料的邏輯？

台股上市櫃全市場有近 2,000 檔個股，若像 `ai_trend_core` 一樣對每一檔個別呼叫
yfinance，會非常慢且容易被限流。本專案改為重用 repo 根目錄
`sector-rotation-research` 專案已維護的基礎設施：

- **股票池**：`data/databases/tw/security-master.csv`（該專案已整理好的上市 + 上櫃
  證券主檔，含 Yahoo ticker），篩選 `Asset type == "股票"` 即為全部個股清單。
- **價格下載**：`src/sector_rotation/data.py` 的 `download_adjusted_prices()`，
  以 100 檔為一批呼叫 yfinance，內建重試與台股股本/除權息斷點修補邏輯。
- **歷史種子**：`data/databases/tw/adjusted-prices.parquet` 已有 2012 年至今的
  完整歷史（該檔案由 sector-rotation-research 定期更新並提交進 git），
  第一次執行時直接讀取做為種子（零網路成本），只對外抓取種子之後的新鮮區間。

**重要**：本專案只**讀取**這些共用檔案，從不寫回。所有下載結果都寫入自己獨立的
快取檔 `tw_trend_core/data/tw_price_cache.parquet`，不會影響
`sector-rotation-research` 專案自己的資料。

Z-Score 與布林帶的計算則是**向量化**：對整張「日期 × 全部標的」寬表一次做
`rolling(20).mean()` / `rolling(20).std()`，而不是對近 2,000 檔個股逐一跑迴圈，
一次全市場掃描通常在數分鐘內完成。

## 專案結構

```
tw_trend_core/
├── core/
│   ├── universe.py       # 讀取上市＋上櫃全部個股清單
│   └── quant_engine.py   # 向量化布林帶 / Z-Score 計算、價格資料載入與快取
├── tests/                 # pytest 單元測試
├── data/                  # 自有價格快取與最新掃描結果（執行時自動產生，不進 git）
├── daily_scan.py          # 全市場掃描腳本
├── pytest.ini
└── requirements.txt
```

## 使用方式

### 1. 安裝套件

沿用 `ai_trend_core/.venv` 即可（已包含 pandas / yfinance / pyarrow），
或自行建立虛擬環境安裝 `requirements.txt`。

### 2. 執行全市場掃描

```bash
cd tw_trend_core
python daily_scan.py
```

第一次執行會從共用的 `adjusted-prices.parquet` 讀出種子歷史，只對外抓取近期
新鮮資料，通常數分鐘內完成；之後每次執行只需抓取自己快取檔案最後一筆日期起算
的近期資料，會更快。執行完會：

1. 印出全市場掃描統計（掃描標的數、成功計算 Z-Score 的標的數、異常標的數）。
2. 把所有 `|Z-Score| > 2` 的異常標的（代碼、名稱、市場、產業、價格、均值、
   布林帶、Z-Score）整理成 JSON，印在終端機，同時寫入 `data/latest_tw_scan.json`。

### 3. 交給 Claude Code 分析

把終端機印出的 JSON 複製貼給 Claude Code，或請它直接讀取
`data/latest_tw_scan.json`。由於全市場掃描可能同時找到數十檔異常標的，
建議先請 Claude Code 依 `|Z-Score|`、產業別、或你關心的個股篩選出幾檔即可，
不需要對每一檔都做完整新聞分析與查證。

## 執行測試

```bash
pytest
```

測試涵蓋向量化 Z-Score 計算與逐檔手動計算結果一致性、歷史不足標的的正確排除
（同時避免把「均值/標準差本身缺值」與「價格持平導致 0/0 = NaN」兩種情況混為一談）、
異常標的篩選與排序，以及股票池讀取的正確性（涵蓋上市與上櫃、排除 ETF）。

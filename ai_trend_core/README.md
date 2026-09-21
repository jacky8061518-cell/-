# AI Trend Core

**AI Trend Core** 是一套自主市場掃描與交易決策平台，結合量化統計引擎（布林帶 / Z-Score 異常偵測）
與 CrewAI 多智能體團隊（市場偵察員、情緒分析師、首席策略官），24/7 自動掃描市場、
偵測統計異常，並產出附帶理由與信心評分的交易建議，最終透過 Streamlit 儀表板呈現。

## 專案結構

```
ai_trend_core/
├── core/                 # 核心邏輯
│   ├── quant_engine.py   # 量化引擎：K 線抓取、布林帶、Z-Score、異常偵測
│   └── database.py       # SQLite 資料庫：交易訊號、AI 推理過程持久化
├── agents/                # AI 智能體
│   └── trading_crew.py   # CrewAI：市場偵察員 / 情緒分析師 / 首席策略官
├── ui/                    # 前端界面
│   └── app.py             # Streamlit 可視化儀表板
├── data/                  # 資料庫檔案（trading_signals.db 執行時自動產生）
├── main_loop.py           # 24/7 循環執行腳本（每 15 分鐘掃描一次）
└── requirements.txt
```

## 1. 環境設定

### 1.1 建立虛擬環境

```bash
cd ai_trend_core
python3 -m venv .venv
source .venv/bin/activate   # Windows 請使用 .venv\Scripts\activate
```

### 1.2 安裝套件

```bash
pip install -r requirements.txt
```

> 註：`sqlite3` 為 Python 標準函式庫，無需另外安裝。
>
> 註：布林帶與 Z-Score 目前以 pandas 原生運算實作，`pandas_ta` 並非必要套件
> （其 PyPI 版本僅支援 Python 3.12+），因此未列入 `requirements.txt` 的必裝清單。

### 1.3 設定 API Key

CrewAI 的三個智能體皆以 Claude 3.5 Sonnet 作為大腦，需設定 Anthropic API Key。
情緒分析師使用 DuckDuckGo 搜尋，無需額外金鑰。

於 `ai_trend_core/` 目錄下建立 `.env` 檔案（或直接匯出環境變數）：

```bash
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxxxxx"

# 選填：覆寫預設使用的 Claude 模型
export ANTHROPIC_MODEL="claude-3-5-sonnet-latest"
```

## 2. 啟動系統

系統由兩個獨立程序組成，建議開兩個終端機視窗分別執行：

### 2.1 啟動後端 24/7 掃描迴圈

```bash
python main_loop.py
```

此程序會每 15 分鐘：
1. 呼叫 `quant_engine` 對預設標的（BTC-USD、ETH-USD、NVDA、TSLA、AAPL）進行全市場掃描。
2. 若發現 Z-Score 異常（超買 / 超賣），立即啟動 CrewAI 智能體團隊進行深度分析。
3. 將 AI 交易建議（Action / Entry / TP / SL / 信心評分 / 理由）連同當時市場數據存入
   `data/trading_signals.db`。
4. 於終端機輸出精簡的執行日誌。

### 2.2 啟動前端儀表板

```bash
streamlit run ui/app.py
```

儀表板包含：
- **即時看板**：目前掃描標的的價格、Z-Score 與風險等級。
- **信號牆（Signal Wall）**：最新的 AI 交易指令，並以顏色標註信心值高低。
- **圖表分析**：帶有布林帶與均線的 Plotly K 線圖。
- **AI 思考過程**：展示情緒分析師與首席策略官的推理摘要。

## 3. 自訂監控標的

可於 `core/quant_engine.py` 修改 `DEFAULT_SYMBOLS` 清單，或在 Streamlit 儀表板側邊欄中
即時調整欲監控的標的。

## 4. 注意事項

- 本平台僅供研究與教育用途，所有交易建議皆由 AI 生成，不構成投資建議，實際交易請自行評估風險。
- `main_loop.py` 與 `ui/app.py` 各自獨立讀寫同一個 SQLite 資料庫，可同時執行。

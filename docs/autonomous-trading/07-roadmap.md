# 07 落地路線與現況

## 0. 目前狀態（v1：訊號與情報看板）

本 repo 已經實作出設計文件中 L0–L4 與 L6 的可運作版本，**不含 L5 執行層**。
系統只發訊號，不下單。

```text
✅ L6 介面層   Streamlit 多頁交易台（總覽／訊號流／風險面板／情報／簡報／成效／健康）
⬜ L5 執行層   不在 v1 範圍
✅ L4 風控層   RiskEngine：波動率定量、分數凱利、叢集限額、熔斷矩陣
✅ L3 決策層   DecisionLayer：證據融合、信心分解、訊號預算
✅ L2 Agent 層 Scanner / Regime / Anomaly / Flow / Sentiment / Corroborator / Narrative
✅ L1 特徵層   Blackboard + 品質標記（正常／過期／推估／缺漏）
✅ L0 資料層   台股本地資料庫（2,333 檔、2012 起、法人買賣超）；新聞為可插拔介面
```

## 1. 程式碼位置對照

| 設計文件章節 | 實作位置 |
| --- | --- |
| [01](01-architecture.md) 分層與 point-in-time | `src/trading_desk/loaders.py`、`agents/base.py` 的 `MarketContext.visible_*` |
| [01](01-architecture.md) 三種運行模式 | `MarketContext.mode`（backtest／paper／live 共用同一條路徑） |
| [01](01-architecture.md) 失效與降級 | `agents/base.py` 的 `AgentRunner`、`Blackboard.record_degradation` |
| [01](01-architecture.md) 回放與稽核 | `contracts.stable_id`、`store.DeskStore.find_decision`、系統健康頁 |
| [02](02-agents.md) Agent 名冊 | `src/trading_desk/agents/` 各檔 |
| [02](02-agents.md) 黑板模式 | `src/trading_desk/blackboard.py` |
| [02](02-agents.md) 防幻覺三件套 | `news.LLMAnnotator._validate`（引用驗證＋標的驗證） |
| [03](03-quant-models.md) 穩健統計 | `src/trading_desk/statistics.py` |
| [03](03-quant-models.md) 市場中性化 | `statistics.market_neutral_returns`、`agents/anomaly.py` |
| [03](03-quant-models.md) 情緒量化 | `news.aggregate_sentiment` |
| [04](04-risk-control.md) 三層防線 | `risk.RiskEngine.size` / `apply_portfolio_limits` / `evaluate_breakers` |
| [05](05-signals-alerts.md) 訊號卡 | `contracts.SignalCard`、`ui.render_signal_card` |
| [05](05-signals-alerts.md) 警示分級與預算 | `alerts.AlertRouter` |
| [05](05-signals-alerts.md) 每日簡報 | `src/trading_desk/brief.py` |
| [05](05-signals-alerts.md) 交易員否決權 | `store.Feedback`、訊號卡上的採用／忽略／否決 |
| [06](06-data-contracts.md) 資料契約 | `src/trading_desk/contracts.py`、`store.py` 的 JSONL 格式 |

## 2. v1 已知的限制

誠實列出，因為知道系統哪裡不能用，和知道它哪裡能用一樣重要：

| 限制 | 影響 | 何時該處理 |
| --- | --- | --- |
| **沒有成交量資料** | 流動性只能用「發行股數×收盤價」代理；無法算 Amihud、無法估容量 | 接入日成交量後立即補 |
| **ATR 由波動率近似** | 停損距離不如真實 ATR 精確 | 接入高低價後 |
| **沒有真實 point-in-time 資料庫** | 目前靠「只讀 as_of 之前的列」達成，但資料本身沒有 `ingested_at`／`revision` | 上真錢前必須做 |
| **語意層是規則式** | 詞典標註遠不如 LLM，但永遠可用且不會幻覺 | 接上 LLM annotator（介面已備妥） |
| **敘事聚類靠產業** | 跨產業的主題（如「AI 資本支出」）抓不到 | 需要 embedding 聚類 |
| **只有台股** | 美股需接資料源 | 依需求 |
| **沒有回測驗證** | 訊號成效頁只能評估已記錄的訊號，樣本極小 | 見下方第 3 節 |

**最後一項最重要**：目前系統能產出結構良好、有理由、有風險預算的訊號，
但**尚未證明這些訊號會賺錢**。這是兩件完全不同的事。

## 3. 下一步（依重要性排序）

### 第一優先：驗證訊號是否有 edge
沒有這一步，前面所有工程都只是漂亮的管線。

1. 寫 walk-forward 回測器，直接餵歷史 `as_of` 進 `run_desk()`
   （架構已支援：`MarketContext.as_of` 就是唯一需要改的東西）
2. 用 [03](03-quant-models.md) §4.2 的三重障礙法產生標籤
3. 產出 [03](03-quant-models.md) §5.2 的完整指標，含 Deflated Sharpe
4. 跑 §5.4 的對抗檢驗：隨機標籤、成本×3、延遲一根 K 棒
5. **如果 edge 不存在，就停在這裡**，回頭改特徵，不要往下做執行層

### 第二優先：資料品質
1. 補上 `ingested_at` 與 `revision`，把「只讀 as_of 之前」升級為真正的 point-in-time
2. 接入成交量與高低價
3. 建立 [06](06-data-contracts.md) §6 的自動品質檢查

### 第三優先：語意層
1. 接上 LLM annotator（`news.LLMAnnotator` 已定義好介面與驗證）
2. 建立 200~500 則的人工標註評測集
3. 加入對抗樣本：諷刺語氣、舊聞重發、同名公司、數字單位陷阱

### 第四優先（僅在前三項完成後）：paper trading
1. 影子帳本：依訊號模擬建倉、追蹤 P&L
2. 連續 60 個交易日
3. 通過 [04](04-risk-control.md) §7 的上線檢查表

**執行層（真錢下單）不在可預見的路線圖上**，除非以上全部完成。

## 4. 排程與自動化

現況是手動觸發（開啟頁面即執行）。要做到 24 小時自動運作：

```text
每個交易日 08:00  scripts/daily_update.py         更新價格與法人資料
每個交易日 08:15  python -m trading_desk.cli run  執行流水線並寫入 runs.jsonl
每個交易日 08:20  產出盤前簡報並推播
盤中每 2 小時      重跑流水線，只在有實質變化時通知
每個交易日 15:00  盤後覆盤，評分昨日訊號
每週日            Post-mortem 週報、模型漂移檢查（PSI）
```

`cli` 尚未實作；目前可用 `run_desk()` 直接寫排程腳本。

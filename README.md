# 自主市場情報交易台 · Autonomous Market Intelligence Desk

一套給交易員使用的多 Agent 市場情報系統，以及它底下原本的輪動研究實驗室。

系統目標**不是**「AI 自動下單賺錢」，而是：

> 讓一群分工明確的 Agent 持續掃描市場，把價格、統計分佈、資金流、情緒與新聞
> 壓縮成**少數幾條帶證據、帶信心分數、帶風險預算的訊號**，
> 交易員只做「執行與否決」。

**目前版本只發訊號，不接券商、不下單。**

## 快速開始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
streamlit run app.py
```

台股資料庫（2,333 檔、2012 年起、法人買賣超）已含在 repo 中，
**不需要外網即可完整運作**。

## 系統長什麼樣子

```text
2,333 檔標的
      ↓  Scanner        全市場價格結構特徵（向量化，約 4 秒）
      ↓  Regime         市場狀態判定，決定所有訊號的信心倍率
      ↓  Anomaly        市場中性化後的統計異常（不是「大盤跌所以每檔都異常」）
      ↓  Flow           法人買賣超與階段判定
   568 個候選
      ↓  Sentiment      只對前 30 檔抓新聞 ← 成本閘門就在這裡
      ↓  Corroborator   跨來源交叉驗證，只能調整信心（有上下限），不能創造訊號
      ↓  Narrative      當日市場主軸聚類
      ↓  Decision       證據融合 → 信心分解 → 訊號預算
      ↓  Risk Engine    波動率定量 × 信心縮放 × 分數凱利 × 叢集限額 × 熔斷
    ≤10 條訊號卡  →  交易員採用／忽略／否決
```

每條訊號都必須回答三個問題，答不出來的不會發出：

1. **依據哪些證據？** 每項證據標明類型、權重、方向、來源 Agent
2. **信心多少？** 分解成 base × 交叉驗證 × 市場環境 − 反面證據扣分
3. **錯了會賠多少？** 建議權重、停損、最大虧損 %NAV、以及**失效條件**

訊號卡同時顯示**反面證據**。只講單邊理由的系統，交易員三週後就不會再信它。

## 頁面

| 頁面 | 用途 |
| --- | --- |
| **總覽** | 今天需要決定的事、市場主軸、系統狀態 |
| **訊號流** | 完整訊號清單，可依方向／級別／證據類別篩選，展開看完整證據鏈 |
| **風險面板** | 熔斷距離、相關性叢集曝險、有效持倉數、每筆定量的綁定限制 |
| **情報與敘事** | 市場狀態判定、主題聚類、語意層輸出與被拒絕的標註 |
| **每日簡報** | 一頁式盤前簡報與盤後覆盤，可下載 Markdown |
| **訊號成效** | 誠實的覆盤：命中率、信心分層表現、最差訊號、交易員回饋分佈 |
| **系統健康與回放** | Agent 執行狀態、資料新鮮度、依 `decision_id` 重建完整決策鏈 |
| **研究實驗室** | 原有的輪動研究工作台（資金流雷達、RRG、回測） |

## 設計文件

完整的架構設計在 [`docs/autonomous-trading/`](docs/autonomous-trading/)：

| 文件 | 內容 |
| --- | --- |
| [README](docs/autonomous-trading/README.md) | 六條不可妥協的原則 |
| [01 底層架構](docs/autonomous-trading/01-architecture.md) | 分層、事件匯流排、point-in-time、降級策略、回放 |
| [02 Agent 分工](docs/autonomous-trading/02-agents.md) | Agent 名冊、DAG 編排、LLM 邊界、防幻覺 |
| [03 量化模型](docs/autonomous-trading/03-quant-models.md) | 穩健統計、情緒量化、meta-labeling、驗證方法論 |
| [04 風險控制](docs/autonomous-trading/04-risk-control.md) | 三層防線、定量公式、熔斷矩陣、上線檢查表 |
| [05 訊號與警示](docs/autonomous-trading/05-signals-alerts.md) | 訊號卡格式、警示分級、簡報、工具清單 |
| [06 資料契約](docs/autonomous-trading/06-data-contracts.md) | 所有 schema 與資料品質 SLA |
| [07 落地路線](docs/autonomous-trading/07-roadmap.md) | 現況、已知限制、下一步 |
| [08 技術選型](docs/autonomous-trading/08-tech-stack.md) | 依賴取捨、LLM 成本控管、部署 |

## 幾個刻意的設計決定

**風控獨立於策略。** 策略只能提出請求，`RiskEngine` 決定通不通過。
任何讓策略修改自身限額的設計，都是把煞車踏板接到油門上。

**LLM 不做算術。** 語意層只負責分類、抽取、引用；所有涉及金額、機率、
權重的計算一律走確定性程式碼。LLM 標註若無法逐字引用原文，整筆丟棄。

**訊號有每日預算。** 超過就自動提高門檻，而不是照發。
被洗版的告警系統，第二週就會被交易員關掉。

**信心永遠不到 100%。** 四個啟發式互相同意是鼓舞，不是確定。上限 90%。

**「未啟用」與「過期」是不同狀態。** 沒訂閱的資料源不該讓所有訊號被打折。

**同一份程式碼跑回測、paper 與 live。** 模式只由 `MarketContext.mode` 注入。
若兩者走不同分支，那條分支就是未來所有 bug 的溫床。

## 已知限制（重要）

系統目前能產出結構良好、有理由、有風險預算的訊號，
但**尚未證明這些訊號會賺錢**。這是兩件完全不同的事。

- 沒有成交量資料，流動性只能用市值代理
- ATR 由波動率近似，停損距離不如真實 ATR 精確
- 資料尚未有 `ingested_at`／`revision`，point-in-time 是靠「只讀 as_of 之前的列」達成
- 語意層預設是詞典規則式（永遠可用、不會幻覺，但遠不如 LLM）
- 尚未跑過 walk-forward 回測驗證

完整清單與處理順序見 [07 落地路線](docs/autonomous-trading/07-roadmap.md) §2、§3。

## 執行測試

```bash
pip install -e ".[dev]"
pytest
```

---

# 研究實驗室（原有功能）

以下是本專案原有的多層輪動研究工具，現在是交易台的「研究實驗室」頁面。

An explainable, parameter-driven Streamlit workbench for researching daily,
weekly, and monthly momentum rotation across sectors, detailed industries,
structural themes, investment styles, and curated thematic stock baskets.

### 研究流程

```text
Adjusted ETF prices
        ↓
Daily / weekly / monthly observations
        ↓
Frequency-specific multi-horizon momentum
        ↓
Composite or risk-adjusted score
        ↓
Positive-momentum filter and top-N ranking
        ↓
Equal, momentum, or inverse-volatility weights
        ↓
One-month signal shift
        ↓
Cost-aware backtest and research dashboard
```

The one-period shift is deliberate: a ranking observed at the end of a day,
week, or month is applied to the following period. This prevents the backtest
from earning a return before the signal was known.

## Research universes

Version 0.2 includes:

- all 11 U.S. Select Sector SPDR ETFs;
- detailed technology, financial, health-care, industrial, consumer, energy,
  mining, real-estate, and infrastructure industry ETFs;
- AI, robotics, cloud, cybersecurity, clean-energy, nuclear, EV, space,
  genomics, FinTech, blockchain, digital-consumer, and sustainability themes;
- growth, value, dividend, quality, low-volatility, size, international,
  real-asset, and rates ETFs;
- curated stock baskets for AI compute, cloud software, cybersecurity,
  semiconductor equipment, data-center power, nuclear, space and defense,
  EVs, metabolic health, FinTech, and computational biology;
- user-entered custom tickers.

Version 0.3 adds:

- an incremental Parquet market-data cache;
- dated daily, weekly, and monthly ranking snapshots;
- scheduled weekday updates after the U.S. close;
- ETF top-holdings retrieval;
- holding-weight × stock-momentum leadership analysis;
- an in-app "attention and leading players" workflow.

Version 0.4 adds a Taiwan market system with:

- official TWSE and TPEx company-master ingestion;
- automatic `.TW` and `.TWO` ticker resolution;
- official Taiwan industry groups and curated Taiwan themes;
- Taiwan-listed equity and bond ETF universes;
- `0050.TW` as the equity benchmark and `00679B.TWO` as the defensive asset;
- separate U.S. and Taiwan Parquet databases and ranking histories.

Version 0.4.1 adds a Taiwan corporate-action data-quality guard. Yahoo Finance
occasionally leaves a split or capital-reduction boundary unadjusted. Internal
Taiwan price jumps above 40% are now rebased before signal and return
calculation, and the same repair is applied to previously saved caches. This
fixes the false 75% drop in the 0050 history at the start of 2014.

Version 0.4.2 makes the 0050 benchmark return basis explicit. The performance
chart now shows both dividend-reinvested total return and split-adjusted,
price-only return. The official 4-for-1 split effective 2025-06-18 is treated
as a unit conversion and never as an investment gain or loss.

Version 0.4.3 versions the Streamlit price-data cache so stale pre-fix 0050
history cannot survive a code update. The Taiwan performance view also reports
the active pipeline version, benchmark start date, cumulative benchmark growth,
and maximum adjusted daily move. Curated theme backtests now display an
explicit survivorship- and selection-bias warning.

Version 0.4.4 adds an independently adjustable backtest end date. Every
sidebar parameter change now reruns the requested historical window
automatically, including market, universe, groups, rebalance frequency,
start/end dates, top-N, weighting, risk adjustment, momentum filter,
defensive asset, turnover cost, and multi-horizon momentum weights.

Version 0.4.5 presents the main performance chart as cumulative return
percentages instead of growth-of-$1 multiples. Hover labels now identify each
series and report an unambiguous cumulative-return percentage.

Version 0.4.6 adds direct numeric inputs for multi-horizon momentum weights and
a `Custom rank weight` portfolio method. Users can enter Rank 1 through Rank N
position percentages; the app normalizes the inputs and applies them to each
period's newly selected securities.

Version 0.5.0 adds an explainable industry-rotation workbench:

- RRG-style Leading, Improving, Weakening, and Lagging quadrants;
- configurable RS-Ratio, RS-Momentum, and trajectory windows;
- equal-weight industry indices relative to SPY or 0050;
- short- and medium-horizon excess returns;
- relative breadth showing how many constituents beat the benchmark;
- top stocks driving each industry's move;
- deterministic explanations for why each group is strengthening or weakening;
- downloadable industry-rotation diagnostics.

Version 0.6.0 expands Taiwan coverage from curated baskets to the current
official full market:

- every company in the TWSE and TPEx company masters;
- every currently quoted TWSE and TPEx ETF, excluding ETNs;
- uncapped official-industry selection;
- official market, asset-type, industry, and ETF-type metadata;
- chunked full-history downloads with a daily incremental Parquet cache;
- a security master and coverage report showing observation counts and
  backtest readiness for every security.

Version 0.7.0 adds a Taiwan institutional fund-flow layer:

- official TWSE and TPEx foreign-investor, investment-trust, and dealer flows;
- 5-day flow acceleration and 20-day accumulation;
- estimated net-flow value and market-cap-normalized flow intensity;
- industry flow breadth and the leading stocks behind each group;
- separate accumulation, price-confirmation, deceleration, and outflow stages;
- adjustable flow/price weights and downloadable stock/industry flow tables.

Version 0.8.0 makes fund flow the primary Taiwan research surface:

- a fund-flow summary appears before momentum and performance;
- the fund-flow radar is the first analysis tab;
- investor-to-industry Sankey flows;
- five-dimensional industry radar comparisons;
- daily industry flow trajectories and investor-structure heatmaps;
- flow acceleration, breadth, dominant investor, and concentration diagnostics;
- deterministic flow-reason narratives and research-action labels;
- optional headline lookup for the stocks driving a selected industry's flow.

SPY is the benchmark. SHY is the defensive asset when the positive-momentum
filter rejects every sector.

## 專案結構

```text
.
├── app.py                          導覽入口（st.navigation）
├── pages/
│   ├── 0_總覽.py
│   ├── 1_訊號流.py
│   ├── 2_風險面板.py
│   ├── 3_情報與敘事.py
│   ├── 4_每日簡報.py
│   ├── 5_訊號成效.py
│   ├── 6_系統健康與回放.py
│   └── 9_研究實驗室.py
├── src/
│   ├── trading_desk/               交易台引擎
│   │   ├── contracts.py            所有跨層資料契約
│   │   ├── statistics.py           穩健估計（MAD、EWMA、rank→常態、PSI）
│   │   ├── news.py                 新聞去重、標註、防幻覺驗證
│   │   ├── blackboard.py           Agent 共享狀態
│   │   ├── agents/                 Scanner / Regime / Anomaly / Flow /
│   │   │                           Sentiment / Corroborator / Narrative
│   │   ├── decision.py             證據 → 訊號卡
│   │   ├── risk.py                 定量、組合限額、熔斷
│   │   ├── alerts.py               警示分級與注意力預算
│   │   ├── brief.py                每日簡報
│   │   ├── store.py                append-only 執行日誌與交易員回饋
│   │   ├── loaders.py              point-in-time 資料存取
│   │   ├── desk.py                 run_desk()：唯一的執行入口
│   │   └── ui.py                   共用 Streamlit 元件
│   └── sector_rotation/            研究實驗室的原有模組
├── docs/autonomous-trading/        架構設計文件
├── scripts/daily_update.py
└── tests/
```

## Automated daily update

Run the update manually with:

```bash
.venv/bin/python scripts/daily_update.py
```

The first run downloads full history. Later runs retrieve only a short overlap,
merge it into two separate databases:

```text
data/databases/us/adjusted-prices.parquet
data/databases/tw/adjusted-prices.parquet
data/databases/tw/security-master.csv
data/databases/tw/coverage-report.csv
data/databases/tw/institutional-flows.parquet
```

Market-specific snapshots are written to:

```text
data/snapshots/us/latest-daily-ranking.csv
data/snapshots/us/latest-weekly-ranking.csv
data/snapshots/us/latest-monthly-ranking.csv
data/snapshots/tw/latest-daily-ranking.csv
data/snapshots/tw/latest-weekly-ranking.csv
data/snapshots/tw/latest-monthly-ranking.csv
data/snapshots/us/latest-{daily,weekly,monthly}-industry-rotation.csv
data/snapshots/tw/latest-{daily,weekly,monthly}-industry-rotation.csv
```

The Codex automation created for this project runs at 18:00 America/New_York
on weekdays. Weekend and exchange-holiday runs simply retain the most recent
available trading session.

## Deploy on Streamlit Community Cloud

1. Push the project root to a GitHub repository. `app.py`, `pages/`,
   `requirements.txt`, `pyproject.toml`, `src/`, `data/databases/tw/`, and
   `.streamlit/config.toml` must all be included.
2. In the local app, click **Deploy** and choose
   **Streamlit Community Cloud → Deploy now**.
3. Connect GitHub, then select the repository and the `main` branch.
4. Set **Main file path** to `app.py`.
5. In **Advanced settings**, select Python 3.11. No secrets are required for
   the current public-data configuration.
6. Click **Deploy**. Subsequent pushes to the selected GitHub branch update the
   hosted application automatically.

## Important limitations

This software is for research and education. It does not include every source
of data-vendor bias, survivorship bias, taxes, bid-ask spread, market impact, or
execution uncertainty. Yahoo Finance data may be delayed, revised, or
unavailable. The Taiwan discontinuity guard reduces large omitted
corporate-action errors but cannot guarantee that every vendor revision is
identified. Backtested and synthetic results do not guarantee future
performance. Nothing in this repository is personalized investment advice.

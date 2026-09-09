# Phase 1 可重用模組清單

依 SPEC 第 11 節 Phase 1 第 1 點：「先讀那個專案的程式碼，列出可重用的模組清單再動手。」

**來源**：`sector-rotation-research`（即 SPEC 所稱 taiwan-fund-flow-lab），
位於本 repo 根目錄 `src/sector_rotation/`，2098 行，11 個模組。已全數讀畢。

**既有資料資產**（可直接繼承，不需重抓）：

| 檔案 | 內容 | 規模 |
|---|---|---|
| `data/databases/tw/security-master.csv` | 台股上市櫃公司與 ETF 主檔 | 2333 檔 × 10 欄 |
| `data/databases/tw/adjusted-prices.parquet` | 日收盤價 | 3560 日 × 2333 檔，2012-01-02 起 |
| `data/databases/tw/institutional-flows.parquet` | 三大法人買賣超 | 47317 列 |

---

## 一、結論先講

**可以直接沿用的：對外部資料源的抓取與解析知識。** 那是這 2098 行裡最有價值、
最不該重寫的部分——TWSE/TPEx 的 API 端點、民國年日期格式、欄位索引位置、
上市上櫃的欄位差異、ETF 與 ETN 的代號前綴區別。這些是踩過坑才知道的事。

**不能直接沿用的：資料的儲存形狀與計算路徑。** 因為既有專案是一個
「看當下」的研究工具，本系統要的是「可以回到任一歷史時點」的系統。這兩者的
差別不是品質高低，是目標不同。

一個數字說明差距：**既有 2098 行裡，`asof` 參數出現 0 次。**

---

## 二、分類清單

### A 級：直接沿用，只包一層轉接（約 620 行）

這些是純粹的「對外部世界的知識」，邏輯正確且與時間紀律無關。

| 模組 | 具體內容 | 用在 Phase 1 哪裡 |
|---|---|---|
| `taiwan.py` 的 API 常數與 `_get_json` | TWSE/TPEx 五個 openapi 端點、User-Agent 設定 | L0 ingestion worker |
| `taiwan.py::fetch_taiwan_company_master` | 上市上櫃公司主檔抓取與欄位正規化 | L0 → 實體主檔 |
| `taiwan.py::fetch_taiwan_etf_master` | ETF 主檔。**含一個關鍵知識**：TPEx 的收盤報價 feed 同時含股票與 ETF，ETF 代號以 `00` 開頭而 ETN 用 `02`，需排除 | L0 → 實體主檔 |
| `taiwan.py::INDUSTRY_NAMES` | 37 個官方產業代碼對照表 | L1 產業分類，橫斷面中性化要用 |
| `taiwan.py::_classify_etf` | 由基金名稱判斷 ETF 類別（債券／商品／槓桿反向／主動式／海外） | L1 資產分類 |
| `fund_flow.py::fetch_twse_institutional_flows` | TWSE T86 三大法人。**欄位索引 4/10/11/18 是硬知識** | L0 籌碼 worker |
| `fund_flow.py::fetch_tpex_institutional_flows` | TPEx 三大法人。**民國年格式 `f"{year-1911:03d}/{m}/{d}"` 與欄位索引 4/13/22/23 與上市完全不同** | L0 籌碼 worker |
| `fund_flow.py::_number` | 處理 `"1,234"`、`"--"`、`"---"`、空字串的數值轉換 | L1 標準化 |
| `taiwan.py::TW_THEME_CODES` | 19 個主題、約 150 檔股票的人工分類 | 實體解析的別名種子 |

**沿用方式**：包成 `ingestion/twse.py`、`ingestion/tpex.py`，
函式簽章加上 `fetch_time` 並回傳原始 payload 一併落地（SPEC 3.2 raw landing zone
要求原始 payload 永不覆寫）。內部解析邏輯逐字沿用，不改。

### B 級：邏輯有用，但要改寫成 point-in-time（約 400 行）

| 模組 | 為什麼要改 | 改成什麼 |
|---|---|---|
| `snapshots.py::update_price_cache` | 增量更新的骨架是對的，但它**覆寫**同一個 Parquet，且用 `date.today()` 決定抓到哪 | 改為 append-only，路徑含抓取時間戳；時間改由 `core.clock.utc_now()` 提供 |
| `fund_flow.py::update_institutional_flow_cache` | 同上。`drop_duplicates(keep="last")` 會**丟掉舊版本**，正是雙時間戳要保留的東西 | 保留每個版本，用 `ingest_time` 區分 |
| `holdings.py::fetch_top_holdings` | ETF 持股抓取邏輯可用 | 加 `asof`；持股是會變的，需版本化 |
| `data.py::download_adjusted_prices` | 批次抓取、失敗後逐檔重試的策略很實用 | 保留重試策略，但改抓**未調整**價（見 C 級） |
| `metrics.py` 全部 | CAGR/Sharpe/回撤/勝率，數學正確 | Phase 2 回測用，但要接 `periods_per_year` 的正確參數 |

### C 級：與 SPEC 直接衝突，必須重做（約 200 行）

這一類不是「寫得不好」，是**設計目標不同**，照抄會破壞系統的核心保證。

| 模組 | 衝突點 | SPEC 依據 |
|---|---|---|
| `data.py::repair_taiwan_price_discontinuities`（60 行） | 它偵測單日 >40% 跳動並**回頭乘算修改歷史價格**。SPEC 要求存未調整原始價 + 獨立調整因子表，因為「新的企業行動會讓歷史調整價全部改變，破壞可重現性」。這個函式做的正是被禁止的事 | SPEC 3.2 第 1 點 |
| `download_adjusted_prices(auto_adjust=True)` | 拿到的是已調整價，來源方一旦改調整方式，歷史就變了 | 同上 |
| 全部價格用 `float` | 浮點誤差在累乘後不可逆 | Phase 0 已定 `Bar` 用 `Decimal` |
| `strategy.py::resample_prices` 的 `tz_localize(None)` | 直接剝除時區資訊 | CLAUDE.md 第 4 條：內部一律 UTC |
| 4 處 `date.today()` / `Timestamp.today()` | 直接讀牆上時鐘 | CLAUDE.md 第 1 條；會被 Phase 0 的 guard 測試攔下 |
| 無 `universe_membership` 概念 | 主檔只有「現在」還在市的標的，下市股票不存在 → 存活者偏誤 | SPEC 3.2 第 2 點 |

**注意 `repair_taiwan_price_discontinuities` 的處理方式**：這個函式的**偵測**部分
（台股有漲跌幅限制，所以單日 >40% 必是企業行動而非報酬）是有價值的領域知識，
要保留下來當作**調整因子表的推導線索與資料品質檢查**；但它的**修改**部分
（回頭乘算歷史）必須丟掉，改成產生一筆調整因子紀錄。

### D 級：與本系統無關，不沿用（約 880 行）

`universe.py` 的美股 ETF 分類表（約 265 行）、`rrg.py`（188 行，RRG 圖表計算）、
`strategy.py` 的動量排序與組合建構（251 行）、`config.py` 的 GICS 板塊 ETF。

這些屬於 SPEC 分層裡的 L2／L4，不是 Phase 1 的資料層。`strategy.py` 的
`run_backtest` 有做 `shift(1)` 避免前視，思路正確，但 Phase 2 要的是
**逐事件推進**的引擎，不是向量化回測——SPEC 明文禁止「向量化捷徑造成的隱性前視」。
所以 Phase 2 會重寫，不是沿用。

`universe.py::AssetInfo` 這個 dataclass 會被 Phase 0 的 `Instrument` 取代。

---

## 三、既有資料檔怎麼處理

`adjusted-prices.parquet` 有 3560 天 × 2333 檔的歷史，這是**十四年的資料**，
重抓要很久而且 Yahoo 未必給得回來。但它是「已調整價」，違反 SPEC 3.2。

**建議做法（需你確認）**：把它當作**唯讀的歷史種子**保留，
標記為 `source="legacy-adjusted"`、`ingest_time` 設為檔案的 mtime，
並在資料品質元資料上標明「此段歷史無法重建原始未調整價」。
新資料一律走新路徑（未調整價 + 因子表）。

這樣做的代價是：2012–2026 這段歷史的企業行動無法重新推導，
回測跨越這段期間時，調整假設是繼承而來的、不是自己算的。
好處是不用為了純度丟掉十四年資料。

替代方案是從頭重抓未調整價，成本是時間與 API 限制，且更早的歷史可能拿不回來。

---

## 四、數字總結

| 級別 | 行數 | 處置 |
|---|---|---|
| A 直接沿用 | ~620 | 包轉接層，解析邏輯逐字保留 |
| B 改寫 | ~400 | 保留骨架，加 asof 與 append-only |
| C 衝突重做 | ~200 | 偵測邏輯保留，修改邏輯丟棄 |
| D 不相關 | ~880 | 屬 L2／L4，Phase 2 之後再談 |

**約 1000 行有實質重用價值，佔既有程式碼的一半。**
最有價值的部分（外部 API 的踩坑知識）幾乎全部可用，
最需要重做的部分（資料的時間形狀）本來就是 Phase 1 存在的理由。

---

## 五、需要你決定的兩件事

1. **既有 14 年已調整價要保留還是重抓？**（見第三節）
   保留＝快但純度打折；重抓＝慢且可能拿不回完整歷史。我的建議是保留並明確標記。

2. **企業行動的因子表從哪裡來？** 既有程式碼是用「>40% 跳動」反推的，
   那是猜測不是事實。正解是抓 TWSE 的除權除息公告（另一個 API），
   但這是**新的外部資料源**——依 CLAUDE.md「詢問時機」第 1 條，我要先問過你。

這兩題答完我就開始寫 Phase 1 的程式碼。

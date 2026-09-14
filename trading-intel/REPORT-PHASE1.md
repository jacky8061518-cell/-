# Phase 1 驗收報告：資料層與品質閘門

規模：`src/` 2912 行、`tests/` 3206 行、**328 個測試**，`make check` 全綠，
`src/trading_intel` 整體覆蓋率 **95%**。

---

## 驗收條件

| # | SPEC 第 11 節 Phase 1 驗收條件 | 結果 |
|---|---|---|
| 1 | 給定任一歷史日期，可完整重建當日的可交易宇宙 | ✅ 機制通過，**但既有資料本身有存活者偏誤**，見下方 |
| 2 | 刻意注入五種髒資料，五種都被閘門攔下且產生正確告警 | ✅ 五種全數攔下 |
| 3 | 事件流重放兩次產生相同結果 | ✅ 記憶體與真 Redis 兩套實作皆通過 |

另外，SPEC 第 3 點要求的「asof 在企業行動前後查詢同一段歷史，結果應不同」
也通過：`test_the_two_queries_differ`。

---

## 1. 宇宙重建

`normalize/universe.py` 把成員資格記成**區間**而非快照，因此任一天的宇宙都能重建。
用你的真實資料實測：

```
2012-01-03 → 1210 檔
2015-06-30 → 1595 檔
2025-06-30 → 2262 檔
```

機制本身通過：早於某標的上市日的快照不含它；處置股（`RESTRICTED`）
留在宇宙內但不在 `tradable` 集合裡；`asof` 過濾正確——事實上第一次跑測試時
整個宇宙回傳 0 檔，原因就是 `asof` 設得比資料的 `ingest_time` 早，
那是過濾機制正常運作。

### ⚠ 但我實測發現一個必須讓你知道的問題

寫測試時量測既有資料，結果是：

> **2333 檔標的中，最後有值日期等於資料結尾者：2333 檔。**

**沒有任何一檔曾經退出宇宙。**

原因不是前值填補掩蓋了下市，而是更根本的：既有的 `security-master.csv`
是從 TWSE openapi 抓「**現在還在市**」的清單，下市股票從來沒被抓進來過。

這代表：**這份十四年資料不能用來做選股回測。** 從一個全部由倖存者組成的
宇宙裡選股，績效必然虛高，而且不會有任何徵兆。

我的處置是把缺陷變成**有紀錄、可量測的事實**，而不是放鬆測試讓它變綠：

- `ingestion/legacy.py::diagnose_survivorship()` 量化這個偏誤並產生報告；
- `test_legacy_data_is_diagnosed_as_survivorship_biased` 斷言缺陷**確實存在**，
  當資料被換成無偏誤版本時這個測試會失敗——那正是我們希望被通知的時刻；
- 詳細記載在 `docs/ADR/0005-legacy-price-seed.md`，含「可以用它做什麼、
  不可以做什麼」。

**解除條件**：接上交易所的上市下市公告。這是另一個新資料源，等你決定。

## 2. 五種髒資料

`normalize/quality.py` 實作 SPEC 3.3 的五類閘門，26 個測試：

| 髒資料 | 攔截點 | 告警 |
|---|---|---|
| 缺 bar | `check_completeness` | P0，訊息含首個缺漏日期 |
| 負成交量 | `check_raw_sanity` | P0，`成交量為負數` |
| high < low | `check_consistency` | P0，`high 99 < low 101` |
| 重複／異常跳動 | `check_sanity` | P0，`超過 TW 市場的 10.5% 限制（可能是未登錄的企業行動）` |
| 過期 | `check_freshness` | P0，`已過期 N 分鐘` |

外加 PSI 分布漂移（0.25 警示 / 0.5 停用，SPEC 明訂的門檻）。
全部門檻都在 `configs/base.yaml` 的 `quality:` 區塊，程式碼裡沒有魔術數字。

**寫測試時發現的設計缺陷並已修正**：原本 `check_sanity` 吃已驗證的 `RawPrice`，
但 `RawPrice.volume` 有 `ge=0`，負成交量根本建構不起來——閘門永遠測不到。
品質閘門的職責正是在資料**變成型別之前**把關，因此拆出 `check_raw_sanity()`
吃原始 tuple。等到型別驗證失敗才發現，就只剩一個例外，沒有告警、沒有 NO_TRADE 狀態。

彙整規則刻意不做平均、不做投票：任一告警要求停止，結果就是 `NO_TRADE`
（CLAUDE.md 第 6 條）。

## 3. 事件流重放

`core/bus.py` 提供 `EventBus` 協定與兩個實作。**同一組 20 個測試同時跑
`InMemoryBus` 與真正的 `RedisStreamBus`**，強制兩者語意一致。

驗證項目：consumer group 不重複投遞、不同 group 各自看到全部（fan-out）、
ack 減少 pending、**重放不受消費進度影響**、重放兩次結果完全相同、
內容相同的事件共用同一個 `idempotency_key`（重放才能去重）。

除錯過程值得一記：Redis 版本一度失敗，原因是測試的 key 名含 `[redis]`，
而 `[` 在 Redis `KEYS` 的 glob 裡是字元集語法，導致清理靜默失效、
測試之間互相污染。已改用 `scan_iter` 加安全鍵名。

---

## 交付內容

| 模組 | 行數 | 說明 |
|---|---|---|
| `ingestion/landing.py` | 175 | raw landing zone，原始 payload 永不覆寫，路徑含抓取時間戳 |
| `ingestion/twse.py` | 280 | TWSE/TPEx 抓取與解析，**兩者刻意分離** |
| `ingestion/legacy.py` | 260 | 既有十四年資料匯入 + 存活者偏誤診斷 |
| `normalize/corporate_actions.py` | 230 | 企業行動引擎，`get_adjusted_prices(..., asof)` |
| `normalize/universe.py` | 150 | 成員資格區間與快照重建 |
| `normalize/quality.py` | 300 | 五類品質閘門 + PSI |
| `normalize/entities.py` | 240 | 實體解析，低信心進人工佇列 |
| `core/bus.py` | 210 | 事件匯流排，Redis + 記憶體雙實作 |

新增文件：`docs/PHASE1-REUSE.md`（可重用清單）、
`docs/ADR/0005`（既有資料決策）、`docs/ADR/0006`（除權息資料源）。

---

## 沒做到的事

**TWSE API 在本環境連不上。** proxy 明確回報：

```
host: openapi.twse.com.tw:443
kind: connect_rejected
detail: gateway answered 403 to CONNECT (policy denial or upstream failure)
```

這是環境的網路政策，我不能也不該繞過。受影響的範圍與處置：

| 項目 | 狀態 |
|---|---|
| `fetch_twse_dividends()` 等抓取函式 | 已實作，**無法在此環境執行**，網路放行即可用 |
| `parse_twse_dividends()` 等解析函式 | ✅ 已用真實格式樣本完整測試（25 個測試） |
| 真實除權息因子 | ❌ 拿不到。目前只能用 `INFERRED`（跳動反推）或 `LEGACY` |

架構上這是分離的：抓取回傳 bytes 並落地，解析是吃 bytes 的純函式。
網路放行後不需要改任何解析邏輯。信心值機制（`OFFICIAL` = 1.0、
`INFERRED` < 1.0、`LEGACY` = 0.8）讓下游知道手上的因子有多可信。

**SPEC 提到但本階段未做的**：多源交叉核對（需要第二個資料源，
目前連第一個都連不上）、PostgreSQL/TimescaleDB 落地（目前用 Parquet 與
行程內儲存體，介面已是 point-in-time，換儲存體不影響呼叫端）。

---

## 需要你決定的

1. **要不要接上市下市公告資料源？** 這是修好存活者偏誤的唯一方法。
   又是一個新資料源，而且在目前的網路政策下同樣連不上。
2. **TWSE 的網路封鎖要不要處理？** 如果這個環境的政策可以調整，
   Phase 1 就能拿到真實的除權息因子；否則企業行動因子會一直停在推測等級。

這兩件都不擋 Phase 2 開工——Phase 2 的回測引擎本來就規定禁止網路。

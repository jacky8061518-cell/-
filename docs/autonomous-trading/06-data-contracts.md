# 06 資料契約

所有 schema 用同一套原則：明確型別、必填欄位、版本欄位、雙時間戳。
契約先定，程式後寫。契約變更走版本號，不改舊欄位語意。

## 1. 核心維度表

### `security_master`（point-in-time）
```sql
symbol            TEXT      -- 系統內部統一代號，例 "2330.TW"
figi              TEXT      -- 全球識別碼，跨來源對齊用
name              TEXT
exchange          TEXT
currency          TEXT
sector            TEXT
industry          TEXT
listing_date      DATE
delisting_date    DATE      -- NULL 代表仍在市；保留下市股是反倖存者偏誤的關鍵
valid_from        TIMESTAMP -- 這筆屬性從何時起有效
valid_to          TIMESTAMP
PRIMARY KEY (symbol, valid_from)
```
> 現有 `data/databases/tw/security-master.csv` 是此表的雛形，需補
> `valid_from/valid_to` 與 `delisting_date`。

### `session_calendar`
```sql
exchange, trade_date, open_utc, close_utc, is_half_day, is_holiday
```

## 2. 事實表

### `bars`
```sql
symbol, freq ('1m'|'5m'|'1d'), event_time, open, high, low, close,
volume, vwap, adj_factor, ingested_at, revision, source_id
PRIMARY KEY (symbol, freq, event_time, revision)
```
調整因子分開存，不要直接存調整後價格——除權息回溯調整會改寫歷史，
存 `adj_factor` 才能重建「當時看到的價格」。

### `fundamentals`
```sql
symbol, fiscal_period_end, metric, value, unit,
event_time,        -- 實際公布時間，不是期末！
ingested_at, revision, source_id
```
`fiscal_period_end` 與 `event_time` 的區別是避免 look-ahead 的核心。

### `flows`
```sql
symbol, session_date, flow_type, net_value, net_shares,
event_time, ingested_at, source_id
-- flow_type: 'foreign'|'trust'|'dealer'|'margin'|'short'|'etf_creation'|...
```

### `news_raw`
```sql
event_id UUID, source, source_tier, url, headline, body,
published_at, ingested_at, content_hash, simhash
```

### `news_enriched`
```sql
event_id, symbol, role, entity_confidence, event_type, polarity,
materiality, novelty, horizon, evidence_span, model_version,
conflicting BOOL, processed_at
FOREIGN KEY (event_id) REFERENCES news_raw
```

## 3. 特徵倉

### `features`
```sql
symbol, feature_name, feature_set_version, event_time,
value DOUBLE, quality ('ok'|'stale'|'imputed'|'missing'),
computed_at, ingested_at
PRIMARY KEY (symbol, feature_name, feature_set_version, event_time)
```
`quality` 欄位是硬性要求。缺值不能靜默補 0——那會讓「沒資料」被當成
「數值為零」，是自動化系統最陰險的錯誤來源之一。

**查詢介面（唯一入口）**
```python
def get(symbols: list[str],
        features: list[str],
        as_of: datetime,
        lookback: timedelta) -> pd.DataFrame:
    """只回傳 ingested_at <= as_of 的最新 revision。
    backtest 與 live 都走這個函式，沒有第二條路徑。"""
```

## 4. 決策鏈

### `signals`
```sql
signal_id, decision_id, created_at, symbol, direction,
conviction, horizon, thesis, evidence JSONB, counter_evidence JSONB,
regime_context, risk JSONB, invalidation, data_quality JSONB,
model_provenance JSONB, status ('proposed'|'confirmed'|'rejected'|'expired')
```

### `risk_decisions`
```sql
decision_id, signal_id, verdict ('approved'|'reduced'|'rejected'),
requested_weight, approved_weight, binding_constraint, -- 哪條限制卡住了
constraint_snapshot JSONB,   -- 當時所有限額的使用率
decided_at
```
`binding_constraint` 讓你事後能回答「為什麼這單只給了 1% 而不是 3%」。

### `orders` / `fills`
```sql
order_id, decision_id, idempotency_key UNIQUE, symbol, side,
qty, order_type, limit_price, max_slippage_bps, tif,
status, submitted_at, broker_order_id

fill_id, order_id, qty, price, fee, filled_at, broker_fill_id
```
`idempotency_key = sha256(decision_id|symbol|side|target_qty)`。
重送同一個 key 一定不產生第二筆單。

### `trader_feedback`
```sql
signal_id, trader_id, action ('adopted'|'ignored'|'vetoed'),
veto_reason_code, note, responded_at
```

## 5. 訊息（bus payload）

所有訊息共用信封：
```json
{
  "event_id": "uuid",
  "schema_version": "1.0",
  "topic": "signal.proposed",
  "produced_at": "ISO8601",
  "producer": "agent_name@version",
  "causation_id": "觸發這則訊息的上游 event_id",
  "correlation_id": "整條決策鏈的共同 ID = decision_id",
  "payload": { }
}
```
`causation_id` + `correlation_id` 就是回放能力的全部基礎。少了它們，
你只有一堆孤立的日誌，拼不出因果。

## 6. 資料品質 SLA 與監控

| 來源 | 新鮮度 SLA | 缺漏處理 | 品質檢查 |
| --- | --- | --- | --- |
| 即時行情 | < 1 秒 | 延遲 > 3 秒觸發降級 | 價格跳動 > 20% 需複核；買賣價交叉檢查 |
| 日線 | 收盤後 30 分內 | 缺 1 日標記 stale | 與第二來源交叉比對 |
| 法人買賣超 | T+1 18:00 前 | 缺當日則沿用並標記 | 買賣超總和應接近平衡 |
| 財報 | 公布後 4 小時 | — | 與前期比對，變動 > 100% 需複核 |
| 新聞 | < 60 秒 | 斷線 > 5 分告警 | 去重、來源可信度、時間戳合理性 |

**自動品質檢查（每次入庫執行）**：
- 型別與值域（價格 > 0、成交量 ≥ 0、百分比在合理範圍）
- 時間戳單調性與未來時間偵測（`event_time > now` 一律拒收）
- 重複偵測（同 key 同 revision）
- 統計突變（今日值 vs 過去 60 日分佈，超過 6σ 進人工佇列）
- 跨來源一致性（兩個行情源價差 > 0.5% 告警）

現有 `src/sector_rotation/data.py` 的
`repair_taiwan_price_discontinuities()` 正是這類檢查的實例，
應提升為通用的品質檢查框架，並把「修復」改為「標記 + 記錄修復動作」——
靜默修復資料等於銷毀證據。

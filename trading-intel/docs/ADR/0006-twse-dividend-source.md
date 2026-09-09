# ADR 0006 — 以 TWSE 除權除息公告作為企業行動因子的來源

- **狀態**：已採納（經核定，但**目前無法在本環境驗證**）
- **日期**：2026-09-09
- **階段**：Phase 1

## 背景

企業行動的調整因子必須從某處來。既有 sector_rotation 專案的做法是
`repair_taiwan_price_discontinuities`：偵測單日跳動超過 40%，然後回頭
乘算修改歷史價格。

那個做法有兩個問題：

1. **它是猜測。** 「跳動超過 40%」能偵測到「發生了某件事」，但推不出
   「是哪一種企業行動、確切比例多少」。現金股利、分割、減資都會造成跳動。
2. **它修改歷史。** SPEC 3.2 明文禁止，因為新的企業行動會讓歷史調整價全部改變。

## 決策

新增外部資料源：TWSE 除權除息計算結果表
（`https://openapi.twse.com.tw/v1/exchangeReport/TWT49U`）。

該表提供每筆除權息的「除權息前收盤價」與「除權息參考價」，
兩者相除即為調整因子。這是交易所公告的**事實**，不是估計，
因此 `ActionSource.OFFICIAL`、`confidence = 1.0`。

依 CLAUDE.md「詢問時機」第 1 條，新增外部資料源已先徵得核定。

既有的「跳動偵測」邏輯不丟棄，但改變用途：從「修改歷史的依據」
降級為**資料品質檢查**（`quality.check_sanity`）。台股有 10% 漲跌幅限制，
所以更大的跳動必然是未登錄的企業行動——這時該做的是發告警要求補資料，
而不是自己動手改價格。

## 目前的阻礙

**本執行環境的網路政策擋住了 TWSE。** proxy 明確回報：

```
host: openapi.twse.com.tw:443
kind: connect_rejected
detail: gateway answered 403 to CONNECT (policy denial or upstream failure)
```

因此 `fetch_twse_dividends()` 已實作但無法在此環境執行。

**這不影響架構的正確性，理由是抓取與解析刻意分離**：

- `fetch_*` 只負責把 bytes 拿回來並落地到 raw landing zone；
- `parse_twse_dividends()` 是吃 bytes 的純函式，用真實格式的樣本完整測試；
- 網路放行後，不需要改任何解析邏輯就能運作。

在網路放行前，因子表可以由 `ActionSource.INFERRED`（跳動反推，信心 < 1.0）
或 `ActionSource.LEGACY`（繼承自既有已調整價）填充，
下游可以依 `min_confidence` 決定要不要採用。

## 後果

- 因子的信心值成為第一級資料：`AdjustedPrice.min_confidence` 會把
  「這段調整有多可信」一路帶到查詢結果。
- 抓取與解析分離讓解析器可以改版後重跑舊 bytes，這是 raw landing zone
  存在的主要理由（見 `ingestion/landing.py`）。

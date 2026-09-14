# ADR 0004 — 套件佈局：`src/trading_intel/` 而非 `src/core/`

- **狀態**：已採納
- **日期**：2026-09-08
- **階段**：Phase 0

## 背景

SPEC 第 9 節列出的目錄結構把各層直接放在 `src/` 底下：

```
src/
├── core/
├── ingestion/
├── features/
├── models/
├── risk/
└── ...
```

照字面實作，會讓 `core`、`models`、`risk`、`features`、`backtest`
成為五個**頂層** import 名稱。

## 決策

保留 SPEC 的分層與名稱，但全部收在一個套件命名空間底下：

```
src/trading_intel/core/
src/trading_intel/ingestion/
...
```

import 寫成 `from trading_intel.core.clock import utc_now`，
與 SPEC 表格的對應是逐項一致的，只是多一層前綴。

## 理由

- **頂層通用名稱會撞名。** `models`、`core`、`features` 都是極常見的套件名。
  一旦有任何相依套件（或未來的 plugin）用了同名頂層模組，
  import 解析結果會取決於 `sys.path` 的順序——那是最難查的一類 bug。
- **SPEC 2 要求把 `taiwan-fund-flow-lab` 的 `src/sector_rotation`
  以套件相依方式引入。** 兩個專案同時安裝在一個環境裡時，
  沒有命名空間的頂層模組更容易互相干擾。
- **可安裝性。** 專案要能 `pip install` 進其他環境（Phase 5 的儀表板服務、
  排程 worker）。散落的頂層模組會把整個 `src/` 的內容倒進 site-packages。
- **成本極低。** 差別只有 import 敘述多一個前綴，
  目錄結構、分層邊界與單向相依規則全部原封不動。

## 後果

- SPEC 第 9 節的表格與實際路徑差一層前綴。已在該節加註實作說明指向本 ADR，
  避免日後有人以為是實作偏離了規格。
- 若之後決定改回扁平佈局，成本是一次全域改名，不影響任何架構決策。

## 已否決的替代方案

- **完全照 SPEC 的扁平佈局。** 語意上與本決策相同，但承擔了上述撞名風險，
  換到的好處只有 import 敘述短一點。
- **改用 flat layout（不要 `src/`）。** 會讓測試意外 import 到工作目錄下的
  原始碼而非安裝版本，掩蓋掉打包設定的錯誤。

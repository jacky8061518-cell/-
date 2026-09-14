# ADR 0003 — 全面禁用直接取時間，改為注入時鐘

- **狀態**：已採納
- **日期**：2026-09-08
- **階段**：Phase 0（`core/clock.py`，由 ruff 與 `tests/guard/` 雙重執行）

## 背景

`datetime.now()` 是交易系統裡最危險的函式，正因為它看起來人畜無害。

特徵計算裡的一次 `datetime.now()`，就讓該特徵無法測試（每次執行結果都不同）、
無法重現（同一批事件重放會產出不同輸出），並且在回測中不安全
（該計算看到的是真實的現在，系統其他部分卻以為現在是 2020 年）。
失敗是無聲的：不會拋錯，數字只是錯了，而且往樂觀的方向錯。

`datetime.utcnow()` 更糟——它回傳一個看起來像 UTC 的 *naive* datetime，
拿去和 aware 時間戳比較會在執行期拋錯，事後補上時區則會產出一個
安靜地差了本地時差的值。

## 決策

1. `trading_intel.core.clock.utc_now()` 是唯一被認可的取時間方式，
   永遠回傳 tz-aware 的 UTC datetime。
2. 作用中的時鐘放在 `ContextVar`，因此某個任務或測試安裝的模擬時鐘
   不會洩漏到另一個。
3. `datetime.now`、`datetime.utcnow`、`datetime.today` 與 `time.time`
   由 ruff 的 `flake8-tidy-imports` banned-api 規則在全專案禁用，
   唯一豁免是 `core/clock.py`。
4. `tests/guard/test_no_direct_time_access.py` 走訪每個原始檔的 AST，
   對同一組呼叫判定失敗。

## 理由

第 3 與第 4 條刻意重疊，因為它們的失效方式不同：

- ruff 的規則作用在 import graph 上。它抓得到常見寫法，
  而且在 commit 之前就在編輯器裡抓到。
- AST 守門抓得到透過別名繞過的呼叫——`import datetime as dt; dt.datetime.now()`
  ——那是以 import 為基礎的規則看不到的。

兩者單獨都不夠；合起來則讓這條規則不需要任何人記得就會成立。
這正是重點：這不是靠審查者執行的慣例，而是建置流程的一個性質。

選 `ContextVar` 而非模組層級全域變數是第二個刻意的決策。
全域時鐘在兩個測試平行執行、或某個 async 任務在回測中途被暫停之前都沒問題，
到那時某個 context 的模擬時間會安靜地變成另一個 context 的。
那種 bug 基本上無法診斷，所以一開始就選較便宜的結構。

`ensure_utc` 對 naive 值拋 `NaiveDatetimeError` 而不是假設它是 UTC，理由相同：
naive datetime 是一個我們答不出來的問題，而用猜的正是差 8 小時的 bug
進入正式環境的方式。

## 後果

**接受的代價**

- 每個需要時間的模組都相依於 `core.clock`。這是一個小而刻意的耦合。
- 第三方套件內部會呼叫 `datetime.now()`，我們攔不住。
  `no_network()` 與 `backtest_mode()` 藉由切斷資料路徑限制影響範圍，
  但會用牆上時間蓋自己紀錄的套件，超出本 ADR 能執行的範圍。
- 貢獻者會撞到這條禁令並短暫覺得煩。錯誤訊息裡寫明了替代方案，
  那已經是修正的大部分。

**支付這些代價換得的好處**

`backtest_mode(asof)` 可以用一個 context manager 把整個系統凍結在某個時點，
因為全系統只有一個地方讀時鐘。

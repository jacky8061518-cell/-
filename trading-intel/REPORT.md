# Phase 0 驗收報告

專案位置：`trading-intel/`（此 repo 根目錄已有另一個專案 `sector-rotation-research`，
為避免覆蓋既有的 `pyproject.toml` 與 `src/`，Phase 0 建成獨立子目錄。所有指令都在
`trading-intel/` 內執行。）

規模：`src/` 905 行、`tests/` 1543 行、177 個測試。

---

## 總覽

| # | 驗收條件 | 結果 |
|---|---|---|
| 1 | `make check` 全綠，core 覆蓋率 ≥ 95% | ✅ 通過（**100%**） |
| 2 | 植入 `datetime.now()`，ruff 與 guard 測試都攔下 | ✅ 兩道都攔下 |
| 3 | `backtest_mode` 內連網拋 `NetworkAccessDenied` | ✅ 通過，訊息可讀 |
| 4 | `make_signal_id` 跨程序結果一致 | ✅ 兩次獨立程序輸出相同 |
| 5 | 修改 `Signal` 欄位拋 `ValidationError` | ✅ 通過 |
| 6 | `load_settings("dev")` 成功，缺欄位錯誤指出欄位路徑 | ✅ 通過 |
| 7 | `SimulatedClock(2020-03-19)` 內 `utc_now()` 回 2020 年 | ✅ 通過 |

---

## 1. `make check` 全綠，覆蓋率 ≥ 95%

```
$ make check
uv run ruff format .
25 files left unchanged

uv run ruff check .
All checks passed!

uv run mypy src tests
Success: no issues found in 26 source files

uv run pytest -q --cov=src/trading_intel --cov-report=term-missing
177 passed

Name                                 Stmts   Miss Branch BrPart  Cover   Missing
--------------------------------------------------------------------------------
src/trading_intel/__init__.py            1      0      0      0   100%
src/trading_intel/core/__init__.py       0      0      0      0   100%
src/trading_intel/core/clock.py         61      0      8      0   100%
src/trading_intel/core/enums.py         39      0      0      0   100%
src/trading_intel/core/errors.py        24      0      2      0   100%
src/trading_intel/core/events.py        23      0      0      0   100%
src/trading_intel/core/ids.py           27      0      2      0   100%
src/trading_intel/core/logging.py       18      0      0      0   100%
src/trading_intel/core/sandbox.py       32      0      6      0   100%
src/trading_intel/core/settings.py      82      0     12      0   100%
src/trading_intel/core/types.py        134      0     20      0   100%
--------------------------------------------------------------------------------
TOTAL                                  441      0     50      0   100%

$ echo $?
0
```

`mypy` 為 `strict = true` 加 `pydantic.mypy` plugin，`src` 與 `tests` 都納入檢查。

## 2. 植入 `datetime.now()`，兩道閘門都攔下

在 `src/trading_intel/core/ids.py`（非 clock 檔案）暫時加入：

```python
def _planted_violation() -> datetime:
    return datetime.now()
```

**ruff：**

```
$ uv run ruff check src/trading_intel/core/ids.py
TID251 `datetime.datetime.now` is banned: 禁止直接取現在時間，改用 trading_intel.core.clock.utc_now
  --> src/trading_intel/core/ids.py:59:12
   |
58 | def _planted_violation() -> datetime:
59 |     return datetime.now()
   |            ^^^^^^^^^^^^

DTZ005 `datetime.datetime.now()` called without a `tz` argument
Found 2 errors.
$ echo $?
1
```

**guard 測試：**

```
$ uv run pytest tests/guard/test_no_direct_time_access.py -q
E       AssertionError: direct wall-clock access is banned outside core/clock.py;
        use trading_intel.core.clock.utc_now instead:
E           /home/user/-/trading-intel/src/trading_intel/core/ids.py:59: datetime.now()
FAILED tests/guard/test_no_direct_time_access.py::test_no_direct_time_access_outside_clock
```

錯誤訊息含檔名與行號。植入的程式碼已移除。

兩道規則刻意重疊：ruff 走 import graph，抓不到 `import datetime as dt; dt.datetime.now()`
這種別名寫法；AST guard 走語法樹，補上這個缺口。測試檔本身另有
`test_guard_detects_a_planted_violation`，確保 guard 真的會 fail 而不是一個永遠綠燈的空測試。

## 3. `backtest_mode` 內連網

```
$ uv run python -c "..."
NetworkAccessDenied: network access via socket.create_connection is disabled inside this sandbox
  (call="args=(('example.com', 443),) kwargs={'timeout': 1}", target='socket.create_connection')
```

例外的 `context` 帶結構化欄位（`target`、`call`），可直接進日誌。
`tests/guard/test_no_network_in_backtest.py` 另外驗證：離開 context 後
`socket.socket` / `create_connection` / `ssl.wrap_socket` 都還原為原物件、block 內拋例外
也會還原、巢狀使用時內層離開不會提早解除外層封鎖。

## 4. `make_signal_id` 跨程序決定性

```
$ for i in 1 2; do uv run python -c "...make_signal_id(...)"; done
run1: c7e518d87a27b29f
run2: c7e518d87a27b29f
```

`tests/unit/test_ids.py` 以硬編碼期望值比對（`TW:2330`、`0361101ba4934e81`、
`c7e518d87a27b29f`），並實際 spawn 兩個子程序確認 `PYTHONHASHSEED` 隨機化不會外洩到 id。
`make_evidence_id` 的 source 與 payload 之間插入 `\x1f` 分隔，避免 `("ab", b"c")` 與
`("a", b"bc")` 碰撞——這點也有測試。

## 5. `Signal` 不可變

```
$ uv run python -c "...s.score = 0.9..."
ValidationError: 1 validation error for Signal
score
  Instance is frozen [type=frozen_instance, input_value=0.9, input_type=float]
```

`tests/guard/test_models_are_frozen.py` 用反射列出 `core.types` 與 `core.events` 中所有
`BaseModel` 子類（目前 12 個），逐一斷言 `frozen is True` 且 `extra == "forbid"`，
未來新增模型漏掉設定會直接失敗。該檔另有一個「至少要找到 10 個模型」的斷言，
避免反射找不到東西而假性通過。

## 6. 設定載入

```
$ uv run python -c "load_settings('dev')"
env= dev | display_tz= Asia/Taipei
data.max_staleness_minutes= 1440
risk.max_position_weight= 0.05
```

缺欄位（移除 `risk.drawdown_flatten`）：

```
ConfigError: invalid configuration
  fields = risk/drawdown_flatten
```

疊加順序 `configs/base.yaml` → `configs/{env}.yaml` → 環境變數。
pydantic-settings 預設 init kwargs 優先於環境變數，與需求相反，因此覆寫
`settings_customise_sources` 把 `env_settings` 排到 `init_settings` 之前，
`test_env_vars_outrank_yaml` 驗證此行為。

## 7. 模擬時鐘

```
$ uv run python -c "with use_clock(SimulatedClock(datetime(2020,3,19,tzinfo=UTC))): ..."
utc_now() = 2020-03-19T00:00:00+00:00 | year = 2020
outside the context, year = 2026
```

---

## 需要你決定的事項

1. **風控數字全是佔位值。** `configs/*.yaml` 中 `risk:` 與 `costs:` 底下每個數字都標了
   `# TODO: placeholder, needs sign-off`，只挑了結構上合法的值，沒有任何一個是風控判斷。
   `tw_transaction_tax` 0.003 與 `tw_daytrade_tax` 0.0015 是法定稅率，不是佔位值。

2. **`trading_day` 的邊界語意。** 目前定義為：以當地收盤時間切日，收盤（台股 13:30
   Asia/Taipei、美股 16:00 America/New_York）當下與之前屬於當日，之後屬於次一個日曆日。
   「盤後消息算次一交易日」是一個判斷，另一種合理讀法是「盤後仍歸屬當日」。
   假日行事曆不在 Phase 0 範圍，所以次日只是次一日曆日、不是次一個真正的交易日。
   若語意要改，改動點集中在 `core/clock.py` 的 `trading_day` 與 `MARKET_CLOSE`。

3. **`TemporalIntegrityError` 直接拋出，不包成 `ValidationError`。** pydantic 只會把
   `ValueError` / `AssertionError` 包進 `ValidationError`，自訂例外會原樣往外傳。
   規格寫的是「否則拋 `TemporalIntegrityError`」，所以維持原樣拋出——好處是呼叫端可以
   精準 catch 時間錯誤，代價是時間錯誤與其他欄位驗證錯誤的例外型別不同。若希望統一，
   讓 `TemporalIntegrityError` 也繼承 `ValueError` 即可。

## 與規格的其他差異

- **專案放在 `trading-intel/` 子目錄**，理由如開頭所述。
- **`AgentOutput` / `EventEnvelope` 用 PEP 695 泛型語法**（`class AgentOutput[T: BaseModel]`）
  而非 `Generic[T]`，因為 ruff `UP046` 在 `target-version = py312` 下會擋下後者。
  語意相同，`T = TypeVar("T", bound=BaseModel)` 仍保留在 `core/types.py` 供下游引用。
- **`Bar` 的 OHLC 驗證只保留一條規則。** 規格另列的 `high >= low` 在數學上被
  `low <= min(open, close) <= max(open, close) <= high` 完全涵蓋，是不可達分支，故移除。
- **新增 `configs/staging.yaml`**，因為 `Settings.env` 的 `Literal` 含 `staging`。
- **新增 `TI_CONFIG_DIR` 環境變數**覆寫設定目錄位置，供測試與安裝後部署使用。
- **`Instrument` 加了 `currency` 的 ISO 4217 格式驗證與 delisted/listed 先後檢查**，
  `FeatureVector` 加了「每個 feature 都必須有版本號」的驗證（規格註解要求「缺一即拒收」）。
- **三個 ruff 抑制**，各有理由：`errors.py` 的 `N818`（`NetworkAccessDenied` 名稱由規格指定）、
  `settings.py` 的 `ARG003`（`settings_customise_sources` 簽章由 pydantic-settings 決定）、
  `test_no_network_in_backtest.py` 的 `SIM117`（巢狀 `with` 正是待測行為）。

## 下一步

`docs/ADR/` 已補上三份決策紀錄：

- `0001-event-driven-architecture.md`
- `0002-bitemporal-storage.md`
- `0003-clock-injection.md`

Phase 1 開始前，建議先核定上面第 1、2 項。

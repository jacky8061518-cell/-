# trading-intel

自主式多代理人交易研究系統。完整規格見 [`docs/SPEC.md`](docs/SPEC.md)，
開發守則見 [`CLAUDE.md`](CLAUDE.md)。

**目前進度：Phase 0（骨架與型別系統）已完成。** 尚未實作任何資料源、模型或 agent。

Phase 0 只做三件之後改不動的事：

1. **時間紀律** — `core.clock.utc_now` 是全專案取得現在時間的唯一入口。
   直接呼叫 `datetime.now` / `utcnow` / `today` / `time.time` 會被 ruff 與
   AST 守門測試兩道閘門攔下。
2. **型別契約** — `core.types` 中每個模型都是 frozen、`extra="forbid"` 的
   Pydantic 模型，時間不變條件在建構當下就強制執行。
3. **防呆閘門** — `core.sandbox.backtest_mode` 同時凍結時鐘與切斷網路，
   讓回測既不能前視也不能對外連線。

## 快速開始

```bash
make install   # uv sync --all-extras
make check     # fmt -> lint -> type -> test
```

## 目錄結構

對應 SPEC 第 9 節。`src/` 採 src layout 加套件命名空間，理由見
[`docs/ADR/0004-package-layout.md`](docs/ADR/0004-package-layout.md)。

```
src/trading_intel/
  core/         型別、事件定義、設定、時間工具        ← Phase 0 已完成
  ingestion/    L0 各資料源 worker                    ← Phase 1
  normalize/    L1 標準化、企業行動、實體解析          ← Phase 1
  features/     L2 特徵，全部純函式                   ← Phase 2
  backtest/     事件驅動回測引擎、成本模型、驗證協定    ← Phase 2
  models/       訊號模型、regime 分類、融合            ← Phase 2 起
  agents/       L3 LLM 代理人層                       ← Phase 3
  portfolio/    組合建構、最佳化                       ← Phase 4
  risk/         三層風控                              ← Phase 4
  surface/      L5 儀表板 API、告警、報告生成          ← Phase 5
tests/
  guard/        會自我執行的規則（時間、網路、frozen）
  unit/         各模組行為
  integration/  跨模組整合                            ← Phase 1 起
  property/     hypothesis 性質測試
configs/        全部參數，環境分離
docs/ADR/       架構決策紀錄
docs/runbook/   故障排除手冊                          ← Phase 1 起
```

## core 模組一覽

| 模組 | 職責 |
|---|---|
| `clock.py` | 全專案唯一允許讀取現在時間的檔案 |
| `enums.py` | 封閉詞彙表（StrEnum / IntEnum） |
| `ids.py` | 決定性識別碼，重放必得同一組 id |
| `errors.py` | 例外階層，每個例外帶結構化 context |
| `types.py` | frozen 的 Pydantic 契約 |
| `events.py` | 事件信封與順序無關的內容雜湊 |
| `sandbox.py` | `no_network()` 與 `backtest_mode()` |
| `settings.py` | 分層 YAML 加環境變數設定 |
| `logging.py` | structlog 接線 |

## 驗收狀態

Phase 0 七項驗收條件的實際指令輸出見 [`REPORT.md`](REPORT.md)。

#!/usr/bin/env python
"""Run the research pipeline end to end and record the verdict.

    python scripts/research_pipeline.py [--factor momentum] [--holding 21]

Walks one hypothesis through the whole path the design calls for: point-in-time
backtest, cost stress, execution-delay stress, a randomised null, the validation
gate, and finally the registry, which records the result whether or not it is
flattering. A rejected model is registered too — knowing what failed, and why,
is what stops the same idea being re-tested every quarter.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_platform.control.registry import ModelRegistry, validation_record  # noqa: E402
from quant_platform.data.pit import TAIWAN_SOURCES, PointInTimeStore  # noqa: E402
from quant_platform.research.backtest import BacktestSpec, run_backtest  # noqa: E402
from quant_platform.research.costs import DEFAULT_COSTS  # noqa: E402
from quant_platform.research.factors import FACTORS, Factor  # noqa: E402
from quant_platform.research.validation import sharpe_ratio, validate  # noqa: E402

DATABASE = ROOT / "data" / "databases" / "tw"


def load_universe(min_market_cap: float) -> tuple[pd.DataFrame, list[str]]:
    prices = pd.read_parquet(DATABASE / "adjusted-prices.parquet")
    master = pd.read_csv(DATABASE / "security-master.csv")
    stocks = master[master["Asset type"] == "股票"]
    last = prices.ffill().iloc[-1]
    cap = pd.to_numeric(stocks["Issued shares"], errors="coerce") * stocks["Yahoo ticker"].map(last)
    tickers = stocks.loc[cap >= min_market_cap, "Yahoo ticker"]
    return prices, [t for t in tickers if t in prices.columns]


def null_sharpes(store, factor, spec, universe, draws: int) -> list[float]:
    """Sharpe ratios from the identical pipeline with the signal permuted.

    Anything the pipeline scores well without a real signal is leakage, and this
    is the only way to see it.
    """
    results = []
    for seed in range(draws):
        rng = np.random.default_rng(seed)

        def permuted(prices, _factor=factor, _rng=rng):
            scores = _factor.compute(prices)
            return pd.Series(_rng.permutation(scores.to_numpy()), index=scores.index)

        shuffled = Factor(
            name=f"{factor.name}_null{seed}",
            version=factor.version,
            claim="null",
            fails_when="null",
            compute=permuted,
            min_history=factor.min_history,
        )
        result = run_backtest(store, shuffled, spec, universe=universe, check_lookahead=False)
        results.append(sharpe_ratio(result.net_returns, result.periods_per_year))
    return [value for value in results if np.isfinite(value)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", default="momentum", choices=sorted(FACTORS))
    parser.add_argument("--holding", type=int, default=21, help="持有期間（交易日）")
    parser.add_argument("--quantile", type=float, default=0.1)
    parser.add_argument("--long-only", action="store_true", help="只做多，不做空")
    parser.add_argument("--min-cap", type=float, default=5e9, help="最小市值（元）")
    parser.add_argument("--null-draws", type=int, default=8)
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="多重檢定的嘗試次數。誠實填寫；填小會讓 Deflated Sharpe 過度樂觀。",
    )
    args = parser.parse_args()

    factor = FACTORS[args.factor]
    prices, universe = load_universe(args.min_cap)
    store = PointInTimeStore()
    store.register("prices", prices, TAIWAN_SOURCES["prices"])

    print(f"因子：{factor.id}")
    print(f"  假說：{factor.claim}")
    print(f"  預期失效條件：{factor.fails_when}")
    print(f"  universe：{len(universe)} 檔｜期間 {prices.index[0]:%Y-%m-%d} ~ {prices.index[-1]:%Y-%m-%d}\n")

    spec = BacktestSpec(
        factor_id=factor.id,
        holding_days=args.holding,
        quantile=args.quantile,
        long_short=not args.long_only,
    )

    print("執行回測…")
    base = run_backtest(store, factor, spec, universe=universe)
    if not base.summary():
        print("回測沒有產生任何交易期間，停止。")
        return 1

    print("執行成本壓力測試（×3）…")
    stressed = run_backtest(store, factor, spec, costs=DEFAULT_COSTS.stressed(3), universe=universe)
    print("執行延遲測試（+1 日）…")
    delayed = run_backtest(store, factor, replace(spec, signal_delay_days=1), universe=universe)
    print(f"執行隨機化對照（{args.null_draws} 次）…")
    nulls = null_sharpes(store, factor, spec, universe, args.null_draws)

    summary = base.summary()
    print("\n── 回測結果 " + "─" * 52)
    print(f"  期數 {summary['periods']}｜交易檔次 {summary['trades']}｜平均換手 {base.turnover.mean():.0%}")
    print(f"  毛報酬 {summary['gross_return_per_period']:+.3%}/期"
          f"｜成本 {summary['cost_per_period']:.3%}"
          f"｜淨報酬 {summary['net_return_per_period']:+.3%}/期")
    print(f"  年化 {summary['annual_return']:+.2%}｜Sharpe {summary['sharpe']:.2f}"
          f"｜最大回撤 {summary['max_drawdown']:.1%}")
    print(f"  比較基準（{summary['hurdle']}）｜0050 同期年化 {summary['benchmark_annual']:+.2%}")

    report = validate(base, stressed, delayed, nulls, trials=args.trials)
    print("\n── 驗證閘門 " + "─" * 52)
    print(report.to_frame().to_string(index=False))
    metrics = report.metrics
    print(f"\n  Sharpe {metrics['sharpe']:.2f}"
          f"｜95% bootstrap 區間 [{metrics['sharpe_ci_low']:.2f}, {metrics['sharpe_ci_high']:.2f}]"
          f"｜Deflated Sharpe {metrics['deflated_sharpe']:.3f}（trials={args.trials}）")

    print("\n── 逐年表現 " + "─" * 52)
    print(base.by_year().round(4).to_string(index=False))

    registry = ModelRegistry()
    model_id = f"{factor.id}_h{args.holding}_q{int(args.quantile*100)}"
    registry.register(
        model_id=model_id,
        factor_id=factor.id,
        spec=asdict(spec) | {"universe_size": len(universe), "min_market_cap": args.min_cap},
        validation=validation_record(report),
        notes=factor.claim,
    )

    print("\n── 判定 " + "─" * 56)
    print(f"  {report.verdict()}")
    print(f"  已登錄 registry：{model_id}（階段：研究中）")
    if not report.approved:
        print("  下一步不是調參數直到通過——那叫過度配適。要改的是假說本身。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

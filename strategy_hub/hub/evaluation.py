"""
hub/evaluation.py
依照「如何評估 AI 交易系統」的五個面向，對每個策略產出同一張成績單：

01 報酬率 / 夏普比率 / 最大回撤
02 換手率與交易成本
03 跨市場體制的穩定度
04 相較於基準線的決策品質
05 風控違規與營運故障（策略層；系統營運面另見 health.py）

夏普比率以無風險利率 0 計算（台灣定存約 1.5%，換算後所有策略 Sharpe 約低 0.1，
不影響策略間的相對排名）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hub.regimes import REGIMES

TRADING_DAYS = 252
CONCENTRATION_CAP = 0.05  # 單一個股權重上限（每 sleeve 20 格位 = 5%）


def _clean(x):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 4)
    return x


def cagr(r: pd.Series) -> float:
    if len(r) == 0:
        return float("nan")
    growth = float((1 + r).prod())
    years = len(r) / TRADING_DAYS
    return growth ** (1 / years) - 1 if growth > 0 else -1.0


def sharpe(r: pd.Series) -> float:
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def drawdown(r: pd.Series) -> pd.Series:
    eq = (1 + r).cumprod()
    return eq / eq.cummax() - 1


def performance_block(r: pd.Series) -> dict:
    dd = drawdown(r)
    mdd = float(dd.min())
    c = cagr(r)
    end = r.index[-1]
    def window(days):
        w = r.iloc[-days:]
        return float((1 + w).prod() - 1) if len(w) else float("nan")
    ytd = r[r.index.year == end.year]
    return {
        "cagr": c,
        "ann_vol": float(r.std() * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe(r),
        "max_drawdown": mdd,
        "calmar": c / abs(mdd) if mdd < 0 else float("nan"),
        "current_drawdown": float(dd.iloc[-1]),
        "ret_1m": window(21),
        "ret_3m": window(63),
        "ret_1y": window(252),
        "ret_ytd": float((1 + ytd).prod() - 1),
    }


def cost_block(gross: pd.Series, net: pd.Series, cost: pd.Series, turnover: pd.Series,
               gross_exposure: pd.Series, holdings: pd.Series) -> dict:
    return {
        "gross_cagr": cagr(gross),
        "net_cagr": cagr(net),
        "cost_drag_annual": float(cost.mean() * TRADING_DAYS),
        "turnover_annual": float(turnover.mean() * TRADING_DAYS),
        "avg_gross_exposure": float(gross_exposure.mean()),
        "avg_holdings": float(holdings.mean()),
    }


def regime_block(r: pd.Series, regimes: pd.Series) -> dict:
    """regimes 必須是「前一日收盤已知」的體制（呼叫端負責 shift(1)）。

    若用當天收盤算出的體制去切當天報酬，RISK_OFF 的定義本身就包含「當天大跌」，
    任何多頭策略在 RISK_OFF 都會機械性地很難看——那是循環論證，不是策略特性。
    """
    out = {}
    lab = regimes.reindex(r.index)
    for reg in REGIMES:
        rr = r[lab == reg]
        out[reg] = {
            "share_of_days": float(len(rr) / len(r)) if len(r) else 0.0,
            "mean_daily_return": float(rr.mean()) if len(rr) else float("nan"),
            "hit_rate": float((rr > 0).mean()) if len(rr) else float("nan"),
            # 幾乎沒有波動（例如整段都在現金、只剩換倉成本）時 Sharpe 會爆成無意義的大數，不顯示
            "sharpe": sharpe(rr) if len(rr) > 20 and rr.std() * np.sqrt(TRADING_DAYS) > 0.02 else float("nan"),
        }
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    out["yearly"] = {str(y): float(v) for y, v in yearly.items()}
    out["pct_positive_years"] = float((yearly > 0).mean()) if len(yearly) else float("nan")
    out["worst_year"] = float(yearly.min()) if len(yearly) else float("nan")
    return out


def relative_block(r: pd.Series, bench: pd.Series) -> dict:
    b = bench.reindex(r.index).fillna(0.0)
    active = r - b
    te = float(active.std() * np.sqrt(TRADING_DAYS))
    cov = np.cov(r, b)
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else float("nan")
    monthly = (1 + pd.DataFrame({"s": r, "b": b})).groupby(r.index.to_period("M")).prod() - 1
    return {
        "excess_cagr": cagr(r) - cagr(b),
        "tracking_error": te,
        "information_ratio": float(active.mean() * TRADING_DAYS / te) if te > 0 else float("nan"),
        "beta": beta,
        "monthly_hit_rate": float((monthly["s"] > monthly["b"]).mean()),
    }


def constraint_block(weights: pd.DataFrame | None, regimes: pd.Series,
                     short_bans: list[tuple[str, str]], is_long_only_regime_filtered: bool = False,
                     entries: pd.DataFrame | None = None, cap: float = CONCENTRATION_CAP,
                     decision_regimes: pd.Series | None = None) -> dict:
    if weights is None:
        return {}
    max_name = weights.abs().max(axis=1)
    out = {
        "cap": cap,
        "max_single_name_weight": float(max_name.max()),
        "days_concentration_breach": int((max_name > cap + 1e-9).sum()),
        "max_gross_exposure": float(weights.abs().sum(axis=1).max()),
    }
    shorts = (weights < 0).any(axis=1)
    in_ban = pd.Series(False, index=weights.index)
    for start, end in short_bans:
        in_ban |= (weights.index >= pd.Timestamp(start)) & (weights.index <= pd.Timestamp(end))
    out["days_short_during_ban"] = int((shorts & in_ban).sum())
    if is_long_only_regime_filtered and entries is not None:
        # 硬規則自我檢查：「做決策當天」是 RISK_OFF 就不得有任何新多單。必須永遠是 0。
        # 注意要用決策當下的體制，不是評估用的落後一日體制，否則會誤報。
        ref = decision_regimes if decision_regimes is not None else regimes
        risk_off = ref.reindex(entries.index) == "RISK_OFF"
        out["new_longs_during_risk_off"] = int(entries[risk_off].sum().sum())
    return out


def scorecard(name: str, label: str, net: pd.Series, *, gross=None, cost=None, turnover=None,
              gross_exposure=None, holdings=None, weights=None, regimes: pd.Series,
              bench: pd.Series, equal_weight: pd.Series, short_bans, entries=None,
              regime_filtered=False, kind="sleeve", cap: float = CONCENTRATION_CAP,
              extra: dict | None = None, decision_regimes: pd.Series | None = None) -> dict:
    card = {
        "id": name,
        "label": label,
        "kind": kind,
        "performance": performance_block(net),
        "regime": regime_block(net, regimes),
        "vs_0050": relative_block(net, bench),
        "vs_equal_weight": relative_block(net, equal_weight),
        "constraints": constraint_block(weights, regimes, short_bans, regime_filtered, entries, cap,
                                        decision_regimes),
    }
    if gross is not None:
        card["costs"] = cost_block(gross, net, cost, turnover, gross_exposure, holdings)
    if extra:
        card.update(extra)

    def walk(o):
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        return _clean(o)
    return walk(card)

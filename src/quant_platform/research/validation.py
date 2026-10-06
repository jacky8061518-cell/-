"""Validation gate. The part of the pipeline whose job is to say no.

A backtest produces a number. This module decides whether that number means
anything, and it is deliberately hostile: every check here exists because a
strategy once passed a naive review and then lost money for a reason the review
could have caught.

The gate answers five questions:

1. **Is the Sharpe real, or the best of many tries?** Testing enough variants
   guarantees a good-looking one. The deflated Sharpe ratio prices that in.
2. **How uncertain is it?** A point estimate with no interval is not a result.
3. **Does it survive worse costs?** Cost assumptions are the softest input.
4. **Does it survive acting later?** An edge that evaporates one day after the
   signal is an edge nobody can capture.
5. **Did it work in more than one year?** A single lucky regime is not evidence.

A strategy that fails any hard gate does not get promoted. There is no override
here on purpose: the place to argue is the gate's parameters, in review, not
the individual verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EULER_MASCHERONI = 0.5772156649015329


def normal_cdf(x: float) -> float:
    """Standard normal CDF via erf, avoiding a SciPy dependency for one function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0 < p < 1:
        raise ValueError("機率必須介於 0 與 1 之間")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    clean = returns.dropna()
    if len(clean) < 3:
        return float("nan")
    std = clean.std(ddof=1)
    if std <= 0:
        return float("nan")
    return float(clean.mean() / std * np.sqrt(periods_per_year))


def probabilistic_sharpe(returns: pd.Series, benchmark_sharpe: float = 0.0) -> float:
    """Probability the true Sharpe exceeds ``benchmark_sharpe``.

    Adjusts for skew and kurtosis, which matter: a strategy that makes small
    gains constantly and loses hugely once has a flattering naive Sharpe.
    """
    clean = returns.dropna()
    n = len(clean)
    if n < 10:
        return float("nan")
    std = clean.std(ddof=1)
    if std <= 0:
        return float("nan")
    sr = float(clean.mean() / std)
    skew = float(clean.skew())
    kurtosis = float(clean.kurtosis()) + 3.0
    denominator = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2
    if denominator <= 0:
        return float("nan")
    return normal_cdf((sr - benchmark_sharpe) * math.sqrt(n - 1) / math.sqrt(denominator))


def expected_max_sharpe(trials: int, sharpe_variance: float) -> float:
    """Expected best Sharpe from ``trials`` independent worthless strategies.

    This is the number an observed Sharpe has to beat to be evidence of
    anything. Search hard enough and a null strategy will look excellent.
    """
    if trials < 2 or sharpe_variance <= 0:
        return 0.0
    scale = math.sqrt(sharpe_variance)
    first = normal_ppf(1.0 - 1.0 / trials)
    second = normal_ppf(1.0 - 1.0 / (trials * math.e))
    return scale * ((1 - EULER_MASCHERONI) * first + EULER_MASCHERONI * second)


def deflated_sharpe(
    returns: pd.Series,
    trials: int,
    sharpe_variance: float | None = None,
) -> float:
    """Probability the strategy's true Sharpe is positive, after multiple testing."""
    clean = returns.dropna()
    if len(clean) < 10:
        return float("nan")
    if sharpe_variance is None:
        # Without the variance of the trial Sharpes, fall back to the sampling
        # variance of a single null Sharpe, which is the conservative choice.
        sharpe_variance = 1.0 / max(len(clean) - 1, 1)
    threshold = expected_max_sharpe(trials, sharpe_variance)
    return probabilistic_sharpe(clean, benchmark_sharpe=threshold)


def bootstrap_sharpe_interval(
    returns: pd.Series,
    periods_per_year: float,
    draws: int = 2000,
    confidence: float = 0.95,
    seed: int = 7,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the annualised Sharpe."""
    clean = returns.dropna().to_numpy()
    if len(clean) < 12:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(draws, len(clean)), replace=True)
    means = samples.mean(axis=1)
    stds = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(stds > 0, means / stds * np.sqrt(periods_per_year), np.nan)
    sharpes = sharpes[np.isfinite(sharpes)]
    if sharpes.size == 0:
        return float("nan"), float("nan")
    tail = (1 - confidence) / 2
    return float(np.quantile(sharpes, tail)), float(np.quantile(sharpes, 1 - tail))


@dataclass
class GateResult:
    """One check, its verdict, and the number behind it."""

    name: str
    passed: bool
    value: float
    threshold: str
    hard: bool
    note: str = ""


@dataclass
class ValidationReport:
    """Everything the registry needs to decide promotion."""

    factor_id: str
    gates: list[GateResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def hard_failures(self) -> list[GateResult]:
        return [gate for gate in self.gates if gate.hard and not gate.passed]

    @property
    def soft_failures(self) -> list[GateResult]:
        return [gate for gate in self.gates if not gate.hard and not gate.passed]

    @property
    def approved(self) -> bool:
        return not self.hard_failures

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "檢驗": g.name,
                    "結果": "通過" if g.passed else "未通過",
                    "數值": round(g.value, 4) if np.isfinite(g.value) else None,
                    "門檻": g.threshold,
                    "類型": "硬性" if g.hard else "警示",
                    "說明": g.note,
                }
                for g in self.gates
            ]
        )

    def verdict(self) -> str:
        if self.approved and not self.soft_failures:
            return "通過所有檢驗，可進入 shadow 階段"
        if self.approved:
            return f"通過硬性門檻，但有 {len(self.soft_failures)} 項警示，需人工複核"
        reasons = "；".join(g.name for g in self.hard_failures)
        return f"未通過硬性門檻（{reasons}），不得晉級"


def validate(
    base_result,
    stressed_result,
    delayed_result,
    shuffled_sharpes: list[float],
    trials: int,
    min_trades: int = 200,
    min_periods: int = 60,
) -> ValidationReport:
    """Run every gate against a family of related backtests.

    The caller supplies the variants rather than this module running them, so a
    reviewer can see exactly which experiments the verdict rests on.
    """
    report = ValidationReport(factor_id=base_result.factor_id)
    summary = base_result.summary()
    if not summary:
        report.gates.append(
            GateResult("回測有結果", False, 0.0, "至少一期", True, "回測沒有產生任何交易期間")
        )
        return report

    scale = base_result.periods_per_year
    net = base_result.net_returns
    sharpe = sharpe_ratio(net, scale)
    low, high = bootstrap_sharpe_interval(net, scale)
    shuffle_variance = float(np.var(shuffled_sharpes, ddof=1)) if len(shuffled_sharpes) > 2 else None
    dsr = deflated_sharpe(net, trials=trials, sharpe_variance=shuffle_variance)
    yearly = base_result.by_year()
    profitable_years = int((yearly["total_return"] > 0).sum()) if not yearly.empty else 0
    total_years = len(yearly)

    report.metrics = {
        "sharpe": sharpe,
        "sharpe_ci_low": low,
        "sharpe_ci_high": high,
        "deflated_sharpe": dsr,
        "annual_return": summary["annual_return"],
        "max_drawdown": summary["max_drawdown"],
        "cost_share_of_gross": summary["cost_share_of_gross"],
        "trades": summary["trades"],
        "periods": summary["periods"],
    }

    g = report.gates.append
    g(GateResult("樣本量：交易檔次", summary["trades"] >= min_trades, summary["trades"],
                 f"≥ {min_trades}", True, "交易次數過少時，統計上什麼都證明不了"))
    g(GateResult("樣本量：期數", summary["periods"] >= min_periods, summary["periods"],
                 f"≥ {min_periods}", True, "獨立觀察期數不足以估計 Sharpe"))
    g(GateResult("扣成本後為正", summary["net_return_per_period"] > 0,
                 summary["net_return_per_period"], "> 0", True,
                 "毛報酬為正但淨報酬為負的策略，是在為券商工作"))
    g(GateResult("Deflated Sharpe", np.isfinite(dsr) and dsr > 0.95, dsr, "> 0.95", True,
                 f"已針對 {trials} 次嘗試做多重檢定修正"))
    g(GateResult("Sharpe 信賴區間下界", np.isfinite(low) and low > 0, low, "> 0", True,
                 "bootstrap 95% 區間下界仍為正，才算穩健"))

    stressed_summary = stressed_result.summary() if stressed_result is not None else {}
    stress_ok = bool(stressed_summary) and stressed_summary["net_return_per_period"] > 0
    g(GateResult("成本壓力 ×3", stress_ok,
                 stressed_summary.get("net_return_per_period", float("nan")), "> 0", True,
                 "成本假設是最軟的輸入，三倍壓力仍為正才可信"))

    delayed_summary = delayed_result.summary() if delayed_result is not None else {}
    decay = (
        1 - delayed_summary["net_return_per_period"] / summary["net_return_per_period"]
        if delayed_summary and summary["net_return_per_period"] != 0
        else float("nan")
    )
    g(GateResult("延遲一日衰減", np.isfinite(decay) and decay < 0.5, decay, "< 50%", True,
                 "延後一天執行就消失的 edge，實務上抓不到"))

    max_shuffle = max(shuffled_sharpes) if shuffled_sharpes else float("nan")
    leak_ok = not np.isfinite(max_shuffle) or sharpe > max_shuffle
    g(GateResult("隨機化對照", leak_ok, max_shuffle, f"< 實際 Sharpe {sharpe:.2f}", True,
                 "打亂訊號後若仍有績效，代表流程有資料洩漏"))

    g(GateResult("跨年份穩定性", total_years > 0 and profitable_years / total_years >= 0.5,
                 profitable_years / total_years if total_years else float("nan"),
                 "≥ 50% 年份為正", False, f"{profitable_years}/{total_years} 個年份為正報酬"))
    g(GateResult("成本佔毛報酬", summary["cost_share_of_gross"] < 0.5,
                 summary["cost_share_of_gross"], "< 50%", False,
                 "成本吃掉大半毛報酬時，任何成本估計誤差都會翻轉結論"))
    g(GateResult("最大回撤", summary["max_drawdown"] > -0.25, summary["max_drawdown"],
                 "> -25%", False, "回撤過深時，實際上撐不到策略回本"))

    return report

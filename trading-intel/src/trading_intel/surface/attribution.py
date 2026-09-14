"""損益歸因（SPEC 4.2）。

**這是純程式碼，不使用 LLM。** SPEC 原本規劃為 agent，改為確定性模組的
理由很直接：拆解損益來源是算術，不是語意任務。alpha、beta、風格因子、
交易成本、時機——每一項都是可以用回歸與差分算出來的數字，讓 LLM 做這件事
只會多一層不必要的不確定性。

敘述文字由報告模板生成，需要時再單次呼叫 LLM 潤飾——這一層不在本模組，
是 surface 層拿到歸因結果之後才決定要不要做的事。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class AttributionResult:
    """一段期間的損益拆解。五項刻意互斥且可加總，方便對帳。"""

    total_return: float
    #: 貝他：由基準報酬解釋的部分。
    beta_return: float
    #: 阿爾法：扣掉貝他之後，扣掉風格因子之前的殘差。
    alpha_return: float
    #: 風格因子：動量、價值、規模等因子暴露解釋的部分。
    style_return: float
    #: 交易成本：手續費、證交稅、滑價、市場衝擊的總和，恆為負值或零。
    cost_drag: float
    #: 時機：實際執行與理想執行（例如以收盤價成交）之間的差異。
    timing_return: float

    def __post_init__(self) -> None:
        components = (
            self.beta_return
            + self.alpha_return
            + self.style_return
            + self.cost_drag
            + self.timing_return
        )
        if abs(components - self.total_return) > 1e-9:
            msg = (
                f"歸因分項總和 {components} 與總報酬 {self.total_return} 不一致，"
                "歸因必須恆等式成立，否則某一項算錯了"
            )
            raise ValueError(msg)


def attribute_returns(
    portfolio_returns: npt.ArrayLike,
    benchmark_returns: npt.ArrayLike,
    style_factor_returns: npt.ArrayLike | None,
    style_exposures: npt.ArrayLike | None,
    cost_returns: npt.ArrayLike,
    ideal_returns: npt.ArrayLike,
) -> AttributionResult:
    """把一段期間的組合報酬拆成五個可加總的分項。

    ``style_factor_returns`` 為 (期數 × 因子數)，``style_exposures`` 為
    對應的組合暴露（純量或每期一組）。兩者皆為 None 時風格項為零，
    全部殘差歸入 alpha——這是誠實的做法：沒有做因子分解就不假裝有。
    """
    portfolio = np.asarray(portfolio_returns, dtype=np.float64)
    benchmark = np.asarray(benchmark_returns, dtype=np.float64)
    costs = np.asarray(cost_returns, dtype=np.float64)
    ideal = np.asarray(ideal_returns, dtype=np.float64)
    if not (portfolio.size == benchmark.size == costs.size == ideal.size):
        msg = "各序列長度必須一致"
        raise ValueError(msg)

    total_return = float(np.prod(1 + portfolio) - 1)
    beta = _estimate_beta(portfolio, benchmark)
    beta_return = float(beta * (np.prod(1 + benchmark) - 1))

    style_return = 0.0
    if style_factor_returns is not None and style_exposures is not None:
        factors = np.asarray(style_factor_returns, dtype=np.float64)
        exposures = np.asarray(style_exposures, dtype=np.float64)
        if factors.ndim == 1:
            factors = factors.reshape(-1, 1)
        style_return = float(np.sum(exposures.mean(axis=0) * factors.sum(axis=0)))

    cost_drag = float(np.sum(costs))
    if cost_drag > 0:
        msg = "交易成本項必須為負值或零，正值代表成本被算成了收益"
        raise ValueError(msg)

    timing_return = float(np.sum(portfolio - ideal))

    alpha_return = total_return - beta_return - style_return - cost_drag - timing_return

    return AttributionResult(
        total_return=total_return,
        beta_return=beta_return,
        alpha_return=alpha_return,
        style_return=style_return,
        cost_drag=cost_drag,
        timing_return=timing_return,
    )


def _estimate_beta(portfolio: FloatArray, benchmark: FloatArray) -> float:
    """以簡單迴歸估計貝他。基準無變異時貝他視為零（無從估計）。"""
    if portfolio.size < 2:
        return 0.0
    variance = float(np.var(benchmark, ddof=1))
    if variance == 0:
        return 0.0
    covariance = float(np.cov(portfolio, benchmark, ddof=1)[0, 1])
    return covariance / variance


@dataclass(frozen=True)
class HitRateSummary:
    """訊號命中率。晨報與盤後檢討共用（SPEC 8.1）。"""

    total_signals: int
    correct_direction: int
    #: 訊號方向與期末實際報酬方向一致的比例。
    hit_rate: float
    average_return_when_correct: float
    average_return_when_wrong: float


def summarize_hit_rate(
    predicted_directions: npt.ArrayLike,
    realized_returns: npt.ArrayLike,
) -> HitRateSummary:
    """統計一批訊號的命中率，供盤後檢討使用。"""
    directions = np.asarray(predicted_directions, dtype=np.float64)
    returns = np.asarray(realized_returns, dtype=np.float64)
    if directions.size != returns.size:
        msg = "方向與報酬序列長度必須一致"
        raise ValueError(msg)
    if directions.size == 0:
        return HitRateSummary(0, 0, float("nan"), float("nan"), float("nan"))

    correct = (np.sign(directions) == np.sign(returns)) & (directions != 0)
    total = int(np.sum(directions != 0))
    correct_count = int(np.sum(correct))

    correct_returns = returns[correct]
    wrong_mask = (directions != 0) & ~correct
    wrong_returns = returns[wrong_mask]

    return HitRateSummary(
        total_signals=total,
        correct_direction=correct_count,
        hit_rate=correct_count / total if total > 0 else float("nan"),
        average_return_when_correct=(
            float(np.mean(correct_returns)) if correct_returns.size else float("nan")
        ),
        average_return_when_wrong=(
            float(np.mean(wrong_returns)) if wrong_returns.size else float("nan")
        ),
    )


@dataclass(frozen=True)
class DailyAttributionReport:
    """一天的歸因報告，含當日日期以便排序與存檔。"""

    trade_date: date
    attribution: AttributionResult
    hit_rate: HitRateSummary

    def to_display_text(self) -> str:
        a = self.attribution
        lines = [
            f"【盤後歸因】{self.trade_date.isoformat()}",
            f"總報酬 {a.total_return:+.2%}"
            f"　貝他 {a.beta_return:+.2%}"
            f"　阿爾法 {a.alpha_return:+.2%}"
            f"　風格 {a.style_return:+.2%}"
            f"　成本 {a.cost_drag:+.2%}"
            f"　時機 {a.timing_return:+.2%}",
        ]
        if self.hit_rate.total_signals > 0:
            lines.append(
                f"命中率 {self.hit_rate.hit_rate:.0%}"
                f"（{self.hit_rate.correct_direction}/{self.hit_rate.total_signals}）"
            )
        return "\n".join(lines)

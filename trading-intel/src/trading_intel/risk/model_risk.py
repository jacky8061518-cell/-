"""模型風險：訊號衰退、追蹤誤差、特徵漂移、agent 一致性（SPEC 7.3）。

這一層問的是一個不同的問題：不是「這筆交易安全嗎」，而是
**「我們的模型還有效嗎」**。

前者的失效是突然的、看得見的；後者是漸進的、看不見的。一個訊號不會某天
宣布自己失效，它只會慢慢變差，而回測的績效數字還掛在牆上。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from trading_intel.core.settings import ModelRiskSettings


class SignalHealth(StrEnum):
    HEALTHY = "HEALTHY"
    #: 觀察中：IC 下滑但尚未達隔離門檻。
    WATCH = "WATCH"
    #: 已隔離：不再進入下單路徑，但繼續記錄以供研究。
    QUARANTINED = "QUARANTINED"


def information_coefficient(
    predictions: npt.ArrayLike,
    realized: npt.ArrayLike,
) -> float:
    """資訊係數：預測值與實現報酬的橫斷面相關性。

    用 Spearman 等級相關而非 Pearson：我們在意的是排序對不對，
    不是預測值的絕對大小。一個系統性高估但排序正確的訊號仍然有用。
    """
    from scipy import stats

    x = np.asarray(predictions, dtype=np.float64).ravel()
    y = np.asarray(realized, dtype=np.float64).ravel()
    if x.size != y.size:
        msg = "預測與實現的長度不一致"
        raise ValueError(msg)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    result = stats.spearmanr(x[mask], y[mask])
    value = float(result.statistic)
    return value if math.isfinite(value) else float("nan")


@dataclass
class SignalMonitor:
    """追蹤單一訊號上線後的表現（SPEC 7.3）。

    核心比較的不是「IC 是不是正的」，而是**「live 的 IC 與回測期的 IC 差多少」**。
    一個回測 IC 0.08、上線後 0.02 的訊號，即使 0.02 仍為正，
    也代表有東西不對——可能是實作有誤，也可能是市場已經變了。
    """

    signal_name: str
    backtest_ic: float
    settings: ModelRiskSettings
    _rolling_ic: list[float] = field(default_factory=list)
    _consecutive_below: int = 0
    _health: SignalHealth = SignalHealth.HEALTHY

    @property
    def health(self) -> SignalHealth:
        return self._health

    @property
    def observations(self) -> int:
        return len(self._rolling_ic)

    def record(self, period_ic: float) -> SignalHealth:
        """記錄一期的 IC 並更新健康狀態。"""
        self._rolling_ic.append(period_ic)
        if math.isnan(period_ic) or period_ic < self.settings.min_rolling_ic:
            self._consecutive_below += 1
        else:
            self._consecutive_below = 0

        if self._consecutive_below >= self.settings.ic_decay_periods:
            self._health = SignalHealth.QUARANTINED
        elif self._consecutive_below > 0:
            self._health = SignalHealth.WATCH
        else:
            self._health = SignalHealth.HEALTHY
        return self._health

    def live_vs_backtest_gap(self) -> float:
        """live IC 與回測 IC 的差距。正值代表 live 表現較差。"""
        if not self._rolling_ic:
            return float("nan")
        finite = [item for item in self._rolling_ic if math.isfinite(item)]
        if not finite:
            return float("nan")
        return self.backtest_ic - float(np.mean(finite))

    def tracking_error_breached(self) -> bool:
        """追蹤誤差超標代表實作有誤或市場已變，應觸發調查（SPEC 7.3）。"""
        gap = self.live_vs_backtest_gap()
        if not math.isfinite(gap):
            return False
        return abs(gap) > self.settings.max_live_backtest_tracking_error

    @property
    def tradable(self) -> bool:
        """被隔離的訊號不得進入下單路徑。"""
        return self._health is not SignalHealth.QUARANTINED


def agent_consistency_acceptable(
    scores: Sequence[float],
    settings: ModelRiskSettings,
) -> bool:
    """agent 一致性監控（SPEC 7.3）。

    同一輸入重複問三次，答案分歧度過高代表該任務不適合 LLM，
    應退回確定性做法。這裡檢查的是一批一致性分數的平均。
    """
    finite = [item for item in scores if math.isfinite(item)]
    if not finite:
        return False
    return float(np.mean(finite)) >= settings.min_agent_consistency

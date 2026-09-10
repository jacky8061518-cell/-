"""驗證協定：判斷一個策略的績效是真的，還是試出來的。

SPEC 5.3：「這一節是整個系統可信度的核心，不得簡化。」

核心問題只有一個：**你試了幾次？**

試一百組參數，總有幾組看起來很好。那不是因為它們有效，是因為一百個隨機數裡
本來就有最大值。這個現象叫多重檢定，它是量化研究裡最常見也最昂貴的錯誤——
因為它產生的假訊號在回測中看起來完美無缺。

本模組提供四道防線：
1. ``purged_walk_forward`` 切分訓練與測試，中間留 embargo，避免標籤重疊洩漏；
2. ``combinatorial_purged_cv`` 產生多條績效路徑，而非單一回測曲線；
3. ``deflated_sharpe_ratio`` 依試驗次數調整 Sharpe；
4. ``probability_of_backtest_overfitting`` 直接估計過度配適的機率。
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Final

import numpy as np
import numpy.typing as npt

TRADING_DAYS_PER_YEAR: Final = 252

#: Euler-Mascheroni 常數，用於期望最大 Sharpe 的推導。
_EULER_GAMMA: Final = 0.5772156649015329

#: 判定「實質無波動」的相對與絕對門檻。見 ``sharpe_ratio`` 的說明。
_RELATIVE_ZERO: Final = 1e-12
_ABSOLUTE_ZERO: Final = 1e-15

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Split:
    """一次訓練／測試切分。索引皆為原序列的位置。"""

    train: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        overlap = set(self.train) & set(self.test)
        if overlap:
            msg = f"訓練與測試集重疊了 {len(overlap)} 個索引"
            raise ValueError(msg)


def purged_walk_forward(
    n_samples: int,
    *,
    n_splits: int = 5,
    embargo: int = 0,
) -> list[Split]:
    """Purged walk-forward 切分，訓練集永遠在測試集之前。

    ``embargo`` 是訓練與測試之間留白的樣本數，應設為**最長持有期**。

    為什麼需要留白：若一個訊號的持有期是 10 天，那麼第 t 天的標籤（未來 10 天
    報酬）與第 t+5 天的標籤有 5 天的重疊。訓練集用了第 t 天、測試集用了第 t+5 天，
    這兩個樣本並不獨立——測試集的答案有一部分已經在訓練集裡了。
    留白把這段重疊切掉。
    """
    if n_samples <= 0:
        msg = "樣本數必須為正"
        raise ValueError(msg)
    if n_splits < 2:
        msg = "切分數至少為 2"
        raise ValueError(msg)
    if embargo < 0:
        msg = "embargo 不得為負數"
        raise ValueError(msg)

    fold_size = n_samples // (n_splits + 1)
    if fold_size == 0:
        msg = f"樣本數 {n_samples} 不足以切成 {n_splits} 份"
        raise ValueError(msg)

    splits: list[Split] = []
    for index in range(1, n_splits + 1):
        train_end = fold_size * index
        test_start = min(train_end + embargo, n_samples)
        test_end = min(test_start + fold_size, n_samples)
        if test_start >= test_end:
            continue
        splits.append(
            Split(
                train=tuple(range(0, train_end)),
                test=tuple(range(test_start, test_end)),
            )
        )
    return splits


def combinatorial_purged_cv(
    n_samples: int,
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo: int = 0,
) -> list[Split]:
    """Combinatorial purged cross-validation。

    把樣本切成 ``n_groups`` 組，每次取 ``n_test_groups`` 組當測試集，
    其餘為訓練集，窮舉所有組合。

    這樣得到的不是一條回測曲線，而是 C(n_groups, n_test_groups) 條路徑。
    單一曲線只能告訴你「這一次表現如何」；一組路徑的分布才能告訴你
    「這個表現有多穩定」——而後者才是決策需要的資訊（SPEC 5.3）。
    """
    if not 1 <= n_test_groups < n_groups:
        msg = "測試組數必須小於總組數且至少為 1"
        raise ValueError(msg)
    group_size = n_samples // n_groups
    if group_size == 0:
        msg = f"樣本數 {n_samples} 不足以切成 {n_groups} 組"
        raise ValueError(msg)

    groups = [
        tuple(range(index * group_size, (index + 1) * group_size)) for index in range(n_groups)
    ]
    # 最後一組吸收餘數。
    groups[-1] = tuple(range((n_groups - 1) * group_size, n_samples))

    splits: list[Split] = []
    for test_ids in combinations(range(n_groups), n_test_groups):
        test_indices = sorted(index for group_id in test_ids for index in groups[group_id])
        purged = _purge(test_indices, embargo)
        train_indices = tuple(index for index in range(n_samples) if index not in purged)
        if not train_indices or not test_indices:
            continue
        splits.append(Split(train=train_indices, test=tuple(test_indices)))
    return splits


def _purge(test_indices: Sequence[int], embargo: int) -> set[int]:
    """測試集本身，加上其前後 embargo 範圍內的索引。"""
    excluded = set(test_indices)
    for index in test_indices:
        for offset in range(1, embargo + 1):
            excluded.add(index - offset)
            excluded.add(index + offset)
    return excluded


def sharpe_ratio(
    returns: npt.ArrayLike,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """年化 Sharpe。序列實質無波動時回傳 NaN。

    「實質無波動」用相對門檻判斷而不是 ``std == 0``。理由是浮點誤差：
    一條常數序列 [0.01] * 100 的樣本標準差會算出 1.7e-18 而非 0，
    於是 Sharpe 變成 9e16——一個荒謬但不會報錯的數字，而且它會在
    任何排序中排第一。壞掉的資料與只持有現金的策略都會踩到這個情況。
    """
    series = np.asarray(returns, dtype=np.float64).ravel()
    series = series[np.isfinite(series)]
    if series.size < 2:
        return float("nan")
    mean = float(np.mean(series))
    std = float(np.std(series, ddof=1))
    scale = max(abs(mean), float(np.max(np.abs(series))) if series.size else 0.0)
    if std <= max(scale * _RELATIVE_ZERO, _ABSOLUTE_ZERO):
        return float("nan")
    return mean / std * math.sqrt(periods_per_year)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_observations: int,
    variance_of_trial_sharpes: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio：把「試了幾次」的影響從 Sharpe 裡扣掉。

    回傳值是一個機率：**在扣除多重檢定效應後，真實 Sharpe 大於零的機率**。

    **尺度很重要**：``observed_sharpe`` 與 ``variance_of_trial_sharpes``
    必須是同一頻率下的**非年化**值。傳年化 Sharpe 進來會讓它與試驗變異數
    的尺度不一致，算出來的機率沒有意義。

    ``variance_of_trial_sharpes`` 刻意沒有預設值。它是「你試過的那些策略，
    它們的 Sharpe 之間的變異數」，只有呼叫端知道。給一個預設值等於替使用者
    猜一個關鍵參數，而猜錯的方向永遠是高估顯著性。

    ``n_trials`` 必須誠實填寫累計試驗次數——包含所有調過的參數、所有測過但
    放棄的假設。這正是 ``LibrarianAgent`` 存在的理由（SPEC 5.3）：
    人類會忘記自己試過幾次，而忘記的方向永遠是低估。
    """
    from scipy import stats

    if n_trials < 1:
        msg = "試驗次數至少為 1"
        raise ValueError(msg)
    if n_observations < 3:
        msg = "觀測數至少為 3"
        raise ValueError(msg)
    if variance_of_trial_sharpes <= 0:
        msg = "試驗 Sharpe 的變異數必須為正數"
        raise ValueError(msg)
    if not math.isfinite(observed_sharpe):
        return float("nan")

    # 在 n_trials 次獨立試驗下，最大 Sharpe 的期望值。
    # 這是即使所有策略都無效，你也會看到的那個 Sharpe。
    if n_trials == 1:
        expected_max = 0.0
    else:
        quantile_a = stats.norm.ppf(1 - 1.0 / n_trials)
        quantile_b = stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
        expected_max = math.sqrt(variance_of_trial_sharpes) * (
            (1 - _EULER_GAMMA) * quantile_a + _EULER_GAMMA * quantile_b
        )

    # Sharpe 估計量的標準誤，考慮偏態與峰態。
    denominator = 1.0 - skewness * observed_sharpe + (kurtosis - 1) / 4.0 * observed_sharpe**2
    if denominator <= 0:
        return float("nan")
    standard_error = math.sqrt(denominator / (n_observations - 1))
    if standard_error == 0:
        return float("nan")

    return float(stats.norm.cdf((observed_sharpe - expected_max) / standard_error))


def probability_of_backtest_overfitting(
    trial_returns: Sequence[npt.ArrayLike],
    *,
    n_groups: int = 8,
) -> float:
    """PBO：過度配適的機率（SPEC 5.3，超過 0.5 直接淘汰）。

    做法是把時間切成若干組，窮舉「一半當樣本內、一半當樣本外」的組合。
    每次在樣本內挑出表現最好的策略，看它在樣本外的排名。

    若這些策略都只是雜訊，樣本內的冠軍在樣本外就會隨機落在中位數附近——
    落在後半段的機率是 50%。**PBO 就是這個機率。**
    因此 PBO 接近 0.5 代表「樣本內的好表現完全無法預測樣本外」。
    """
    matrix = np.asarray(
        [np.asarray(returns, dtype=np.float64).ravel() for returns in trial_returns]
    )
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        msg = "至少需要兩組試驗的報酬序列"
        raise ValueError(msg)

    n_trials, n_observations = matrix.shape
    if n_groups < 2 or n_groups % 2 != 0:
        msg = "組數必須是大於等於 2 的偶數"
        raise ValueError(msg)
    group_size = n_observations // n_groups
    if group_size == 0:
        msg = f"觀測數 {n_observations} 不足以切成 {n_groups} 組"
        raise ValueError(msg)

    groups = [
        list(range(index * group_size, (index + 1) * group_size)) for index in range(n_groups)
    ]
    half = n_groups // 2

    logits: list[float] = []
    for in_sample_ids in combinations(range(n_groups), half):
        out_ids = [index for index in range(n_groups) if index not in in_sample_ids]
        in_indices = [index for group_id in in_sample_ids for index in groups[group_id]]
        out_indices = [index for group_id in out_ids for index in groups[group_id]]

        in_sharpes = np.array(
            [sharpe_ratio(matrix[trial, in_indices]) for trial in range(n_trials)]
        )
        out_sharpes = np.array(
            [sharpe_ratio(matrix[trial, out_indices]) for trial in range(n_trials)]
        )
        if not np.isfinite(in_sharpes).any() or not np.isfinite(out_sharpes).any():
            continue

        best = int(np.nanargmax(in_sharpes))
        finite_out = out_sharpes[np.isfinite(out_sharpes)]
        if finite_out.size < 2 or not math.isfinite(float(out_sharpes[best])):
            continue
        # 樣本內冠軍在樣本外的相對排名。
        rank = float(np.mean(finite_out < out_sharpes[best]))
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1 - rank)))

    if not logits:
        return float("nan")
    # 落在樣本外後半段（logit < 0）的比例。
    return float(np.mean([1.0 if value < 0 else 0.0 for value in logits]))


@dataclass(frozen=True)
class CapacityPoint:
    """容量曲線上的一點。"""

    capital: float
    #: 該資金規模下的年化報酬。
    annualized_return: float
    #: 相對於無衝擊基準的績效衰減比例。
    decay: float


def capacity_curve(
    base_annual_return: float,
    *,
    capital_levels: Sequence[float],
    average_daily_volume_value: float,
    participation_cap: float,
    impact_coefficient: float,
    turnover_per_year: float,
) -> list[CapacityPoint]:
    """容量衰退曲線（SPEC 5.3）。

    隨著資金變大，同一個策略要買的量佔日均量的比例上升，市場衝擊隨之增加，
    報酬被吃掉。這條曲線回答的是「這個策略最多能容納多少錢」——
    一個 Sharpe 2.0 但只能容納一千萬的策略，和能容納十億的策略，
    是完全不同的兩件事。
    """
    if average_daily_volume_value <= 0:
        msg = "日均量金額必須為正數"
        raise ValueError(msg)
    if not 0 < participation_cap <= 1:
        msg = "參與率上限需落在 (0, 1]"
        raise ValueError(msg)

    points: list[CapacityPoint] = []
    for capital in capital_levels:
        if capital < 0:
            msg = "資金規模不得為負數"
            raise ValueError(msg)
        # 每年的成交金額 = 資金 × 換手率。
        annual_notional = capital * turnover_per_year
        daily_notional = annual_notional / TRADING_DAYS_PER_YEAR
        participation = daily_notional / average_daily_volume_value
        # 超過參與率上限的部分無法成交，等於策略被迫縮水。
        effective_participation = min(participation, participation_cap)
        impact_per_trade = impact_coefficient * math.sqrt(max(effective_participation, 0.0))
        annual_impact = impact_per_trade * turnover_per_year
        net_return = base_annual_return - annual_impact
        decay = (
            (base_annual_return - net_return) / base_annual_return
            if base_annual_return != 0
            else 0.0
        )
        points.append(
            CapacityPoint(
                capital=capital,
                annualized_return=net_return,
                decay=decay,
            )
        )
    return points


def iter_split_returns(
    returns: npt.ArrayLike,
    splits: Sequence[Split],
) -> Iterator[FloatArray]:
    """依切分取出各測試集的報酬序列。"""
    series = np.asarray(returns, dtype=np.float64).ravel()
    for split in splits:
        indices = [index for index in split.test if 0 <= index < series.size]
        yield series[indices]

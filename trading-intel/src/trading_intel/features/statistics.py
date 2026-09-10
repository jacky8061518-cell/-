"""統計量模組。

SPEC 5.1 對這一節有明確的警告：**均值與標準差不要用天真的滾動視窗。**

理由是報酬序列非定態。在一段趨勢裡，滾動 z-score 會持續告訴你「已經漲太多、
該反轉了」，然後價格繼續漲。這個錯誤訊號不會停止，因為產生它的統計量
從一開始就假設了序列會回到均值——而趨勢中的序列不會。

因此本模組的設計是：
1. 先做定態檢定（ADF 與 KPSS）；
2. 通過的序列才以 OU 過程估計半衰期，並由半衰期決定視窗長度；
3. **未通過定態檢定的序列，均值回歸類特徵一律回傳 NaN，不硬算。**

第 3 點是重點。回傳 NaN 會讓下游少一個訊號；硬算會讓下游多一個錯誤訊號。
兩者的代價不對稱。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
import numpy.typing as npt

#: 定態檢定的預設顯著水準。
DEFAULT_ALPHA: Final = 0.05

#: OU 半衰期估計所需的最少觀測數。太短的序列估出來的半衰期沒有意義。
MIN_OBSERVATIONS_FOR_HALFLIFE: Final = 30

#: 定態檢定所需的最少觀測數。
MIN_OBSERVATIONS_FOR_STATIONARITY: Final = 20

#: 一年的交易日數，用於年化。
TRADING_DAYS_PER_YEAR: Final = 252

FloatArray = npt.NDArray[np.float64]


class StationarityVerdict(StrEnum):
    """ADF 與 KPSS 合議後的結論。

    兩個檢定的虛無假設相反，刻意併用：
    - ADF 的虛無假設是「有單位根」，拒絕才代表定態；
    - KPSS 的虛無假設是「定態」，拒絕才代表非定態。

    兩者一致時結論可信；兩者矛盾時結論是 INCONCLUSIVE，
    而 INCONCLUSIVE 一律當作「不可用於均值回歸」處理。
    """

    STATIONARY = "STATIONARY"
    NON_STATIONARY = "NON_STATIONARY"
    #: 兩個檢定互相矛盾，或樣本太短無法判定。
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class StationarityResult:
    """定態檢定的完整結果，含兩個檢定各自的 p 值。"""

    verdict: StationarityVerdict
    adf_pvalue: float
    kpss_pvalue: float
    observations: int
    alpha: float

    @property
    def usable_for_mean_reversion(self) -> bool:
        """只有明確定態才可用於均值回歸。存疑一律視為不可用。"""
        return self.verdict is StationarityVerdict.STATIONARY


def _as_clean_array(values: npt.ArrayLike) -> FloatArray:
    """轉為 float 陣列並剔除非有限值。"""
    array = np.asarray(values, dtype=np.float64).ravel()
    return array[np.isfinite(array)]


def check_stationarity(
    values: npt.ArrayLike,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> StationarityResult:
    """以 ADF 與 KPSS 合議判定序列是否定態。

    使用 statsmodels 的實作而非自行撰寫：這兩個檢定需要 MacKinnon 臨界值表，
    自行實作出錯不易察覺，而錯誤的定態判定會讓整個均值回歸類特徵失效。
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    series = _as_clean_array(values)
    if series.size < MIN_OBSERVATIONS_FOR_STATIONARITY:
        return StationarityResult(
            verdict=StationarityVerdict.INCONCLUSIVE,
            adf_pvalue=float("nan"),
            kpss_pvalue=float("nan"),
            observations=int(series.size),
            alpha=alpha,
        )
    # 常數序列在數學上是定態的，但兩個檢定都會在其上失敗。
    if float(np.std(series)) == 0.0:
        return StationarityResult(
            verdict=StationarityVerdict.INCONCLUSIVE,
            adf_pvalue=float("nan"),
            kpss_pvalue=float("nan"),
            observations=int(series.size),
            alpha=alpha,
        )

    adf_pvalue = float(adfuller(series, autolag="AIC", result_object=False)[1])
    # KPSS 在 p 值超出臨界值表範圍時會發 InterpolationWarning，
    # 那是預期行為（p 值被截斷在 0.01 或 0.1），不是錯誤。
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_pvalue = float(kpss(series, regression="c", nlags="auto")[1])

    adf_says_stationary = adf_pvalue < alpha
    kpss_says_stationary = kpss_pvalue > alpha

    if adf_says_stationary and kpss_says_stationary:
        verdict = StationarityVerdict.STATIONARY
    elif not adf_says_stationary and not kpss_says_stationary:
        verdict = StationarityVerdict.NON_STATIONARY
    else:
        verdict = StationarityVerdict.INCONCLUSIVE

    return StationarityResult(
        verdict=verdict,
        adf_pvalue=adf_pvalue,
        kpss_pvalue=kpss_pvalue,
        observations=int(series.size),
        alpha=alpha,
    )


def ou_half_life(values: npt.ArrayLike, *, alpha: float = DEFAULT_ALPHA) -> float:
    """以 Ornstein-Uhlenbeck 過程估計均值回歸的半衰期（單位：期）。

    做法是對 ``Δy_t = a + b·y_{t-1} + ε`` 做最小平方迴歸，
    半衰期為 ``-ln(2) / ln(1 + b)``。

    **必須先通過 ADF 定態檢定才估計**，這與 SPEC 5.1 的原文一致
    （「對通過的序列以 OU 過程估計半衰期」）。不能省的理由是：

    在單位根附近，OLS 的 t 統計量不服從標準 t 分布，而是 Dickey-Fuller 分布，
    後者的臨界值明顯更負。用普通 t 檢定把關會讓隨機漫步「顯著」通過——
    實測一條 2000 點的隨機漫步，普通 t 檢定會讓它估出 163 期的半衰期，
    而那個數字完全是虛假的。下游會拿它去設視窗長度，這正是本模組要防的錯誤。

    無法辨識回歸傾向時回傳 NaN。「沒有半衰期」和「半衰期很長」是不同的事。
    """
    series = _as_clean_array(values)
    if series.size < MIN_OBSERVATIONS_FOR_HALFLIFE:
        return float("nan")

    # 先過定態這一關。ADF 的虛無假設是「有單位根」，拒絕才代表可以談回歸。
    if check_stationarity(series, alpha=alpha).adf_pvalue >= alpha:
        return float("nan")

    lagged = series[:-1]
    delta = np.diff(series)
    if float(np.std(lagged)) == 0.0:
        return float("nan")

    design = np.column_stack([np.ones_like(lagged), lagged])
    coefficients, *_ = np.linalg.lstsq(design, delta, rcond=None)
    beta = float(coefficients[1])

    if beta >= 0 or beta <= -2:
        # beta >= 0：無回歸傾向。beta <= -2：超調到發散，模型不適用。
        return float("nan")

    decay = math.log1p(beta)
    if decay >= 0:
        return float("nan")
    half_life = -math.log(2) / decay

    # 半衰期長於樣本的一半時，樣本不足以辨識它。
    if half_life > series.size / 2:
        return float("nan")
    return half_life


def window_from_half_life(
    half_life: float,
    *,
    multiplier: float = 2.0,
    minimum: int = 5,
    maximum: int = 252,
) -> int | None:
    """由半衰期推導視窗長度。

    SPEC 5.1：「視窗長度由半衰期決定而非拍腦袋設 20 日」。

    倍數預設 2：涵蓋約兩個半衰期，即約 75% 的均值回歸過程。
    半衰期為 NaN（無回歸傾向）時回傳 None，呼叫端據此拒絕計算均值回歸特徵。
    """
    if not math.isfinite(half_life) or half_life <= 0:
        return None
    window = round(half_life * multiplier)
    return max(minimum, min(window, maximum))


def ewma_volatility(
    returns: npt.ArrayLike,
    *,
    span: int = 20,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """指數加權移動平均波動率。

    SPEC 5.1 要求不要用簡單樣本標準差。EWMA 給近期觀測較高權重，
    因此對波動率變化的反應比等權平均快得多——而波動率的叢聚性
    （高波動之後傾向繼續高波動）是報酬序列少數穩健的性質之一。
    """
    series = _as_clean_array(returns)
    if series.size < 2:
        return float("nan")
    if span < 1:
        msg = "span 必須為正整數"
        raise ValueError(msg)

    alpha = 2.0 / (span + 1.0)
    weights = (1 - alpha) ** np.arange(series.size)[::-1]
    weights /= weights.sum()
    mean = float(np.sum(weights * series))
    variance = float(np.sum(weights * (series - mean) ** 2))
    volatility = math.sqrt(max(variance, 0.0))
    return volatility * math.sqrt(periods_per_year) if annualize else volatility


def yang_zhang_volatility(
    open_: npt.ArrayLike,
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    close: npt.ArrayLike,
    *,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Yang-Zhang 波動率估計量。

    只用收盤價的估計量會忽略盤中的價格資訊，因此效率低落。Yang-Zhang 同時
    使用開高低收，把波動拆成三部分：隔夜跳空、開盤到收盤、以及 Rogers-Satchell
    的盤中項，並以一個係數 k 把三者組合起來。

    它對跳空與趨勢都是無偏的，這是它比 Parkinson 與 Garman-Klass 更適合
    台股的原因——台股有漲跌幅限制，跳空開盤相當常見。
    """
    o = _as_clean_array(open_)
    h = _as_clean_array(high)
    low_array = _as_clean_array(low)
    c = _as_clean_array(close)
    if not (o.size == h.size == low_array.size == c.size):
        msg = "開高低收四個序列長度必須相同"
        raise ValueError(msg)
    n = o.size
    if n < 3:
        return float("nan")
    if np.any(o <= 0) or np.any(h <= 0) or np.any(low_array <= 0) or np.any(c <= 0):
        msg = "價格必須為正數"
        raise ValueError(msg)

    # 隔夜項：前一日收盤到今日開盤。
    overnight = np.log(o[1:] / c[:-1])
    # 開盤到收盤項。
    open_to_close = np.log(c[1:] / o[1:])
    # Rogers-Satchell 盤中項，對趨勢無偏。
    rs = np.log(h[1:] / c[1:]) * np.log(h[1:] / o[1:]) + np.log(low_array[1:] / c[1:]) * np.log(
        low_array[1:] / o[1:]
    )

    m = overnight.size
    if m < 2:
        return float("nan")
    overnight_var = float(np.sum((overnight - overnight.mean()) ** 2) / (m - 1))
    open_close_var = float(np.sum((open_to_close - open_to_close.mean()) ** 2) / (m - 1))
    rs_var = float(np.sum(rs) / m)

    k = 0.34 / (1.34 + (m + 1) / (m - 1))
    variance = overnight_var + k * open_close_var + (1 - k) * rs_var
    volatility = math.sqrt(max(variance, 0.0))
    return volatility * math.sqrt(periods_per_year) if annualize else volatility


def winsorize(values: npt.ArrayLike, *, lower: float = 0.01, upper: float = 0.99) -> FloatArray:
    """把極值拉回指定百分位，而不是刪除。

    SPEC 5.1：「極值處理用 winsorize 到 1 與 99 百分位，不要直接刪除」。
    刪除會改變樣本數並在橫斷面上製造缺口；拉回則保留「這個值很極端」的資訊。
    """
    if not 0 <= lower < upper <= 1:
        msg = "百分位需滿足 0 <= lower < upper <= 1"
        raise ValueError(msg)
    array = np.asarray(values, dtype=np.float64).ravel().copy()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return array
    low_bound = float(np.quantile(finite, lower))
    high_bound = float(np.quantile(finite, upper))
    mask = np.isfinite(array)
    array[mask] = np.clip(array[mask], low_bound, high_bound)
    return array


def zscore(values: npt.ArrayLike, *, ddof: int = 1) -> FloatArray:
    """標準化。標準差為零時回傳全 NaN 而非全零。

    全零會讓下游以為「每個標的都在平均值上」，那是一個看似正常的錯誤答案；
    NaN 則會誠實地讓下游知道這裡算不出東西。
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    finite = array[np.isfinite(array)]
    if finite.size < 2:
        return np.full_like(array, np.nan)
    std = float(np.std(finite, ddof=ddof))
    if std == 0.0:
        return np.full_like(array, np.nan)
    return (array - float(np.mean(finite))) / std

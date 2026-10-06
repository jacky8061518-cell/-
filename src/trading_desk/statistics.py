"""Robust statistics used by every deterministic agent.

Financial returns are fat-tailed, so a single gap can poison a sample standard
deviation and every z-score derived from it. Everything here is chosen to stay
usable when the data misbehaves: median absolute deviation instead of standard
deviation, cross-sectional ranks instead of raw levels, and explicit handling of
missing values rather than a silent ``fillna(0)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826
RISKMETRICS_LAMBDA = 0.94


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip extreme values so one bad print cannot dominate a model."""
    clean = series.dropna()
    if clean.empty:
        return series
    low, high = clean.quantile([lower, upper])
    return series.clip(low, high)


def robust_zscore(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
    min_sigma: float | None = None,
) -> pd.Series:
    """Rolling z-score built on median and MAD rather than mean and std.

    MAD degenerates to zero whenever more than half the window is identical,
    which happens constantly with thinly traded names that print the same close
    for days. Left alone that produces a NaN score for exactly the series a
    z-score is most needed on, so the scale estimate falls back to the rolling
    standard deviation and then to ``min_sigma`` before giving up.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    min_periods = min_periods or max(3, window // 2)
    median = series.rolling(window, min_periods=min_periods).median()
    deviation = (series - median).abs()
    mad = deviation.rolling(window, min_periods=min_periods).median()
    sigma = _usable_scale(
        mad * MAD_TO_SIGMA,
        series.rolling(window, min_periods=min_periods).std(ddof=1),
        min_sigma,
    )
    return (series - median) / sigma


def _usable_scale(
    primary: pd.Series | pd.DataFrame,
    fallback: pd.Series | pd.DataFrame,
    min_sigma: float | None,
) -> pd.Series | pd.DataFrame:
    """Prefer the robust scale, fall back to the ordinary one, then to a floor."""
    scale = primary.replace(0.0, np.nan)
    scale = scale.fillna(fallback.replace(0.0, np.nan))
    if min_sigma is not None:
        scale = scale.fillna(min_sigma).clip(lower=min_sigma)
    return scale.replace(0.0, np.nan)


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardise each row against its own cross-section.

    Answers "how unusual is this name among its peers today", which is the
    comparison that matters for ranking, unlike a purely time-series z-score.
    """
    median = frame.median(axis=1)
    mad = frame.sub(median, axis=0).abs().median(axis=1)
    sigma = (mad * MAD_TO_SIGMA).replace(0.0, np.nan)
    return frame.sub(median, axis=0).div(sigma, axis=0)


def rank_to_normal(series: pd.Series) -> pd.Series:
    """Map a cross-section onto an approximately normal score in [-3, 3].

    Rank-based standardisation is immune to outliers, which makes it far more
    stable than a raw z-score when a handful of names gap overnight.
    """
    clean = series.dropna()
    if clean.empty:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    if len(clean) == 1:
        return pd.Series(0.0, index=series.index).where(series.notna())
    ranks = clean.rank(method="average")
    uniform = (ranks - 0.5) / len(clean)
    # Inverse-normal approximation avoids a SciPy dependency for one function.
    scores = np.sqrt(2.0) * _erfinv(2.0 * uniform.to_numpy() - 1.0)
    return pd.Series(scores, index=clean.index).reindex(series.index).clip(-3.0, 3.0)


def _erfinv(values: np.ndarray) -> np.ndarray:
    """Winitzki's approximation of the inverse error function (max error ~2e-3)."""
    clipped = np.clip(values, -0.999999, 0.999999)
    a = 0.147
    ln_term = np.log(1.0 - clipped**2)
    first = 2.0 / (np.pi * a) + ln_term / 2.0
    return np.sign(clipped) * np.sqrt(np.sqrt(first**2 - ln_term / a) - first)


def ewma_volatility(
    returns: pd.Series,
    lam: float = RISKMETRICS_LAMBDA,
    annualisation: int = 252,
) -> float:
    """Exponentially weighted volatility, annualised.

    Recent observations matter more than a flat 60-day average, and the
    RiskMetrics lambda of 0.94 is the standard daily choice.
    """
    clean = returns.dropna()
    if len(clean) < 5:
        return float("nan")
    weights = lam ** np.arange(len(clean) - 1, -1, -1)
    weights = weights / weights.sum()
    mean = float(np.average(clean, weights=weights))
    variance = float(np.average((clean - mean) ** 2, weights=weights))
    return float(np.sqrt(variance * annualisation))


def realised_volatility(
    returns: pd.Series,
    window: int,
    annualisation: int = 252,
) -> float:
    """Plain trailing volatility, kept alongside EWMA as an independent view."""
    clean = returns.dropna().tail(window)
    if len(clean) < max(5, window // 2):
        return float("nan")
    return float(clean.std(ddof=1) * np.sqrt(annualisation))


def downside_volatility(
    returns: pd.Series,
    window: int,
    annualisation: int = 252,
) -> float:
    """Semi-deviation below zero. Traders care about the downside, not variance."""
    clean = returns.dropna().tail(window)
    negative = clean[clean < 0]
    if len(negative) < 3:
        return float("nan")
    return float(np.sqrt((negative**2).mean() * annualisation))


def volatility_regime_ratio(returns: pd.Series, short: int = 20, long: int = 252) -> float:
    """Short-horizon volatility divided by long-horizon volatility.

    Above ~1.5 the market has moved into a different volatility regime and any
    model trained on the calm period should be treated with suspicion.
    """
    short_vol = realised_volatility(returns, short)
    long_vol = realised_volatility(returns, long)
    if not np.isfinite(short_vol) or not np.isfinite(long_vol) or long_vol == 0:
        return float("nan")
    return float(short_vol / long_vol)


def max_drawdown(prices: pd.Series) -> float:
    """Worst peak-to-trough decline over the supplied window."""
    clean = prices.dropna()
    if clean.empty:
        return float("nan")
    running_peak = clean.cummax()
    return float((clean / running_peak - 1.0).min())


def population_stability_index(
    baseline: pd.Series,
    current: pd.Series,
    buckets: int = 10,
) -> float:
    """PSI between a training distribution and today's distribution.

    Above 0.25 the model is being asked about a world it never saw during
    training, which is a demotion trigger rather than a retraining suggestion.
    """
    base = baseline.dropna()
    live = current.dropna()
    if len(base) < buckets * 2 or len(live) < buckets:
        return float("nan")
    edges = np.unique(np.quantile(base, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        return float("nan")
    base_share = np.histogram(base, bins=edges)[0] / len(base)
    live_share = np.histogram(live, bins=edges)[0] / len(live)
    floor = 1e-6
    base_share = np.clip(base_share, floor, None)
    live_share = np.clip(live_share, floor, None)
    return float(np.sum((live_share - base_share) * np.log(live_share / base_share)))


def effective_positions(weights: pd.Series) -> float:
    """Inverse Herfindahl index: how many independent bets are really on."""
    clean = weights.dropna().abs()
    total = clean.sum()
    if total <= 0:
        return 0.0
    share = clean / total
    return float(1.0 / (share**2).sum())


def correlation_clusters(
    returns: pd.DataFrame,
    threshold: float = 0.7,
) -> dict[str, list[str]]:
    """Group names whose returns move together above a correlation threshold.

    Diversifying across twenty names that all fall together is not
    diversification, so limits are applied to clusters rather than to tickers.
    """
    usable = returns.dropna(axis=1, how="all")
    if usable.shape[1] < 2:
        return {name: [name] for name in usable.columns}
    matrix = usable.corr(min_periods=20).fillna(0.0)
    remaining = list(matrix.columns)
    clusters: dict[str, list[str]] = {}
    while remaining:
        seed = remaining.pop(0)
        members = [seed]
        for candidate in list(remaining):
            if float(matrix.at[seed, candidate]) >= threshold:
                members.append(candidate)
                remaining.remove(candidate)
        clusters[seed] = members
    return clusters


# --- Panel versions -------------------------------------------------------
# The scalar helpers above read clearly for one series, but calling them
# column-by-column across a few thousand tickers is slow enough to be felt in
# the console. These operate on the whole panel using pandas' vectorised
# rolling and ewm kernels and return identical definitions.


def robust_zscore_frame(
    frame: pd.DataFrame,
    window: int,
    min_sigma: float | None = None,
) -> pd.DataFrame:
    """Column-wise :func:`robust_zscore` computed across the whole panel at once.

    ``min_sigma`` floors the scale estimate. Illiquid names barely move most
    days, so their MAD collapses towards zero and an ordinary 2% move prints a
    fifteen-sigma score. Flooring the denominator keeps those names comparable
    with liquid ones instead of letting them monopolise every anomaly list.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    min_periods = max(3, window // 2)
    median = frame.rolling(window, min_periods=min_periods).median()
    mad = (frame - median).abs().rolling(window, min_periods=min_periods).median()
    sigma = _usable_scale(
        mad * MAD_TO_SIGMA,
        frame.rolling(window, min_periods=min_periods).std(ddof=1),
        min_sigma,
    )
    return (frame - median) / sigma


def ewma_volatility_frame(
    returns: pd.DataFrame,
    lam: float = RISKMETRICS_LAMBDA,
    annualisation: int = 252,
    min_periods: int = 20,
) -> pd.Series:
    """Latest exponentially weighted volatility for every column."""
    weighted = returns.ewm(alpha=1.0 - lam, adjust=False, min_periods=min_periods).std()
    if weighted.empty:
        return pd.Series(dtype="float64")
    return weighted.iloc[-1] * np.sqrt(annualisation)


def realised_volatility_frame(
    returns: pd.DataFrame,
    window: int,
    annualisation: int = 252,
) -> pd.Series:
    """Latest trailing volatility for every column."""
    min_periods = max(5, window // 2)
    rolled = returns.rolling(window, min_periods=min_periods).std(ddof=1)
    if rolled.empty:
        return pd.Series(dtype="float64")
    return rolled.iloc[-1] * np.sqrt(annualisation)


def downside_volatility_frame(
    returns: pd.DataFrame,
    window: int,
    annualisation: int = 252,
) -> pd.Series:
    """Latest semi-deviation below zero for every column."""
    negative = returns.where(returns < 0, 0.0)
    counts = (returns < 0).rolling(window, min_periods=1).sum()
    mean_square = (negative**2).rolling(window, min_periods=3).mean()
    if mean_square.empty:
        return pd.Series(dtype="float64")
    latest = np.sqrt(mean_square.iloc[-1] * annualisation)
    return latest.where(counts.iloc[-1] >= 3)


def max_drawdown_frame(prices: pd.DataFrame) -> pd.Series:
    """Worst peak-to-trough decline over the supplied window, per column."""
    if prices.empty:
        return pd.Series(dtype="float64")
    return (prices / prices.cummax() - 1.0).min()


def market_neutral_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Strip the common market move out of each row.

    On a day when everything falls together, every name looks like a three-sigma
    event. Subtracting the cross-sectional median leaves the idiosyncratic part,
    which is the only part a single-name signal can act on.
    """
    return returns.sub(returns.median(axis=1), axis=0)

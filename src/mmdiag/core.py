"""Microstructure diagnostics for a time-indexed price (mid) series and, optionally, its quoted spread.

All functions take pandas Series with a DatetimeIndex; times are whatever clock your feed uses (state it in your reports).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _log_returns(prices: pd.Series, freq: str | None = None) -> pd.Series:
    p = prices.resample(freq).last().dropna() if freq else prices.dropna()
    r = np.log(p).diff()
    same_day = p.index.normalize() == pd.Series(p.index.normalize(), index=p.index).shift(1).to_numpy()
    return r[same_day].dropna()                                  # drop overnight / weekend gaps


def signature_plot(prices: pd.Series, intervals=("1min", "2min", "5min", "10min", "15min", "30min", "1h")) -> pd.DataFrame:
    """Mean daily realised variance (sum of squared intraday returns) at each sampling interval.

    Flat = the price behaves like a random walk at these scales. Rising towards short intervals = microstructure noise
    (bid-ask bounce, stale quotes); falling towards short intervals = short-horizon positive autocorrelation (momentum, smoothing).
    `ratio_to_longest` normalises by the longest interval."""
    rows = []
    for iv in intervals:
        r = _log_returns(prices, iv)
        rv = (r ** 2).groupby(r.index.normalize()).sum()
        rows.append((iv, pd.Timedelta(iv).total_seconds() / 60, float(rv.mean()), len(rv)))
    out = pd.DataFrame(rows, columns=["interval", "minutes", "mean_daily_rv", "days"]).set_index("interval")
    out["ratio_to_longest"] = out.mean_daily_rv / out.mean_daily_rv.iloc[-1]
    return out


def variance_ratio(prices: pd.Series, base: str = "5min", k: int = 12) -> dict:
    """Lo-MacKinlay variance ratio VR(k) = Var(k-period return) / (k Var(1-period return)) on intraday returns,
    with the heteroskedasticity-robust z statistic. VR > 1 trending, < 1 mean-reverting, = 1 random walk."""
    r = _log_returns(prices, base).to_numpy(); r = r - r.mean(); n = len(r)
    if n < 10 * k:
        raise ValueError("not enough returns for this k")
    rk = np.convolve(r, np.ones(k), "valid")
    vr = (rk @ rk / len(rk)) / (k * (r @ r / n))
    d = r ** 2; denom = (d.sum()) ** 2
    theta = sum((2 * (k - j) / k) ** 2 * (d[j:] @ d[:-j]) / denom for j in range(1, k))
    z = (vr - 1) / math.sqrt(theta) if theta > 0 else float("nan")
    return dict(vr=float(vr), z=float(z), k=k, base=base, n=n)


def intraday_profile(prices: pd.Series, spread: pd.Series | None = None, freq: str = "5min", by: str = "hour") -> pd.DataFrame:
    """Average activity by time of day: return volatility, quote/print count and (if given) median spread."""
    r = _log_returns(prices, freq)
    key = getattr(r.index, by)
    out = pd.DataFrame({"abs_return_bp": (r.abs() * 1e4).groupby(key).mean(),
                        "updates": prices.groupby(getattr(prices.index, by)).size() / prices.index.normalize().nunique()})
    if spread is not None:
        out["median_spread"] = spread.groupby(getattr(spread.index, by)).median()
    out.index.name = by
    return out


def spread_vol_elasticity(prices: pd.Series, spread: pd.Series, freq: str = "1h", min_obs: int = 5) -> dict:
    """Slope of log(mean spread) on log(realised vol) across `freq` buckets.

    Madhavan-Richardson-Roomans-type models imply the spread scales with per-period volatility: elasticity near 1 means
    the spread is mostly a volatility (adverse-selection) price; near 0 means a fixed, order-processing-style cost."""
    r = _log_returns(prices)
    vol = np.sqrt((r ** 2).resample(freq).sum()); cnt = r.resample(freq).count()
    sp = spread.resample(freq).mean()
    df = pd.concat({"vol": vol, "spread": sp, "n": cnt}, axis=1).dropna()
    df = df[(df.n >= min_obs) & (df.vol > 0) & (df.spread > 0)]
    x, y = np.log(df.vol.to_numpy()), np.log(df.spread.to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    se = math.sqrt(resid @ resid / (len(x) - 2) / ((x - x.mean()) @ (x - x.mean())))
    return dict(elasticity=float(slope), se=float(se), r2=float(1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))),
                buckets=len(df), data=df)

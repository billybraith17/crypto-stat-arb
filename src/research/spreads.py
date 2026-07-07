"""OHLC-based effective-spread estimators.

Purpose: quantify how much of a measured short-horizon reversal is mechanical
bid-ask bounce, and feed per-asset half-spread estimates into cost-aware
backtests. With spread ``s``, close prints bounce between bid and ask, which
alone induces a 1-lag autocovariance of about ``-s^2/4`` in price changes
(Roll 1984) even when the mid-price never reverts — so close-to-close reversal
IC has a "bounce floor" that is not monetisable with taker orders.

Two estimators, both requiring only OHLC bars:

- **Corwin-Schultz (2012)** — uses high/low ranges of consecutive bar pairs.
  The high is (almost surely) an ask print and the low a bid print, so the
  observed range embeds the spread once volatility is netted out via the
  two-bar range. Crypto trades 24/7, so the paper's overnight-gap adjustment
  is unnecessary here.
- **Roll (1984)** — infers the spread from the negative first-order
  autocovariance of close-to-close changes. Used as a cross-check; it is
  noisier and undefined when the sample autocovariance is positive.

Both return *proportional* spreads (fraction of price); helpers convert to
half-spread basis points for cost models.
"""

import numpy as np
import pandas as pd

_CS_DENOM = 3.0 - 2.0 * np.sqrt(2.0)


def corwin_schultz_spread(high_wide, low_wide, clip_negative=True):
    """Per-pair Corwin-Schultz proportional spread estimates.

    Parameters
    ----------
    high_wide, low_wide : pd.DataFrame
        Wide (ts × symbol) high/low panels. Bars where either side is NaN, or
        where ``high < low`` or prices are non-positive, yield NaN estimates.
        Stale (forward-filled) bars have ``high == low`` and bias the estimate
        toward zero — mask them out first (see ``build_traded_mask``).
    clip_negative : bool
        The estimator goes negative when the two-bar range is large relative
        to the single-bar ranges (vol >> spread). Following the paper's
        recommendation, negative estimates are floored at 0 by default so
        that time-averages are not dragged below zero.

    Returns
    -------
    pd.DataFrame of proportional full-spread estimates, labelled at the
    *second* bar of each consecutive-bar pair (no lookahead: the estimate at
    ``t`` uses bars ``t-1`` and ``t``).
    """
    high = high_wide.where((high_wide > 0) & (low_wide > 0) & (high_wide >= low_wide))
    low = low_wide.where(high.notna())

    log_hl_sq = np.log(high / low) ** 2
    beta = log_hl_sq.shift(1) + log_hl_sq

    high_pair = np.maximum(high, high.shift(1))
    low_pair = np.minimum(low, low.shift(1))
    gamma = np.log(high_pair / low_pair) ** 2

    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_DENOM - np.sqrt(gamma / _CS_DENOM)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    if clip_negative:
        spread = spread.clip(lower=0.0)
    return spread


def roll_spread(close_wide):
    """Full-sample Roll (1984) proportional spread estimate per symbol.

    ``s = 2 * sqrt(-cov(dp_t, dp_{t-1}))`` on log price changes; NaN where the
    sample autocovariance is non-negative (the model is then uninformative).
    """
    log_ret = np.log(close_wide / close_wide.shift(1))
    estimates = {}
    for col in log_ret.columns:
        x = log_ret[col].dropna()
        if len(x) < 3:
            estimates[col] = np.nan
            continue
        cov = x.autocorr(lag=1) * x.var(ddof=1)
        estimates[col] = 2.0 * np.sqrt(-cov) if cov < 0 else np.nan
    return pd.Series(estimates, name="roll_spread")


def half_spread_bps(spread_panel, stat="median", min_obs=30):
    """Summarise a per-pair spread panel into per-symbol half-spread bps.

    Parameters
    ----------
    spread_panel : pd.DataFrame
        Output of ``corwin_schultz_spread`` (proportional full spreads).
    stat : {"median", "mean"}
        Time-aggregation statistic. Median is robust to the estimator's
        heavy right tail.
    min_obs : int
        Symbols with fewer valid pair-estimates than this return NaN.

    Returns
    -------
    pd.Series of half-spread estimates in basis points per symbol.
    """
    if stat not in ("median", "mean"):
        raise ValueError("stat must be 'median' or 'mean'")
    agg = spread_panel.median() if stat == "median" else spread_panel.mean()
    enough = spread_panel.notna().sum() >= int(min_obs)
    return (agg.where(enough) / 2.0 * 1e4).rename("half_spread_bps")


def estimate_half_spread_bps_from_long(
    df_long,
    estimation_timeframe="1h",
    stat="mean",
    min_obs=30,
):
    """Convenience: long OHLCV rows → per-symbol half-spread bps (Corwin-Schultz).

    Hourly rows with ``trades == 0`` are forward-filled stale bars
    (``high == low == prev close``) that bias the estimator toward zero, so
    they are dropped before resampling.

    Defaults matter here. The estimator's signal-to-noise depends on the
    spread-to-volatility ratio of the estimation bar: crypto trades 24/7, so
    a *daily* range is pure volatility (~3-5%) next to a 5-50 bps spread and
    the estimate comes out vol-dominated (empirically ~10x too high on this
    dataset — e.g. XBT/USD ≈ 30 bps half-spread at 24h vs ≈ 3 bps at 1h).
    Hourly bars keep the ratio workable. With hourly bars roughly half the
    raw pair-estimates clip to zero, which makes the *median* degenerate for
    thin names — the paper's own convention, the **mean with negatives
    floored at zero**, is the default instead.

    Returns ``(half_spread_bps_series, spread_panel)``.
    """
    required = {"ts", "symbol", "high", "low", "trades"}
    missing = required.difference(df_long.columns)
    if missing:
        raise ValueError(f"df_long missing required columns: {sorted(missing)}")

    real = df_long[df_long["trades"] > 0][["ts", "symbol", "high", "low"]].copy()
    real["ts"] = pd.to_datetime(real["ts"], utc=True)

    high_wide = (
        real.pivot(index="ts", columns="symbol", values="high")
        .sort_index()
        .resample(estimation_timeframe)
        .max()
    )
    low_wide = (
        real.pivot(index="ts", columns="symbol", values="low")
        .sort_index()
        .resample(estimation_timeframe)
        .min()
    )

    spread_panel = corwin_schultz_spread(high_wide, low_wide)
    return half_spread_bps(spread_panel, stat=stat, min_obs=min_obs), spread_panel

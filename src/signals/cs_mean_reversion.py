"""Cross-sectional mean reversion research transforms.

All feature builders return panels where **higher values → expect future
outperformance** (i.e. the reversal sign is already baked in where relevant).
Functions imported from cs_momentum are used directly to avoid duplication.
"""

import re

import numpy as np
import pandas as pd

from src.signals.cs_momentum import (
    apply_rebalance_decimation,
    build_monthly_universe_mask,
    compute_bar_returns,
    compute_forward_returns,
    compute_return_horizons,
    cross_sectional_rank_or_zscore,
    resample_to_signal_timeframe,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pivot_volume(df_long, signal_timeframe):
    """Resample hourly volume to signal timeframe (sum within each bar).

    ``min_count=1`` is critical: pandas' default ``.sum()`` on an entirely
    empty bin returns ``0.0``, which would be indistinguishable from a real
    zero-volume bar. We want missing-data intervals to stay NaN so they are
    excluded from downstream volume-conditioned features rather than silently
    treated as "no volume traded".
    """
    panel = df_long[["ts", "symbol", "volume"]].copy()
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    vol_wide = panel.pivot(index="ts", columns="symbol", values="volume").sort_index()
    vol_wide = vol_wide.resample(signal_timeframe).sum(min_count=1)
    return vol_wide.dropna(how="all")


def _align_to(reference, *others):
    """Reindex a sequence of DataFrames to a common (index, columns) reference."""
    return tuple(
        df.reindex(index=reference.index, columns=reference.columns) for df in others
    )


# ---------------------------------------------------------------------------
# 1. Short-Term Return Reversal (Core Feature)
# ---------------------------------------------------------------------------


def build_return_reversal_features(
    close_wide,
    horizons,
    log_returns=True,
    universe_mask=None,
):
    """Negated short-term returns — the canonical mean-reversion signal.

    Returns a dict keyed by horizon (int bars) with sub-keys:
      ``"reversal"``      — negated raw return (buy losers, sell winners)
      ``"xsec_reversal"`` — negated cross-sectionally demeaned return
                            (reversal relative to the peer group, not absolute)

    Parameters
    ----------
    horizons : list[int]
        Look-back windows in bars.  Short windows (1–6) are typical.
    universe_mask : pd.DataFrame or None
        If provided, cross-sectional mean for xsec_reversal is restricted to
        in-universe members (same convention as cs_momentum).
    """
    horizon_returns = compute_return_horizons(close_wide, horizons=horizons, log_returns=log_returns)
    out = {}
    for h, ret_wide in horizon_returns.items():
        if universe_mask is not None:
            mask = universe_mask.reindex(index=ret_wide.index, columns=ret_wide.columns)
            mask = mask.fillna(False).astype(bool)
            xsec_mean = ret_wide.where(mask).mean(axis=1)
        else:
            xsec_mean = ret_wide.mean(axis=1)

        residual = ret_wide.sub(xsec_mean, axis=0)
        out[h] = {
            "reversal": -ret_wide,
            "xsec_reversal": -residual,
        }
    return out


# ---------------------------------------------------------------------------
# 2. Z-Score of Price vs Rolling Mean (Bollinger-style)
# ---------------------------------------------------------------------------


def build_price_zscore_features(
    close_wide,
    price_zscore_window_bars=24,
    bollinger_window_bars=20,
    n_std=2.0,
):
    """Measures how far price has stretched from its own rolling mean.

    The Bollinger feature here is a *band-touch* indicator rather than a
    rescaling of ``price_zscore`` — the textbook %B formulation
    (``(close - lower) / (upper - lower) - 0.5``) is exactly proportional to
    the price z-score (the ratio equals ``2 * n_std``), so it adds no
    information once features are cross-sectionally ranked or z-scored.

    Returns
    -------
    dict with keys:
      ``"price_zscore"``     — negated (close - MA) / rolling_std.
                               Negative: assets trading far above their MA
                               get a low (short) score; far below get a high
                               (long) score.
      ``"bollinger_touch"``  — +1 when close is below the lower band, -1 when
                               above the upper band, 0 otherwise. A discrete,
                               band-conditional companion to ``price_zscore``.
    """
    window_z = int(price_zscore_window_bars)
    window_b = int(bollinger_window_bars)
    n_std = float(n_std)

    rolling_mean_z = close_wide.rolling(window_z, min_periods=window_z).mean()
    rolling_std_z = close_wide.rolling(window_z, min_periods=window_z).std(ddof=0)

    price_zscore = close_wide.sub(rolling_mean_z).div(rolling_std_z.replace(0.0, np.nan))

    rolling_mean_b = close_wide.rolling(window_b, min_periods=window_b).mean()
    rolling_std_b = close_wide.rolling(window_b, min_periods=window_b).std(ddof=0)
    upper_band = rolling_mean_b.add(n_std * rolling_std_b)
    lower_band = rolling_mean_b.sub(n_std * rolling_std_b)

    below = (close_wide < lower_band).astype(float)
    above = (close_wide > upper_band).astype(float)
    bollinger_touch = below.sub(above)
    # Preserve NaN warm-up so IC code doesn't treat it as a "no touch" reading.
    valid_band = upper_band.notna() & lower_band.notna()
    bollinger_touch = bollinger_touch.where(valid_band)

    return {
        "price_zscore": -price_zscore,
        "bollinger_touch": bollinger_touch,
    }


# ---------------------------------------------------------------------------
# 3. Volatility-Adjusted Move
# ---------------------------------------------------------------------------


def build_vol_adjusted_move(
    close_wide,
    horizons,
    vol_window_bars=24,
    log_returns=True,
):
    """Short-term return scaled by trailing volatility, then negated.

    A large move relative to the asset's own recent volatility is a stronger
    mean-reversion signal than the same raw return.  Negation bakes in the
    reversal direction.

    Returns a dict keyed by horizon with keys ``"vol_adj_reversal"`` and
    ``"vol_adj_xsec_reversal"`` (cross-sectional demean of the scaled return,
    then negated).
    """
    vol_window = int(vol_window_bars)
    bar_returns = compute_bar_returns(close_wide, log_returns=log_returns)
    rolling_vol = bar_returns.rolling(vol_window, min_periods=vol_window).std(ddof=0)

    horizon_returns = compute_return_horizons(close_wide, horizons=horizons, log_returns=log_returns)
    out = {}
    for h, ret_wide in horizon_returns.items():
        scaled = ret_wide.div(rolling_vol.replace(0.0, np.nan))
        xsec_mean_scaled = scaled.mean(axis=1)
        out[h] = {
            "vol_adj_reversal": -scaled,
            "vol_adj_xsec_reversal": -(scaled.sub(xsec_mean_scaled, axis=0)),
        }
    return out


# ---------------------------------------------------------------------------
# 4. Extreme Move Indicator
# ---------------------------------------------------------------------------


def build_extreme_move_indicator(
    close_wide,
    vol_window_bars=24,
    threshold_sigma=2.0,
    log_returns=True,
):
    """Identify assets that have undergone extreme single-bar moves.

    Returns
    -------
    dict with keys:
      ``"signed_z"``         — signed z-score of the most recent bar return,
                               negated for reversal direction.
      ``"extreme_flag"``     — continuous: ``signed_z`` where ``|z| > threshold``,
                               else NaN.  Useful for conditional IC studies.
      ``"extreme_magnitude"``— absolute z-score (unsigned), set to NaN where
                               ``|z| <= threshold``.  Captures the size of the
                               extreme move without direction; combine with
                               ``signed_z`` for a signed extreme signal.
    """
    vol_window = int(vol_window_bars)
    bar_ret = compute_bar_returns(close_wide, log_returns=log_returns)
    rolling_mean = bar_ret.rolling(vol_window, min_periods=vol_window).mean()
    rolling_std = bar_ret.rolling(vol_window, min_periods=vol_window).std(ddof=0)

    z = bar_ret.sub(rolling_mean).div(rolling_std.replace(0.0, np.nan))
    signed_z_negated = -z
    threshold = float(threshold_sigma)

    extreme_flag = signed_z_negated.where(z.abs() > threshold)
    extreme_magnitude = z.abs().where(z.abs() > threshold)

    return {
        "signed_z": signed_z_negated,
        "extreme_flag": extreme_flag,
        "extreme_magnitude": extreme_magnitude,
    }


# ---------------------------------------------------------------------------
# 5. Volume Exhaustion
# ---------------------------------------------------------------------------


def build_volume_exhaustion_features(
    close_wide,
    volume_wide,
    horizons,
    volume_window_bars=24,
    log_returns=True,
    volume_ratio_clip=(0.1, 10.0),
):
    """Volume-conditioned mean-reversion features.

    Two hypotheses:
    1. **Exhaustion (low-volume move)**: a price move on below-average volume
       suggests weak conviction → stronger reversion candidate.
       ``price_move / volume_ratio`` — scaled up when volume is low.
    2. **Blow-off (high-volume move)**: a sharp move on anomalously high volume
       can signal capitulation / exhaustion of the dominant side.
       ``price_move * volume_ratio`` — scaled up when volume is high.

    Both are negated so higher values mean *buy* (expected to recover/revert).

    ``volume_ratio`` is clipped to ``volume_ratio_clip`` (default
    ``[0.1, 10.0]``) before being used in the divisions / multiplications.
    Without the clip, a single thin-volume bar can drive
    ``ret / volume_ratio`` to extreme magnitudes, dominating the
    cross-sectional z-score and making the feature noise-dominated. Clipping
    keeps the conditioning monotone but bounded; the default symmetric range
    treats 10× under- and over-traded bars as the saturation point.

    Returns a dict keyed by horizon with sub-keys:
      ``"vol_exhaustion"``  — low-volume move reversal score
      ``"vol_blowoff"``     — high-volume move reversal score
      ``"volume_ratio"``    — clipped ratio for diagnostics (not a signal)
    """
    vol_window = int(volume_window_bars)
    volume_wide, = _align_to(close_wide, volume_wide)

    rolling_mean_vol = volume_wide.rolling(vol_window, min_periods=vol_window).mean()
    volume_ratio = volume_wide.div(rolling_mean_vol.replace(0.0, np.nan))
    lo, hi = float(volume_ratio_clip[0]), float(volume_ratio_clip[1])
    if not (lo > 0.0 and hi > lo):
        raise ValueError("volume_ratio_clip must satisfy 0 < lo < hi")
    volume_ratio = volume_ratio.clip(lower=lo, upper=hi)

    horizon_returns = compute_return_horizons(close_wide, horizons=horizons, log_returns=log_returns)
    out = {}
    for h, ret_wide in horizon_returns.items():
        low_vol_move = ret_wide.div(volume_ratio)
        high_vol_move = ret_wide.mul(volume_ratio)
        out[h] = {
            "vol_exhaustion": -low_vol_move,
            "vol_blowoff": -high_vol_move,
            "volume_ratio": volume_ratio,
        }
    return out


# ---------------------------------------------------------------------------
# 6. Distance from VWAP / Moving Average
# ---------------------------------------------------------------------------


def build_vwap_distance_features(
    close_wide,
    volume_wide,
    vwap_window_bars=24,
    ma_window_bars=24,
):
    """Distance of price from rolling VWAP and simple moving average.

    Rolling VWAP is computed as ``sum(close * volume) / sum(volume)`` over the
    window, giving a volume-weighted fair-value anchor.  Assets trading far
    above VWAP (positive distance) are expected to mean-revert downward; the
    features are negated so higher scores → long (underpriced vs anchor).

    Returns
    -------
    dict with keys:
      ``"distance_from_vwap"`` — negated (close - rolling_vwap) / close
      ``"distance_from_ma"``   — negated (close - rolling_ma) / close
    """
    vwap_w = int(vwap_window_bars)
    ma_w = int(ma_window_bars)
    volume_wide, = _align_to(close_wide, volume_wide)

    dollar_volume = close_wide.mul(volume_wide)
    rolling_dollar = dollar_volume.rolling(vwap_w, min_periods=vwap_w).sum()
    rolling_vol_sum = volume_wide.rolling(vwap_w, min_periods=vwap_w).sum().replace(0.0, np.nan)
    rolling_vwap = rolling_dollar.div(rolling_vol_sum)

    rolling_ma = close_wide.rolling(ma_w, min_periods=ma_w).mean()

    dist_vwap = close_wide.sub(rolling_vwap).div(close_wide.replace(0.0, np.nan))
    dist_ma = close_wide.sub(rolling_ma).div(close_wide.replace(0.0, np.nan))

    return {
        "distance_from_vwap": -dist_vwap,
        "distance_from_ma": -dist_ma,
    }


# ---------------------------------------------------------------------------
# 7. RSI Proxy
# ---------------------------------------------------------------------------


def build_rsi_proxy(
    close_wide,
    rsi_window_bars=14,
    log_returns=True,
):
    """RSI-style overbought/oversold indicator, centred on zero, negated.

    Classic RSI = 100 * avg_gain / (avg_gain + avg_loss), range [0, 100].
    Here we return ``0.5 - RSI/100`` so the output is centred on zero:
      * Positive → oversold (RSI < 50) → buy expected
      * Negative → overbought (RSI > 50) → sell expected

    Uses a simple rolling window (Wilder's EMA is a valid extension but adds
    look-ahead complexity for bar-by-bar research panels).
    """
    window = int(rsi_window_bars)
    bar_ret = compute_bar_returns(close_wide, log_returns=log_returns)

    gains = bar_ret.clip(lower=0.0)
    losses = (-bar_ret).clip(lower=0.0)

    avg_gain = gains.rolling(window, min_periods=window).mean()
    avg_loss = losses.rolling(window, min_periods=window).mean()

    # RSI = avg_gain / (avg_gain + avg_loss) * 100, equivalent to the textbook
    # 100 - 100/(1+RS) but numerically safer when avg_loss == 0 (flat losses
    # give RSI = 100). NaN is preserved during the rolling warm-up so downstream
    # IC / backtest code drops those rows rather than treating them as a real
    # "RSI = 50, no opinion" reading.
    avg_sum = avg_gain.add(avg_loss)
    rsi = avg_gain.div(avg_sum.replace(0.0, np.nan)) * 100.0

    return {"rsi_proxy": 0.5 - rsi / 100.0}


# ---------------------------------------------------------------------------
# 8. High-Low Range Position (Stochastic %K)
# ---------------------------------------------------------------------------


def build_range_position(
    close_wide,
    range_window_bars=12,
):
    """Position of close within its recent high-low range (%K stochastic).

    ``%K = (close - rolling_low) / (rolling_high - rolling_low)``
    Range [0, 1]: 1 = at top of range, 0 = at bottom.
    Negated and centred: higher score → near bottom of range → buy signal.

    Returns
    -------
    dict with keys:
      ``"range_position"`` — negated, centred (%K - 0.5)
    """
    window = int(range_window_bars)
    rolling_high = close_wide.rolling(window, min_periods=window).max()
    rolling_low = close_wide.rolling(window, min_periods=window).min()
    band = rolling_high.sub(rolling_low).replace(0.0, np.nan)
    pct_k = close_wide.sub(rolling_low).div(band)

    return {"range_position": -(pct_k - 0.5)}


# ---------------------------------------------------------------------------
# 9. Cross-Sectional Rank of Price Z-Score
# ---------------------------------------------------------------------------


def build_xsec_rank_of_price_z(
    close_wide,
    price_zscore_window_bars=24,
    min_assets_per_timestamp=6,
):
    """Cross-sectional relative deviation from rolling mean.

    Takes the time-series price z-score and re-ranks it cross-sectionally
    each bar (percentile rank, 0–1).  An asset may be the *most* stretched
    versus its own history, but the signal is strongest when it is also the
    most stretched *relative to its peers*.

    Negated so highest cross-sectional rank (most overbought vs peers) maps to
    the lowest signal score (short candidate).

    Timestamps with fewer than ``min_assets_per_timestamp`` valid price-z
    observations are NaN'd out to avoid spurious 2- or 3-asset rankings
    leaking into the IC or backtest (matches the convention enforced by
    ``cross_sectional_rank_or_zscore`` for the headline pipeline).

    Returns
    -------
    dict with keys:
      ``"xsec_rank_price_z"`` — negated cross-sectional percentile rank of
                                 price z-score
    """
    window = int(price_zscore_window_bars)
    rolling_mean = close_wide.rolling(window, min_periods=window).mean()
    rolling_std = close_wide.rolling(window, min_periods=window).std(ddof=0)
    price_z = close_wide.sub(rolling_mean).div(rolling_std.replace(0.0, np.nan))

    min_assets = int(min_assets_per_timestamp)
    xsec_rank = price_z.rank(axis=1, pct=True).where(price_z.notna())
    enough_assets = price_z.notna().sum(axis=1) >= min_assets
    xsec_rank = xsec_rank.where(enough_assets, np.nan)
    return {"xsec_rank_price_z": -xsec_rank}


# ---------------------------------------------------------------------------
# Composite signal (equal-weight across features)
# ---------------------------------------------------------------------------


def build_composite_signal(
    feature_panels,
    min_assets_per_timestamp=6,
    min_features=2,
):
    """Equal-weight composite of cross-sectionally z-scored feature panels.

    The honest alternative to quoting the best single feature: an argmax over
    correlated features is a max-statistic, while a combination rule fixed
    *a priori* (z-score each panel, average with equal weights) counts as ONE
    trial in multiple-testing accounting. Averaging correlated same-sign
    features also stabilises the signal — feature-level noise diversifies
    away while the common reversal component adds up.

    Parameters
    ----------
    feature_panels : dict[str, DataFrame] or list[DataFrame]
        Sign-normalised feature panels (higher = buy), already restricted to
        the tradable universe where applicable.
    min_assets_per_timestamp : int
        Passed to the per-panel cross-sectional z-score; rows with fewer
        valid names are NaN.
    min_features : int
        Cells averaged over fewer than this many valid feature values are
        NaN — a "composite" of one surviving feature is just that feature
        wearing a composite's label.

    Returns
    -------
    DataFrame — the composite raw panel. Not re-normalised: feed it through
    ``cross_sectional_rank_or_zscore`` for the final signal, exactly as with
    any single raw feature.
    """
    panels = (
        list(feature_panels.values())
        if isinstance(feature_panels, dict)
        else list(feature_panels)
    )
    if len(panels) == 0:
        raise ValueError("feature_panels is empty")
    min_features = int(min_features)
    if min_features < 1:
        raise ValueError("min_features must be >= 1")

    zscored = [
        cross_sectional_rank_or_zscore(
            panel,
            method="zscore",
            min_assets_per_timestamp=min_assets_per_timestamp,
        )
        for panel in panels
    ]

    index = zscored[0].index
    columns = zscored[0].columns
    for panel in zscored[1:]:
        index = index.union(panel.index)
        columns = columns.union(panel.columns)

    total = pd.DataFrame(0.0, index=index, columns=columns)
    count = pd.DataFrame(0, index=index, columns=columns)
    for panel in zscored:
        aligned = panel.reindex(index=index, columns=columns)
        total += aligned.fillna(0.0)
        count += aligned.notna().astype(int)

    return total.div(count.where(count >= min_features))


# ---------------------------------------------------------------------------
# Feature lookup
# ---------------------------------------------------------------------------

# Fixed-window features: name fragment -> settings key holding the lookback.
_FIXED_WINDOW_FEATURES = {
    "price_zscore": "price_zscore_window_bars",
    "bollinger": "bollinger_window_bars",
    "rsi": "rsi_window_bars",
    "range": "range_position_window_bars",
    "xsec_rank_price_z": "price_zscore_window_bars",
}


def mean_reversion_feature_lookback(name, settings):
    """Signal lookback in bars for a feature name, as used for NW lags.

    Horizon features (``*_h{n}``) return ``n``; fixed-window features return
    their window setting; anything else falls back to
    ``settings["reversal_lookback_bars"]``.
    """
    match = re.search(r"_h(\d+)$", name)
    if match:
        return int(match.group(1))
    for fragment, key in _FIXED_WINDOW_FEATURES.items():
        if fragment in name:
            return int(settings[key])
    return int(settings["reversal_lookback_bars"])


def get_mean_reversion_feature(name, feature_packs):
    """Look up one feature panel by its IC-table name.

    ``feature_packs`` maps family keys to builder outputs: ``reversal``,
    ``price_z``, ``vol_adj``, ``extreme``, ``vol_exh``, ``vwap``, ``rsi``,
    ``range``, ``xsec_z``.
    """
    match = re.search(r"_h(\d+)$", name)
    h = int(match.group(1)) if match else None
    horizon_features = [
        ("reversal_h", "reversal", "reversal"),
        ("xsec_reversal_h", "reversal", "xsec_reversal"),
        ("vol_adj_xsec_reversal_h", "vol_adj", "vol_adj_xsec_reversal"),
        ("vol_adj_reversal_h", "vol_adj", "vol_adj_reversal"),
        ("vol_exhaustion_h", "vol_exh", "vol_exhaustion"),
        ("vol_blowoff_h", "vol_exh", "vol_blowoff"),
    ]
    for prefix, pack, key in horizon_features:
        if name.startswith(prefix):
            return feature_packs[pack][h][key]
    fixed_features = {
        "price_zscore": ("price_z", "price_zscore"),
        "bollinger_touch": ("price_z", "bollinger_touch"),
        "extreme_signed_z": ("extreme", "signed_z"),
        "extreme_flag": ("extreme", "extreme_flag"),
        "distance_from_vwap": ("vwap", "distance_from_vwap"),
        "distance_from_ma": ("vwap", "distance_from_ma"),
        "rsi_proxy": ("rsi", "rsi_proxy"),
        "range_position": ("range", "range_position"),
        "xsec_rank_price_z": ("xsec_z", "xsec_rank_price_z"),
    }
    if name in fixed_features:
        pack, key = fixed_features[name]
        return feature_packs[pack][key]
    raise ValueError(f"Unknown feature: {name!r}")


def build_reversal_family_panel(
    close_wide,
    family,
    lookback_bars,
    vol_window_bars=14,
    log_returns=True,
    universe_mask=None,
):
    """One reversal-family feature at an arbitrary lookback (grid building).

    ``family`` is the feature name without its horizon suffix: ``reversal``,
    ``xsec_reversal``, ``vol_adj_reversal`` or ``vol_adj_xsec_reversal``.
    Cells outside ``universe_mask`` are set to NaN.
    """
    lookback = int(lookback_bars)
    if "vol_adj" in family:
        panels = build_vol_adjusted_move(
            close_wide, [lookback], vol_window_bars=vol_window_bars, log_returns=log_returns
        )
        key = "vol_adj_xsec_reversal" if "xsec" in family else "vol_adj_reversal"
    else:
        panels = build_return_reversal_features(
            close_wide, [lookback], log_returns=log_returns
        )
        key = "xsec_reversal" if family == "xsec_reversal" else "reversal"
    raw = panels[lookback][key]
    if universe_mask is not None:
        aligned = universe_mask.reindex(index=raw.index, columns=raw.columns).fillna(False)
        raw = raw.where(aligned)
    return raw


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def build_mean_reversion_signal(
    df_long,
    signal_timeframe,
    reversal_lookback_bars,
    universe_df=None,
    cross_sectional_transform="rank",
    min_assets_per_timestamp=6,
    log_returns=True,
    ic_rebalance_bars=1,
):
    """End-to-end mean-reversion signal from long OHLCV rows.

    The *raw signal* is the short-term return-reversal over
    ``reversal_lookback_bars``.  All other feature families are available
    separately via the individual builders above.

    Returns
    -------
    dict with keys:
      ``"close_wide"``    — resampled close price panel (ts × symbol)
      ``"volume_wide"``   — resampled volume panel (ts × symbol, sum)
      ``"return_wide"``   — bar-by-bar log returns
      ``"raw_signal"``    — negated rolling return (pre-transform)
      ``"signal"``        — normalised signal, decimated by `ic_rebalance_bars`
                            (IC analysis only)
      ``"signal_fresh"``  — normalised signal pre-decimation (use in backtests;
                            trading cadence there is `holding_period_bars`)
      ``"universe_mask"`` — boolean ts × symbol mask (or None)
    """
    close_wide = resample_to_signal_timeframe(df_long, signal_timeframe)
    volume_wide = _pivot_volume(df_long, signal_timeframe)
    volume_wide = volume_wide.reindex(index=close_wide.index, columns=close_wide.columns)

    ret_wide = compute_bar_returns(close_wide, log_returns=log_returns)

    universe_mask = build_monthly_universe_mask(
        index=close_wide.index,
        columns=close_wide.columns,
        universe_df=universe_df,
    )

    reversal_features = build_return_reversal_features(
        close_wide,
        horizons=[reversal_lookback_bars],
        log_returns=log_returns,
        universe_mask=universe_mask,
    )
    raw_signal = reversal_features[reversal_lookback_bars]["reversal"]

    if universe_mask is not None:
        raw_signal = raw_signal.where(universe_mask)

    signal_fresh = cross_sectional_rank_or_zscore(
        raw_signal,
        method=cross_sectional_transform,
        min_assets_per_timestamp=min_assets_per_timestamp,
    )
    signal = apply_rebalance_decimation(signal_fresh, ic_rebalance_bars)

    return {
        "close_wide": close_wide,
        "volume_wide": volume_wide,
        "return_wide": ret_wide,
        "raw_signal": raw_signal,
        "signal": signal,
        "signal_fresh": signal_fresh,
        "universe_mask": universe_mask,
    }

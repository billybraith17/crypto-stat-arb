"""Cross-sectional momentum research transforms."""

import numpy as np
import pandas as pd


def _validate_long_panel(df):
    required = {"ts", "symbol", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        return
    keys = df[["ts", "symbol"]].copy()
    keys["ts"] = pd.to_datetime(keys["ts"], utc=True)
    dup_mask = keys.duplicated(keep=False)
    if dup_mask.any():
        n_dup_rows = int(dup_mask.sum())
        sample = keys.loc[dup_mask].drop_duplicates().head(10)
        raise ValueError(
            "duplicate (ts, symbol) rows found: pivot requires exactly one "
            f"close per timestamp per symbol ({n_dup_rows} rows involved). "
            "Example (ts, symbol) pairs:\n"
            f"{sample.to_string(index=False)}"
        )


def resample_to_signal_timeframe(df_long, signal_timeframe):
    """Resample hourly close to a configurable signal timeframe."""
    _validate_long_panel(df_long)
    panel = df_long[["ts", "symbol", "close"]].copy()
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)

    close_wide = panel.pivot(index="ts", columns="symbol", values="close").sort_index()
    # For close-to-close signals, last observed close in each bar is appropriate.
    close_wide = close_wide.resample(signal_timeframe).last()
    return close_wide.dropna(how="all")


def compute_bar_returns(close_wide, log_returns=True):
    """Compute close-to-close returns per symbol."""
    if log_returns:
        return np.log(close_wide / close_wide.shift(1))
    return close_wide.pct_change()


def compute_return_horizons(close_wide, horizons, log_returns=True, skip_bars=0):
    """Compute return panels for each horizon in `horizons` bars.

    Returns are measured from ``t - skip_bars - horizon`` to ``t - skip_bars``.
    With ``skip_bars=0`` this reduces to standard trailing horizon returns.
    """
    skip = int(skip_bars)
    if skip < 0:
        raise ValueError("skip_bars must be >= 0")
    horizon_returns = {}
    for horizon in horizons:
        h = int(horizon)
        if h <= 0:
            raise ValueError("All horizons must be > 0")
        if log_returns:
            horizon_returns[h] = np.log(
                close_wide.shift(skip) / close_wide.shift(skip + h)
            )
        else:
            horizon_returns[h] = close_wide.shift(skip).div(
                close_wide.shift(skip + h)
            ).sub(1.0)
    return horizon_returns


def rolling_momentum_score(return_wide, lookback_bars, log_returns=True, skip_bars=0):
    """Aggregate past returns over lookback bars with optional skip window."""
    lookback_bars = int(lookback_bars)
    if lookback_bars <= 0:
        raise ValueError("lookback_bars must be > 0")
    skip = int(skip_bars)
    if skip < 0:
        raise ValueError("skip_bars must be >= 0")
    working = return_wide.shift(skip)
    if log_returns:
        return working.rolling(lookback_bars, min_periods=lookback_bars).sum()
    return (1.0 + working).rolling(
        lookback_bars, min_periods=lookback_bars
    ).apply(np.prod, raw=True) - 1.0


def build_relative_momentum_features(
    return_wide,
    benchmark_symbol="XBT/USD",
    residual_space="log",
    log_returns=True,
    universe_mask=None,
):
    """Create relative/residual variants from a return panel.

    Parameters
    ----------
    residual_space : {"log", "simple"}
        ``"log"`` (default): subtract in the same units as ``return_wide``.
        When ``log_returns`` is True, residuals are log-return differences
        (interpretable as log gross-return ratios vs benchmark).

        ``"simple"``: convert log inputs to simple returns with ``exp(r)-1``,
        then subtract cross-sectional mean or benchmark simple return. Use when
        residuals must align with per-period additive P&L of long-short books.

    log_returns : bool
        Whether ``return_wide`` is log returns (True) or simple returns (False).
        Ignored when ``residual_space`` is ``"log"`` except that simple inputs
        must use ``log_returns=False``.

    universe_mask : pd.DataFrame or None
        If provided, boolean panel aligned to ``return_wide`` (after any space
        transform). For ``minus_xsec_mean``, the cross-sectional average each
        bar is taken only over symbols where the mask is True; the residual is
        still returned for every column in ``return_wide``. When None, the mean
        is over all non-NaN symbols in the row (previous behavior).
    """
    if benchmark_symbol not in return_wide.columns:
        sample = list(return_wide.columns)[:6]
        raise KeyError(
            f"benchmark_symbol '{benchmark_symbol}' not found in panel columns "
            f"(sample: {sample}). Check that the ticker matches the database "
            f"(Kraken stores Bitcoin as XBT/USD, not BTC/USD)."
        )
    space = str(residual_space).lower()
    if space not in ("log", "simple"):
        raise ValueError("residual_space must be 'log' or 'simple'")

    if space == "log":
        working = return_wide
    else:
        if log_returns:
            working = np.exp(return_wide) - 1.0
        else:
            working = return_wide

    if universe_mask is not None:
        mask = universe_mask.reindex(index=working.index, columns=working.columns)
        mask = mask.fillna(False).astype(bool)
        xsec_mean = working.where(mask).mean(axis=1)
    else:
        xsec_mean = working.mean(axis=1)
    return {
        "minus_xsec_mean": working.sub(xsec_mean, axis=0),
        "minus_benchmark": working.sub(working[benchmark_symbol], axis=0),
    }


def build_vol_adjusted_features(return_wide, vol_window_bars=24):
    """Build volatility-adjusted and time-series z-scored returns."""
    window = int(vol_window_bars)
    if window <= 1:
        raise ValueError("vol_window_bars must be > 1")

    rolling_vol = return_wide.rolling(window, min_periods=window).std(ddof=0)
    rolling_mean = return_wide.rolling(window, min_periods=window).mean()

    # Return-over-vol (no demean) vs textbook z-score (demean then scale): distinct.
    vol_adjusted = return_wide.div(rolling_vol.replace(0.0, np.nan))
    z_scored = return_wide.sub(rolling_mean).div(rolling_vol.replace(0.0, np.nan))
    return {
        "rolling_vol": rolling_vol,
        "return_over_vol": vol_adjusted,
        "zscored_return": z_scored,
    }


def build_monthly_universe_mask(index, columns, universe_df):
    """Build a timestamp x symbol boolean mask from monthly universe rows."""
    if universe_df is None:
        return None
    if len(universe_df) == 0:
        return pd.DataFrame(False, index=index, columns=columns)
    required = {"rebalance_date", "symbol"}
    missing = required.difference(universe_df.columns)
    if missing:
        raise ValueError(f"universe_df missing required columns: {sorted(missing)}")

    members = universe_df[["rebalance_date", "symbol"]].dropna().copy()
    if members.empty:
        return pd.DataFrame(False, index=index, columns=columns)
    # Universe is ranked on month-end but becomes tradable at next midnight UTC.
    members["effective_ts"] = pd.to_datetime(members["rebalance_date"], utc=True) + pd.Timedelta(
        days=1
    )
    # Build explicit monthly snapshots: at each rebalance date, non-members are False.
    membership_points = pd.crosstab(members["effective_ts"], members["symbol"]).astype(bool)
    membership_points = membership_points.sort_index()
    membership_points = membership_points.reindex(columns=columns, fill_value=False)

    mask = membership_points.reindex(index, method="ffill")
    mask = mask.astype("boolean").fillna(False)
    return mask.astype(bool)


def cross_sectional_rank_or_zscore(signal_wide, method="zscore", min_assets_per_timestamp=6):
    """Transform raw signal into comparable cross-sectional scores."""
    method = str(method).lower()
    min_assets = int(min_assets_per_timestamp)
    out = signal_wide.copy()
    valid_counts = out.notna().sum(axis=1)
    enough_assets = valid_counts >= min_assets

    if method == "rank":
        ranks = out.rank(axis=1, pct=True).where(signal_wide.notna())
        return ranks.where(enough_assets, np.nan)

    if method == "zscore":
        mean = out.mean(axis=1)
        std = out.std(axis=1, ddof=0)
        z = out.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)
        return z.where(enough_assets, np.nan)

    if method == "none":
        return out.where(enough_assets, np.nan)

    raise ValueError("method must be one of: zscore, rank, none")


def apply_rebalance_decimation(signal_wide, every_n_bars):
    """Hold the signal across multiple signal bars between rebalances.

    Samples the signal at every `every_n_bars`th row and forward-fills the held
    value to the intermediate rows. With `every_n_bars=1` (the default in
    `build_momentum_signal`), rebalance cadence is locked to `signal_timeframe`
    and this is a no-op. Larger values let you study turnover sensitivity, e.g.
    `every_n_bars=3` at a 4h signal cadence rebalances every 12h.
    """
    n = int(every_n_bars)
    if n <= 1 or signal_wide.empty:
        return signal_wide
    sampled = signal_wide.iloc[::n]
    return sampled.reindex(signal_wide.index, method="ffill")


def compute_forward_returns(close_wide, holding_period_bars, log_returns=True):
    """Compute realized forward returns over configurable holding bars."""
    steps = int(holding_period_bars)
    if steps <= 0:
        raise ValueError("holding_period_bars must be > 0")
    if log_returns:
        return np.log(close_wide.shift(-steps) / close_wide)
    return close_wide.shift(-steps).div(close_wide).sub(1.0)


def build_feature_panels(
    close_wide,
    horizons,
    benchmark_symbol="XBT/USD",
    vol_window_bars=24,
    log_returns=True,
    residual_space="log",
    universe_mask=None,
    skip_bars=0,
):
    """Build core, relative, and volatility-adjusted momentum feature families."""
    core_returns = compute_return_horizons(
        close_wide, horizons=horizons, log_returns=log_returns, skip_bars=skip_bars
    )
    relative_features = {}
    vol_adjusted_features = {}
    for horizon, ret_wide in core_returns.items():
        relative_features[horizon] = build_relative_momentum_features(
            ret_wide,
            benchmark_symbol=benchmark_symbol,
            residual_space=residual_space,
            log_returns=log_returns,
            universe_mask=universe_mask,
        )
        vol_adjusted_features[horizon] = build_vol_adjusted_features(
            ret_wide, vol_window_bars=vol_window_bars
        )
    return {
        "core_returns": core_returns,
        "relative_features": relative_features,
        "vol_adjusted_features": vol_adjusted_features,
    }


def apply_universe_mask(panel, universe_mask):
    """Restrict a panel to the tradable universe when a mask is provided."""
    if universe_mask is None:
        return panel
    return panel.where(universe_mask)


def select_momentum_feature_panel(
    feature_pack,
    baseline_signal,
    available_horizons,
    feature_family="baseline",
    feature_horizon=None,
    relative_feature="minus_xsec_mean",
    vol_feature="zscored_return",
):
    """Select one momentum feature panel (or baseline signal) for diagnostics."""
    family = str(feature_family).lower()
    if family == "baseline":
        return baseline_signal, "baseline_signal"

    if feature_horizon is None:
        raise ValueError("feature_horizon is required when feature_family != 'baseline'")
    h = int(feature_horizon)
    horizon_set = {int(x) for x in available_horizons}
    if h not in horizon_set:
        raise ValueError(
            f"feature_horizon={h} not in available_horizons={sorted(horizon_set)}"
        )

    if family == "core":
        return feature_pack["core_returns"][h], f"core_ret_{h}bar"

    if family == "relative":
        rel_panels = feature_pack["relative_features"][h]
        if relative_feature not in rel_panels:
            raise ValueError(
                "relative_feature must be one of "
                f"{list(rel_panels.keys())}, got {relative_feature!r}"
            )
        return rel_panels[relative_feature], f"{relative_feature}_{h}bar"

    if family == "vol_adjusted":
        vol_panels = feature_pack["vol_adjusted_features"][h]
        if vol_feature not in vol_panels:
            raise ValueError(
                "vol_feature must be one of "
                f"{list(vol_panels.keys())}, got {vol_feature!r}"
            )
        return vol_panels[vol_feature], f"{vol_feature}_{h}bar"

    raise ValueError(
        "feature_family must be one of: baseline, core, relative, vol_adjusted"
    )


def build_selected_momentum_signal(
    raw_panel,
    universe_mask=None,
    cross_sectional_transform="zscore",
    min_assets_per_timestamp=6,
    rebalance_every_n_bars=1,
    apply_xsec_transform=True,
    apply_rebalance_decimation_flag=True,
):
    """Build final signal from a selected raw panel using notebook-style options."""
    signal = apply_universe_mask(raw_panel, universe_mask)
    if apply_xsec_transform:
        signal = cross_sectional_rank_or_zscore(
            signal,
            method=cross_sectional_transform,
            min_assets_per_timestamp=min_assets_per_timestamp,
        )
    if apply_rebalance_decimation_flag:
        signal = apply_rebalance_decimation(signal, rebalance_every_n_bars)
    return signal


def build_momentum_signal(
    df_long,
    signal_timeframe,
    momentum_lookback_bars,
    momentum_skip_bars=0,
    universe_df=None,
    cross_sectional_transform="zscore",
    min_assets_per_timestamp=6,
    log_returns=True,
    rebalance_every_n_bars=1,
):
    """End-to-end signal construction from long OHLCV rows.

    By default the signal is produced and rebalanced at the cadence given by
    `signal_timeframe` (one signal per bar, executed at that bar's close).
    Set `rebalance_every_n_bars` > 1 to rebalance less often than the signal
    cadence and study the turnover/cost trade-off.
    """
    close_wide = resample_to_signal_timeframe(df_long, signal_timeframe)
    ret_wide = compute_bar_returns(close_wide, log_returns=log_returns)
    raw_signal = rolling_momentum_score(
        ret_wide,
        momentum_lookback_bars,
        log_returns=log_returns,
        skip_bars=momentum_skip_bars,
    )
    universe_mask = build_monthly_universe_mask(
        index=raw_signal.index,
        columns=raw_signal.columns,
        universe_df=universe_df,
    )
    if universe_mask is not None:
        raw_signal = raw_signal.where(universe_mask)
    signal = cross_sectional_rank_or_zscore(
        raw_signal,
        method=cross_sectional_transform,
        min_assets_per_timestamp=min_assets_per_timestamp,
    )
    signal = apply_rebalance_decimation(signal, rebalance_every_n_bars)

    return {
        "close_wide": close_wide,
        "return_wide": ret_wide,
        "raw_signal": raw_signal,
        "signal": signal,
        "universe_mask": universe_mask,
    }

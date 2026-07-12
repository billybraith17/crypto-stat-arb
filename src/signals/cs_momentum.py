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


def build_traded_mask(df_long, signal_timeframe):
    """Boolean ts × symbol panel: True where the bar contains >= 1 real trade.

    The hourly loader forward-fills no-trade hours (``volume = 0``,
    ``trades = 0``), so closes on those bars are stale copies of an older
    print. Returns computed across a stale endpoint are fictitious and bias
    short-horizon reversal upward (a stale asset shows 0% while peers move,
    ranks as a "loser", then mechanically catches up — Lo-MacKinlay
    non-synchronous trading). At the signal timeframe a bar counts as traded
    if any underlying hourly row had ``trades > 0``; bins with no rows at all
    (before listing / after delisting) are False.
    """
    if "trades" not in df_long.columns:
        raise ValueError(
            "df_long must include a 'trades' column to build the traded mask "
            "(fetch_ohlcv_long returns it)"
        )
    panel = df_long[["ts", "symbol", "trades"]].copy()
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    trades_wide = panel.pivot(index="ts", columns="symbol", values="trades").sort_index()
    trades_sum = trades_wide.resample(signal_timeframe).sum(min_count=1)
    return trades_sum.gt(0.0) & trades_sum.notna()


def apply_traded_mask(panel, traded_mask, lookback_bars=0, forward_bars=0):
    """NaN out cells whose defining price endpoints fall on stale bars.

    A cell ``(t, s)`` survives only if symbol ``s`` traded at bar ``t``, at
    ``t - lookback_bars`` (when > 0), and at ``t + forward_bars`` (when > 0).
    Use ``lookback_bars=R`` for an R-bar feature (both return endpoints must
    be real prints) and ``forward_bars=H`` for H-bar forward returns (entry
    and exit prints must be real). Intermediate bars are not required to
    trade — endpoint returns are unaffected by gaps inside the window.
    """
    lookback = int(lookback_bars)
    forward = int(forward_bars)
    if lookback < 0 or forward < 0:
        raise ValueError("lookback_bars and forward_bars must be >= 0")
    aligned = traded_mask.reindex(
        index=panel.index, columns=panel.columns, fill_value=False
    ).astype(bool)
    valid = aligned
    if lookback > 0:
        valid = valid & aligned.shift(lookback, fill_value=False)
    if forward > 0:
        valid = valid & aligned.shift(-forward, fill_value=False)
    return panel.where(valid)


def compute_return_horizons(close_wide, horizons, log_returns=True, skip_bars=0):
    """Compute return panels for each horizon in `horizons` bars.

    Returns are measured from ``t - skip_bars - horizon`` to ``t - skip_bars``.
    With ``skip_bars=0`` this reduces to standard trailing horizon returns.

    ``skip_bars`` is part of the *alpha definition* (Jegadeesh-Titman skip:
    exclude the most recent bars' returns because of short-term reversal), not
    an execution delay — that is `execution_delay_bars` in the backtest, or
    the minute-level delay in `src.research.execution`. Note the side effect:
    with ``skip_bars >= 1`` the feature at bar ``t`` only uses closes up to
    ``t - skip_bars``, so it is fully computable at least one bar before an
    execution at close(t).
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


def build_market_index_returns(
    return_wide,
    mode="equal_weight",
    benchmark_symbol="XBT/USD",
    universe_mask=None,
    min_assets=1,
):
    """Market return series used for beta estimation and residualization.

    ``mode="equal_weight"`` (default): per-bar mean of ``return_wide`` across
    universe members (all non-NaN symbols when ``universe_mask`` is None);
    bars with fewer than ``min_assets`` valid members are NaN. Pass a return
    panel already NaN-masked for non-traded bars (``apply_traded_mask``) so
    stale forward-filled prints do not drag the index toward zero. Note the
    ~1/N self-inclusion bias: each asset is one N-th of its own regressor,
    pulling its estimated beta toward 1 by roughly 1/N; a leave-one-out index
    is a possible refinement, not implemented here.

    ``mode="benchmark"``: the benchmark symbol's own return column. Beware the
    downstream degeneracy: the benchmark's residual return is then identically
    ~0, parking it mid-rank in every residual-momentum feature.
    """
    mode = str(mode).lower()
    if mode not in ("equal_weight", "benchmark"):
        raise ValueError("mode must be 'equal_weight' or 'benchmark'")
    if int(min_assets) < 1:
        raise ValueError("min_assets must be >= 1")

    if mode == "benchmark":
        if benchmark_symbol not in return_wide.columns:
            sample = list(return_wide.columns)[:6]
            raise KeyError(
                f"benchmark_symbol '{benchmark_symbol}' not found in panel columns "
                f"(sample: {sample}). Check that the ticker matches the database "
                f"(Kraken stores Bitcoin as XBT/USD, not BTC/USD)."
            )
        return return_wide[benchmark_symbol].rename("market_ret")

    working = return_wide
    if universe_mask is not None:
        mask = universe_mask.reindex(index=working.index, columns=working.columns)
        mask = mask.fillna(False).astype(bool)
        working = working.where(mask)
    n_valid = working.notna().sum(axis=1)
    market = working.mean(axis=1).where(n_valid >= int(min_assets))
    return market.rename("market_ret")


def estimate_rolling_betas(
    return_wide,
    market_returns,
    window_bars=90,
    min_periods=None,
    shrinkage=0.0,
    shrink_target=1.0,
):
    """Trailing beta of each symbol against a market return series.

    ``beta_raw[t, s] = Cov_w(r_s, m) / Var_w(m)`` over the trailing
    ``window_bars`` bars ending at ``t`` inclusive — the same
    "uses data through t only" convention as ``rolling_momentum_score``, so a
    beta at bar ``t`` is legitimate input to a weight executed at close(t).
    Windows where the market variance is 0 return NaN (not inf). NaN returns
    (e.g. traded-mask holes, pre-listing history) are excluded pairwise: the
    market variance for symbol ``s`` is computed only over bars where ``s``
    itself is valid, so covariance and variance always share the same sample
    (otherwise a hole would bias the ratio). ``min_periods`` defaults to the
    full window (strict).

    ``shrinkage`` pulls the raw estimate linearly toward ``shrink_target``:
    ``beta = (1 - shrinkage) * beta_raw + shrinkage * shrink_target``. With
    ~90 observations per window the raw estimator is noisy; shrinking toward
    1 (liquid crypto is roughly a unit-beta asset class) trades a small bias
    for a much lower-variance hedge ratio.
    """
    window = int(window_bars)
    if window <= 1:
        raise ValueError("window_bars must be > 1")
    mp = window if min_periods is None else int(min_periods)
    if mp < 2 or mp > window:
        raise ValueError("min_periods must be in [2, window_bars]")
    lam = float(shrinkage)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")

    market = market_returns.reindex(return_wide.index)
    cov = return_wide.rolling(window, min_periods=mp).cov(market)
    # Per-symbol market variance over the pairwise-valid sample only.
    market_masked = pd.DataFrame(
        np.where(return_wide.notna(), market.to_numpy()[:, None], np.nan),
        index=return_wide.index,
        columns=return_wide.columns,
    )
    var = market_masked.rolling(window, min_periods=mp).var()
    beta_raw = cov / var.replace(0.0, np.nan)
    return (1.0 - lam) * beta_raw + lam * float(shrink_target)


def build_residual_return_panel(return_wide, betas, market_returns):
    """Beta-residualized returns: ``eps[t, s] = r[t, s] - beta[t, s] * m[t]``.

    NaN in the return, the beta, or the market return propagates to NaN, so
    beta warm-up rows and traded-mask holes drop out of downstream
    ``min_periods`` windows instead of being silently treated as zero.
    """
    betas_aligned = betas.reindex(index=return_wide.index, columns=return_wide.columns)
    market = market_returns.reindex(return_wide.index)
    return return_wide - betas_aligned.mul(market, axis=0)


def build_residual_momentum_features(residual_return_wide, horizons, skip_bars=0):
    """Residual-momentum family (Blitz/Huij/Martens 2011) per horizon.

    For each horizon ``h``, with the same Jegadeesh-Titman ``skip_bars``
    convention as ``rolling_momentum_score``:

    - ``residual_momentum``: sum of the trailing ``h`` residual bar returns.
    - ``residual_momentum_scaled``: that sum divided by the rolling std
      (population, ``ddof=0``) of the same residuals — the standardized
      variant that carries most of the documented improvement in equities.
      The per-asset scaling reweights assets even though a cross-sectional
      transform follows, because it changes relative magnitudes within a row.
    """
    skip = int(skip_bars)
    if skip < 0:
        raise ValueError("skip_bars must be >= 0")
    features = {}
    for horizon in horizons:
        h = int(horizon)
        if h <= 0:
            raise ValueError("All horizons must be > 0")
        working = residual_return_wide.shift(skip)
        resid_sum = working.rolling(h, min_periods=h).sum()
        resid_std = working.rolling(h, min_periods=h).std(ddof=0)
        features[h] = {
            "residual_momentum": resid_sum,
            "residual_momentum_scaled": resid_sum.div(resid_std.replace(0.0, np.nan)),
        }
    return features


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
    value to the intermediate rows. With `every_n_bars=1` this is a no-op.

    This is an **IC-analysis device only**: it answers "what does the signal I
    would actually be holding predict?" at every bar, with the induced overlap
    absorbed by a Newey-West lag of ``max(H, every_n_bars) - 1``. It must NOT
    feed backtests — `run_light_backtest` implements trading cadence itself by
    stepping the book every `holding_period_bars`, and feeding it a decimated
    signal layers two cadence mechanisms with phase-dependent staleness.
    Backtests take the pre-decimation signal (``signal_fresh`` in the pipeline
    outputs).
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
    beta_panel=None,
    market_returns=None,
):
    """Build core, relative, and volatility-adjusted momentum feature families.

    When ``beta_panel`` and ``market_returns`` are both provided (they must be
    passed together), a fourth ``"residual_features"`` family is added:
    momentum on beta-residualized *bar* returns
    (``build_residual_return_panel`` + ``build_residual_momentum_features``),
    generalizing the static beta=1 ``minus_benchmark`` residual to
    estimated-beta residuals. Estimate the betas and market index on
    traded-masked returns (``apply_traded_mask``) so stale prints don't bias
    them; the residual formation itself uses raw bar returns from
    ``close_wide``, consistent with the other families. When neither is
    passed, the output is identical to the three-family version.
    """
    if (beta_panel is None) != (market_returns is None):
        raise ValueError("beta_panel and market_returns must be provided together")

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
    feature_pack = {
        "core_returns": core_returns,
        "relative_features": relative_features,
        "vol_adjusted_features": vol_adjusted_features,
    }
    if beta_panel is not None:
        bar_returns = compute_bar_returns(close_wide, log_returns=log_returns)
        residual_returns = build_residual_return_panel(
            bar_returns, betas=beta_panel, market_returns=market_returns
        )
        feature_pack["residual_features"] = build_residual_momentum_features(
            residual_returns, horizons=horizons, skip_bars=skip_bars
        )
    return feature_pack


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
    residual_feature="residual_momentum_scaled",
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

    if family == "residual":
        if "residual_features" not in feature_pack:
            raise ValueError(
                "feature_pack has no 'residual_features' — rebuild it with "
                "build_feature_panels(..., beta_panel=..., market_returns=...)"
            )
        resid_panels = feature_pack["residual_features"][h]
        if residual_feature not in resid_panels:
            raise ValueError(
                "residual_feature must be one of "
                f"{list(resid_panels.keys())}, got {residual_feature!r}"
            )
        return resid_panels[residual_feature], f"{residual_feature}_{h}bar"

    raise ValueError(
        "feature_family must be one of: baseline, core, relative, vol_adjusted, residual"
    )


def build_selected_momentum_signal(
    raw_panel,
    universe_mask=None,
    cross_sectional_transform="zscore",
    min_assets_per_timestamp=6,
    ic_rebalance_bars=1,
    apply_xsec_transform=True,
    apply_rebalance_decimation_flag=True,
):
    """Build final signal from a selected raw panel using notebook-style options.

    The decimated output is for IC analysis; backtests should use the
    pre-decimation signal (``apply_rebalance_decimation_flag=False``).
    """
    signal = apply_universe_mask(raw_panel, universe_mask)
    if apply_xsec_transform:
        signal = cross_sectional_rank_or_zscore(
            signal,
            method=cross_sectional_transform,
            min_assets_per_timestamp=min_assets_per_timestamp,
        )
    if apply_rebalance_decimation_flag:
        signal = apply_rebalance_decimation(signal, ic_rebalance_bars)
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
    ic_rebalance_bars=1,
):
    """End-to-end signal construction from long OHLCV rows.

    ``"signal"`` in the output is decimated by `ic_rebalance_bars` for IC
    analysis ("what does the held signal predict?"); ``"signal_fresh"`` is the
    same signal pre-decimation and is what backtests should consume — trading
    cadence there is `holding_period_bars`, not decimation.
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
    signal_fresh = cross_sectional_rank_or_zscore(
        raw_signal,
        method=cross_sectional_transform,
        min_assets_per_timestamp=min_assets_per_timestamp,
    )
    signal = apply_rebalance_decimation(signal_fresh, ic_rebalance_bars)

    return {
        "close_wide": close_wide,
        "return_wide": ret_wide,
        "raw_signal": raw_signal,
        "signal": signal,
        "signal_fresh": signal_fresh,
        "universe_mask": universe_mask,
    }

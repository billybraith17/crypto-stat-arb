"""Reusable evaluation helpers for momentum research notebooks."""

import warnings

import numpy as np
import pandas as pd


def _infer_periods_per_year(index, irregular_warn_ratio=0.1):
    """Infer periods-per-year from a DatetimeIndex by median spacing.

    Emits a UserWarning if the spacing looks materially irregular (mean and
    median diverge by more than ``irregular_warn_ratio``), in which case the
    caller should pass an explicit ``periods_per_year``.
    """
    if len(index) < 2:
        return 365.0
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return 365.0
    median_step = diffs.median()
    if pd.isna(median_step) or median_step <= pd.Timedelta(0):
        return 365.0
    mean_step = diffs.mean()
    if mean_step > pd.Timedelta(0):
        ratio = abs((mean_step - median_step) / median_step)
        if ratio > irregular_warn_ratio:
            warnings.warn(
                "Index spacing looks irregular "
                f"(median={median_step}, mean={mean_step}); the inferred "
                "periods_per_year may misannualize statistics. Pass an "
                "explicit periods_per_year to override.",
                UserWarning,
                stacklevel=2,
            )
    return pd.Timedelta(days=365) / median_step


def _newey_west_se(x, lag):
    """Newey-West (Bartlett) standard error of the sample mean.

    For a series with ``lag`` periods of overlap, this inflates the iid-style
    standard error to account for serial correlation up to that lag.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return np.nan
    x = x - x.mean()
    L = max(int(lag), 0)
    L = min(L, n - 1)
    var = float(np.dot(x, x) / n)
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        cov_k = float(np.dot(x[k:], x[:-k]) / n)
        var += 2.0 * w * cov_k
    if not np.isfinite(var) or var <= 0:
        return np.nan
    return float(np.sqrt(var / n))


def compute_ic_series(signal_wide, forward_return_wide, method="spearman", min_assets=6):
    """Compute cross-sectional IC at each timestamp.

    Groups with fewer than ``min_assets`` valid ``(signal, fwd_ret)`` pairs are
    returned as NaN so that the IC sample is aligned with the tradable universe
    (see ``min_assets_per_timestamp`` in ``cross_sectional_rank_or_zscore``).
    Timestamps where either side has zero variance (e.g. a fully-tied rank row,
    or the benchmark column of a ``minus_benchmark`` panel left alone after
    masking) are also returned as NaN, which suppresses scipy's
    ``ConstantInputWarning`` for genuinely undefined correlations.
    """
    joined = pd.concat(
        [signal_wide.stack().rename("signal"), forward_return_wide.stack().rename("fwd_ret")],
        axis=1,
    ).dropna()
    if joined.empty:
        return pd.Series(dtype=float)

    min_assets = int(min_assets)
    if min_assets < 2:
        raise ValueError("min_assets must be >= 2 (Spearman/Pearson need >=2 points)")

    def _corr(group):
        if len(group) < min_assets:
            return np.nan
        signal = group["signal"]
        fwd = group["fwd_ret"]
        if signal.nunique(dropna=True) < 2 or fwd.nunique(dropna=True) < 2:
            return np.nan
        return signal.corr(fwd, method=method)

    return joined.groupby(level=0).apply(_corr)


def ic_summary(ic_series, nw_lag=0):
    """Return standard IC diagnostics including t-stat.

    Parameters
    ----------
    nw_lag : int
        Newey-West (Bartlett) lag truncation for the t-stat. Default 0 keeps
        the iid t-stat (mean / iid_se). Pass ``H - 1`` when the IC series is
        computed bar-by-bar but the forward returns span ``H`` bars (the IC
        observations then share ``H - 1`` bars of overlap and the iid t-stat
        is inflated). When the signal is held across ``D`` bars between
        rebalances, use ``max(H, D) - 1`` to absorb both effects.
    """
    clean = ic_series.dropna()
    n = len(clean)
    nw_lag = max(int(nw_lag), 0)
    if n == 0:
        return {
            "n_obs": 0,
            "mean_ic": np.nan,
            "median_ic": np.nan,
            "std_ic": np.nan,
            "t_stat_ic": np.nan,
            "t_stat_ic_nw": np.nan,
            "nw_lag": nw_lag,
            "p05_ic": np.nan,
            "p95_ic": np.nan,
        }
    std = clean.std(ddof=1)
    mean = clean.mean()
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else np.nan
    if nw_lag > 0:
        nw_se = _newey_west_se(clean.values, lag=nw_lag)
        t_stat_nw = (mean / nw_se) if (nw_se and nw_se > 0) else np.nan
    else:
        t_stat_nw = t_stat
    return {
        "n_obs": int(n),
        "mean_ic": float(mean),
        "median_ic": float(clean.median()),
        "std_ic": float(std),
        "t_stat_ic": float(t_stat) if not pd.isna(t_stat) else np.nan,
        "t_stat_ic_nw": float(t_stat_nw) if not pd.isna(t_stat_nw) else np.nan,
        "nw_lag": nw_lag,
        "p05_ic": float(clean.quantile(0.05)),
        "p95_ic": float(clean.quantile(0.95)),
    }


def compute_ic_decay_heatmap(
    candidate_signals: dict,
    close_wide: pd.DataFrame,
    holding_period_grid: list,
    log_returns: bool = True,
    min_assets: int = 6,
    feature_horizons: dict = None,
) -> tuple:
    """
    Compute NW-adjusted IC t-stats for each (signal, holding_period) pair.

    Parameters
    ----------
    candidate_signals : dict mapping signal name -> wide signal panel (T x N)
    close_wide        : wide close price panel used to compute forward returns
    holding_period_grid : list of integer holding periods (in bars) to evaluate
    log_returns       : whether to compute log forward returns
    min_assets        : passed to compute_ic_series
    feature_horizons  : dict mapping signal name -> feature lookback (bars). For
                        each (signal, h) pair, NW lag = max(feature_horizons[signal], h) - 1,
                        matching the convention used in the feature IC table. Defaults
                        to 1 for any signal not present in the dict.

    Returns
    -------
    Tuple of (t_stat_df, mean_ic_df): DataFrames with signals as rows,
    holding periods as columns. t_stat_df contains NW-adjusted IC t-stats;
    mean_ic_df contains the corresponding mean IC values.
    """
    _horizons = feature_horizons or {}
    tstat_records = {}
    mean_ic_records = {}

    for h in holding_period_grid:
        h = int(h)
        if log_returns:
            fwd_ret = np.log(close_wide.shift(-h) / close_wide)
        else:
            fwd_ret = close_wide.shift(-h).div(close_wide).sub(1.0)

        col_tstats = {}
        col_mean_ic = {}
        for sig_name, sig_panel in candidate_signals.items():
            nw_lag = max(int(_horizons.get(sig_name, 1)), h) - 1
            ic = compute_ic_series(sig_panel, fwd_ret, min_assets=min_assets)
            stats = ic_summary(ic, nw_lag=nw_lag)
            col_tstats[sig_name] = stats["t_stat_ic_nw"]
            col_mean_ic[sig_name] = stats["mean_ic"]

        tstat_records[h] = col_tstats
        mean_ic_records[h] = col_mean_ic

    tstat_df = pd.DataFrame(tstat_records)
    tstat_df.index.name = "signal"
    tstat_df.columns.name = "holding_period_bars"

    mean_ic_df = pd.DataFrame(mean_ic_records)
    mean_ic_df.index.name = "signal"
    mean_ic_df.columns.name = "holding_period_bars"

    return tstat_df, mean_ic_df


def plot_ic_decay_heatmap(t_stat_df: pd.DataFrame, mean_ic_df: pd.DataFrame = None) -> None:
    """Render IC decay heatmap + decay curves and print summary table."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    nan_mask = t_stat_df.isna()
    annot = t_stat_df.copy().astype(object)
    for col in annot.columns:
        annot[col] = [f"{v:.1f}" if pd.notna(v) else "" for v in annot[col]]
    lim = t_stat_df.abs().max().max()
    lim = 1.0 if pd.isna(lim) or lim == 0 else float(lim)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    sns.heatmap(
        t_stat_df, ax=ax1, cmap="RdBu_r", center=0, vmin=-lim, vmax=lim,
        annot=annot, fmt="", mask=nan_mask, linewidths=0.5,
        cbar_kws={"label": "NW t-stat"},
    )
    ax1.set_title(
        "IC Decay Heatmap — NW t-stat by Signal and Holding Period\n"
        "(cells with |t| < 2 are statistically weak)"
    )
    ax1.set_xlabel("Holding Period (bars)")
    ax1.set_ylabel("Signal")

    for sig in t_stat_df.index:
        ax2.plot(t_stat_df.columns, t_stat_df.loc[sig], marker="o", markersize=4, label=sig)
    ax2.axhline(2,  color="gray", linestyle="--", linewidth=1, alpha=0.8, label="|t|=2")
    ax2.axhline(-2, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    ax2.set_title("Signal IC Decay Curves")
    ax2.set_xlabel("Holding Period (bars)")
    ax2.set_ylabel("NW t-stat")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    grid_max = int(t_stat_df.columns.max())
    print(f"{'Signal':<25} | {'Peak hold':>9} | {'Peak t':>8} | {'t @ h=1':>8} | Zero-cross hold")
    print("-" * 72)
    for sig in t_stat_df.index:
        v = t_stat_df.loc[sig].dropna()
        if v.empty:
            continue
        ph = int(v.idxmax())
        t1 = float(t_stat_df.loc[sig, 1]) if 1 in t_stat_df.columns else float("nan")
        zc = f">max ({grid_max})"
        items = list(v.items())
        for i, (h, ts) in enumerate(items):
            if ts < 1.0:
                if i == 0:
                    zc = f"~{h}"
                else:
                    hp, tp = items[i - 1]
                    zc = f"~{hp + (tp - 1.0) / (tp - ts) * (h - hp):.0f}" if tp != ts else f"~{h}"
                break
        print(
            f"{sig:<25} | {ph:>9} | {float(v[ph]):>8.2f}"
            f" | {f'{t1:.2f}' if pd.notna(t1) else 'N/A':>8} | {zc}"
        )


def rolling_mean(series, window):
    """Simple rolling mean helper for notebook reporting."""
    return series.rolling(int(window), min_periods=int(0.9*window)).mean()


def rolling_sharpe(returns, window, periods_per_year):
    """Compute rolling Sharpe on simple returns."""
    window = int(window)
    roll_mean = returns.rolling(window, min_periods=window).mean()
    roll_std = returns.rolling(window, min_periods=window).std(ddof=1)
    ann = np.sqrt(periods_per_year)
    return ann * roll_mean.div(roll_std.replace(0.0, np.nan))


def max_drawdown(cumulative_returns):
    """Compute max drawdown from cumulative return series."""
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns.div(running_max).sub(1.0)
    return float(drawdown.min()), drawdown


def build_quantile_weights(signal_wide, top_quantile=0.2, bottom_quantile=0.2):
    """Create equal-weight long/short sleeves based on cross-sectional quantiles."""
    top_q = float(top_quantile)
    bottom_q = float(bottom_quantile)
    if not (0.0 < top_q < 1.0 and 0.0 < bottom_q < 1.0):
        raise ValueError("top_quantile and bottom_quantile must be in (0, 1)")

    ranks = signal_wide.rank(axis=1, pct=True)

    long_mask = ranks >= (1.0 - top_q)
    short_mask = ranks <= bottom_q

    long_count = long_mask.sum(axis=1).replace(0, np.nan)
    short_count = short_mask.sum(axis=1).replace(0, np.nan)

    long_weights = long_mask.div(long_count, axis=0).fillna(0.0)
    short_weights = short_mask.div(short_count, axis=0).fillna(0.0)
    return long_weights - short_weights


def run_light_backtest(
    signal_wide,
    forward_return_wide,
    top_quantile=0.2,
    bottom_quantile=0.2,
    fee_bps=2.0,
    half_spread_bps=1.0,
    execution_delay_bars=0,
    returns_are_log=False,
    holding_period_bars=1,
    periods_per_year=None,
):
    """Backtest equal-weight top/bottom quantile portfolio with simple costs.

    The book steps in ``holding_period_bars`` increments to avoid double-
    counting overlapping H-bar forward returns. With H=1 the behavior is
    identical to bar-by-bar trading; with H>1 only every H-th observation
    contributes a P&L period (single-sleeve, non-overlapping). Bars at the
    start (before signal coverage) and end (after the last priceable forward
    return) are dropped, so cumulative series and Sharpe are computed only on
    the priceable subset. ``annualized_turnover`` is reported alongside the
    per-period mean for interpretability.
    """
    H = max(int(holding_period_bars), 1)

    weights = build_quantile_weights(
        signal_wide, top_quantile=top_quantile, bottom_quantile=bottom_quantile
    )
    delay = int(execution_delay_bars)
    if delay > 0:
        weights = weights.shift(delay)

    realized_ret = forward_return_wide
    if returns_are_log:
        realized_ret = np.exp(forward_return_wide) - 1.0

    if H > 1:
        step_idx = weights.index[::H]
        weights_step = weights.reindex(step_idx)
        realized_step = realized_ret.reindex(step_idx)
    else:
        weights_step = weights
        realized_step = realized_ret

    # Only count rows where every position we hold has a valid realized return,
    # so we never partially price a basket. Rows with no positions stay valid
    # (they produce a 0 P&L until the signal kicks in, then get trimmed below).
    positioned = weights_step.fillna(0.0) != 0.0
    unpriceable = positioned & realized_step.isna()
    bar_unpriceable = unpriceable.any(axis=1)

    gross_ret = (weights_step * realized_step).sum(axis=1, min_count=1)
    gross_ret = gross_ret.where(~bar_unpriceable, np.nan)

    turnover = 0.5 * weights_step.fillna(0.0).diff().abs().sum(axis=1)
    cost_rate = (float(fee_bps) + float(half_spread_bps)) * 1e-4
    costs = turnover * cost_rate
    net_ret = gross_ret - costs

    # Trim to the priceable holding window: from the first bar where weights
    # take a real position through the last bar with a valid forward return.
    has_position = positioned.any(axis=1)
    first_idx = has_position.idxmax() if has_position.any() else None
    valid = net_ret.notna()
    if first_idx is not None:
        valid &= net_ret.index >= first_idx
    gross_ret = gross_ret[valid]
    net_ret = net_ret[valid]
    turnover = turnover[valid]
    costs = costs[valid]

    cum_gross = (1.0 + gross_ret).cumprod()
    cum_net = (1.0 + net_ret).cumprod()

    if periods_per_year is None:
        ppy = _infer_periods_per_year(net_ret.index)
    else:
        ppy = float(periods_per_year)
    sd = net_ret.std(ddof=1)
    sharpe = np.sqrt(ppy) * net_ret.mean() / sd if (sd is not None and sd > 0) else np.nan
    mdd, dd_series = max_drawdown(cum_net)

    return {
        "weights": weights_step,
        "gross_returns": gross_ret,
        "net_returns": net_ret,
        "turnover": turnover,
        "costs": costs,
        "cum_gross": cum_gross,
        "cum_net": cum_net,
        "drawdown": dd_series,
        "metrics": {
            "periods_per_year": float(ppy),
            "sharpe_net": float(sharpe) if not pd.isna(sharpe) else np.nan,
            "max_drawdown_net": float(mdd),
            "mean_turnover": float(turnover.mean()) if len(turnover) else np.nan,
            "annualized_turnover": (
                float(turnover.mean() * ppy) if len(turnover) else np.nan
            ),
            "mean_cost_per_period": float(costs.mean()) if len(costs) else np.nan,
            "step_bars": int(H),
            "n_periods": int(len(net_ret)),
        },
    }


def quantile_analysis(signal_wide, forward_return_wide, n_quantiles=5, returns_are_log=False):
    """Compute quantile return diagnostics and top-minus-bottom spread.
    """
    n_quantiles = int(n_quantiles)
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be >= 2")

    if returns_are_log:
        forward_return_wide = np.exp(forward_return_wide) - 1.0

    stacked = pd.concat(
        [signal_wide.stack().rename("signal"), forward_return_wide.stack().rename("fwd_ret")],
        axis=1,
    ).dropna()
    if stacked.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    def _bucket(group):
        out = group.copy()
        out["quantile"] = (
            pd.qcut(group["signal"], q=n_quantiles, labels=False, duplicates="drop") + 1
        )
        return out

    # Group by the timestamp index level (not a column) so pandas doesn't emit
    # the "operated on the grouping columns" warning, and the index shape is
    # preserved by apply.
    with_buckets = (
        stacked.groupby(level=0, group_keys=False)
        .apply(_bucket)
        .dropna(subset=["quantile"])
    )
    with_buckets["quantile"] = with_buckets["quantile"].astype(int)

    quantile_means = (
        with_buckets.groupby("quantile")["fwd_ret"]
        .mean()
        .rename("mean_fwd_ret")
        .reset_index()
    )

    # Pivot via reset_index so we don't rely on get_level_values shape, which
    # has historically been brittle across pandas versions.
    flat = with_buckets.reset_index()
    ts_col = flat.columns[0]
    spread_ts = flat.pivot_table(
        index=ts_col,
        columns="quantile",
        values="fwd_ret",
        aggfunc="mean",
    )
    top = (
        spread_ts[n_quantiles]
        if n_quantiles in spread_ts.columns
        else pd.Series(index=spread_ts.index, dtype=float)
    )
    bot = (
        spread_ts[1]
        if 1 in spread_ts.columns
        else pd.Series(index=spread_ts.index, dtype=float)
    )
    spread = (top - bot).rename("q_top_minus_bottom")
    return quantile_means, spread


def long_short_leg_returns(weights, forward_return_wide, returns_are_log=False):
    """Return separate long and short leg returns for decomposition.
    """
    realized_ret = forward_return_wide
    if returns_are_log:
        realized_ret = np.exp(forward_return_wide) - 1.0

    long_w = weights.clip(lower=0.0)
    short_w = weights.clip(upper=0.0)
    long_ret = (long_w * realized_ret).sum(axis=1, min_count=1).fillna(0.0)
    short_ret = (short_w * realized_ret).sum(axis=1, min_count=1).fillna(0.0)
    return long_ret.rename("long_leg_ret"), short_ret.rename("short_leg_ret")


def market_correlation(series, benchmark):
    """Correlation helper with automatic alignment."""
    aligned = pd.concat([series.rename("x"), benchmark.rename("y")], axis=1).dropna()
    if aligned.empty:
        return np.nan
    return float(aligned["x"].corr(aligned["y"]))

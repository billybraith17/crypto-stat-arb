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


def feature_ic_row(
    feature_wide,
    forward_return_wide,
    nw_lag=0,
    min_assets=6,
    tradable_forward_return_wide=None,
):
    """IC summary of one feature panel, optionally against tradable returns.

    Returns the ``ic_summary`` dict for ``forward_return_wide`` (the raw
    predictive-power reference). When ``tradable_forward_return_wide`` is
    given (e.g. execution-priced returns), adds ``mean_ic_trad``,
    ``t_nw_trad`` and ``ic_haircut`` (reference minus tradable mean IC).
    """
    row = ic_summary(
        compute_ic_series(feature_wide, forward_return_wide, min_assets=min_assets),
        nw_lag=nw_lag,
    )
    if tradable_forward_return_wide is not None:
        trad = ic_summary(
            compute_ic_series(
                feature_wide, tradable_forward_return_wide, min_assets=min_assets
            ),
            nw_lag=nw_lag,
        )
        row["mean_ic_trad"] = trad["mean_ic"]
        row["t_nw_trad"] = trad["t_stat_ic_nw"]
        row["ic_haircut"] = row["mean_ic"] - trad["mean_ic"]
    return row


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


def build_banded_quantile_weights(
    signal_wide,
    top_quantile=0.2,
    bottom_quantile=0.2,
    band=0.0,
):
    """Quantile long/short weights with hysteresis (no-trade) bands.

    ``build_quantile_weights`` re-selects the book from scratch every row, so
    a name oscillating around the quantile boundary is churned in and out at
    full cost for no new information. Here membership is *stateful*: a name
    ENTERS the long book when its cross-sectional percentile rank reaches
    ``1 - top_quantile``, but EXITS only when it falls below
    ``1 - top_quantile - band`` (symmetrically for shorts). With ``band=0``
    the entry and exit thresholds coincide and the output reproduces
    ``build_quantile_weights`` exactly (see NaN note below). Held names are
    equal-weighted per side each row.

    Cadence: rows are processed in the order given, at whatever frequency the
    input has. Pass the signal at *decision frequency* (e.g. the H-stepped
    rows the backtest will actually rebalance on), then reindex/ffill the
    result to the full index for ``run_light_backtest(weights_wide=...)`` —
    state evolving at a faster cadence than the rebalance would let names
    exit and re-enter invisibly between rebalances.

    NaN handling: a name whose signal is NaN on a row is forced out (no basis
    to hold it — e.g. it left the tradable universe); a row where the entire
    signal is NaN keeps the previous row's weights (no information is not a
    liquidation event — this is the one divergence from
    ``build_quantile_weights``, which returns a flat row there). A name whose
    long-stay and short-entry conditions hold simultaneously (possible only
    for extreme ``band``) is dropped from both sides that row.
    """
    top_q = float(top_quantile)
    bottom_q = float(bottom_quantile)
    band = float(band)
    if not (0.0 < top_q < 1.0 and 0.0 < bottom_q < 1.0):
        raise ValueError("top_quantile and bottom_quantile must be in (0, 1)")
    if band < 0.0:
        raise ValueError("band must be >= 0")

    ranks = signal_wide.rank(axis=1, pct=True).to_numpy()
    n_rows, n_cols = ranks.shape
    long_enter, long_exit = 1.0 - top_q, 1.0 - top_q - band
    short_enter, short_exit = bottom_q, bottom_q + band

    weights = np.zeros((n_rows, n_cols))
    long_state = np.zeros(n_cols, dtype=bool)
    short_state = np.zeros(n_cols, dtype=bool)
    for i in range(n_rows):
        row = ranks[i]
        valid = ~np.isnan(row)
        if not valid.any():
            if i > 0:
                weights[i] = weights[i - 1]
            continue
        long_state = valid & ((row >= long_enter) | (long_state & (row >= long_exit)))
        short_state = valid & ((row <= short_enter) | (short_state & (row <= short_exit)))
        ambiguous = long_state & short_state
        long_state &= ~ambiguous
        short_state &= ~ambiguous
        n_long = int(long_state.sum())
        n_short = int(short_state.sum())
        if n_long:
            weights[i, long_state] = 1.0 / n_long
        if n_short:
            weights[i, short_state] -= 1.0 / n_short

    return pd.DataFrame(weights, index=signal_wide.index, columns=signal_wide.columns)


def build_banded_book(
    signal_wide,
    holding_period_bars,
    band,
    top_quantile=0.2,
    bottom_quantile=0.2,
):
    """Banded quantile book evolved at decision frequency, on the full index.

    Membership state updates only on every ``holding_period_bars``-th row (the
    rows ``run_light_backtest`` rebalances on, both starting at row 0) and is
    forward-filled in between, so names cannot exit and re-enter invisibly
    between rebalances. Pass the result as ``weights_wide``.
    """
    H = max(int(holding_period_bars), 1)
    decision = build_banded_quantile_weights(
        signal_wide.iloc[::H],
        top_quantile=top_quantile,
        bottom_quantile=bottom_quantile,
        band=band,
    )
    return decision.reindex(signal_wide.index, method="ffill").fillna(0.0)


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
    weights_wide=None,
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

    ``holding_period_bars`` is the single trading-cadence mechanism: pass the
    pre-decimation signal (decimation via ``apply_rebalance_decimation`` is an
    IC-analysis device and layering it under the H-stepping here produces
    phase-dependent staleness). ``execution_delay_bars`` shifts weights by
    whole signal bars — implementation lag, distinct from a Jegadeesh-Titman
    ``skip_bars`` which is part of the feature definition. For sub-bar delays,
    embed the delay in the forward-return panel instead
    (``src.research.execution``) and keep ``execution_delay_bars=0``.

    Rebalances are netted at a single print: the old book exits and the new
    book enters at the same execution close. ``fee_bps`` and
    ``half_spread_bps`` are one-way rates paid on every fill, so costs are
    charged on the full traded notional ``sum(|dw|)``. Reported turnover is
    one-sided, ``0.5 * sum(|dw|)`` (the buy or sell side of the rebalance).
    Weight drift within the holding period is ignored (light backtest).

    ``half_spread_bps`` may be a scalar (flat spread for all assets) or a
    per-symbol ``pd.Series`` in bps (e.g. from
    ``src.research.spreads.estimate_half_spread_bps_from_long``), in which
    case each asset's traded notional is charged at its own rate — flat spreads
    understate costs on the illiquid tail of the universe. Symbols missing
    from the Series (or NaN) are filled with the cross-sectional median of
    the supplied values; pass explicit values to override.

    ``weights_wide`` bypasses the internal quantile construction and backtests
    an externally built book (e.g. ``build_beta_hedged_weights`` output) under
    the identical delay / H-stepping / turnover / cost accounting —
    ``signal_wide`` may then be None and ``top_quantile``/``bottom_quantile``
    are ignored. Pass weights indexed like the signal would be: the same
    ``execution_delay_bars`` shift is applied.
    """
    H = max(int(holding_period_bars), 1)

    if weights_wide is not None:
        weights = weights_wide
    elif signal_wide is not None:
        weights = build_quantile_weights(
            signal_wide, top_quantile=top_quantile, bottom_quantile=bottom_quantile
        )
    else:
        raise ValueError("pass either signal_wide or weights_wide")
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

    # Every unit of |dw| is one fill paying the one-way fee plus half-spread.
    traded_by_asset = weights_step.fillna(0.0).diff().abs()
    traded = traded_by_asset.sum(axis=1)
    turnover = 0.5 * traded
    if isinstance(half_spread_bps, pd.Series):
        spread_by_symbol = half_spread_bps.reindex(weights_step.columns)
        spread_by_symbol = spread_by_symbol.fillna(half_spread_bps.median())
        rate_by_symbol = (float(fee_bps) + spread_by_symbol) * 1e-4
        costs = traded_by_asset.mul(rate_by_symbol, axis=1).sum(axis=1)
    else:
        cost_rate = (float(fee_bps) + float(half_spread_bps)) * 1e-4
        costs = traded * cost_rate
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
    gross_exposure = weights_step.fillna(0.0).abs().sum(axis=1)[valid]

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
            "mean_gross_exposure": (
                float(gross_exposure.mean()) if len(gross_exposure) else np.nan
            ),
            "step_bars": int(H),
            "n_periods": int(len(net_ret)),
        },
    }


def gross_sharpe(backtest):
    """Annualised Sharpe of a ``run_light_backtest`` result's gross returns."""
    gross = backtest["gross_returns"].dropna()
    sd = gross.std(ddof=1)
    if not sd > 0:
        return np.nan
    return float(np.sqrt(backtest["metrics"]["periods_per_year"]) * gross.mean() / sd)


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


# ---------------------------------------------------------------------------
# Market-beta exposure and hedging
# ---------------------------------------------------------------------------


def portfolio_beta_series(weights, betas):
    """Ex-ante portfolio beta and how much of the book it is measured on.

    ``portfolio_beta[t] = sum_s w[t, s] * beta[t, s]`` over positions with a
    valid beta; ``beta_coverage[t]`` is the share of gross exposure
    ``sum_s |w[t, s]|`` those positions represent. Coverage < 1 means the
    beta (and any hedge sized from it) ignores part of the book — surface it
    instead of letting missing betas silently read as "no exposure".

    Dollar neutrality (``sum w = 0``) does not imply ``portfolio_beta = 0``:
    equal-weight long/short sleeves are beta-neutral only if both sleeves
    carry the same average beta, which a momentum sort systematically
    violates (winners in a rally are the high-beta names).
    """
    betas_aligned = betas.reindex(index=weights.index, columns=weights.columns)
    portfolio_beta = weights.mul(betas_aligned).sum(axis=1, min_count=1)
    abs_w = weights.abs()
    gross = abs_w.sum(axis=1)
    covered = abs_w.where(betas_aligned.notna(), 0.0).sum(axis=1)
    coverage = covered.div(gross.replace(0.0, np.nan))
    return pd.DataFrame(
        {"portfolio_beta": portfolio_beta, "beta_coverage": coverage}
    )


def realized_beta_diagnostics(
    strategy_returns,
    market_returns,
    rolling_window=20,
    min_regime_obs=10,
):
    """Realized beta of strategy P&L vs the market, full-sample and by regime.

    Returns a dict with the full-sample beta/correlation, up-market and
    down-market betas (``Cov/Var`` over the ``m > 0`` / ``m < 0`` subsamples,
    NaN when a regime has fewer than ``min_regime_obs`` points, with counts
    always reported), and a rolling-beta series. A dollar-neutral book can
    print a full-sample correlation near 0 while carrying large
    opposite-signed betas in up and down regimes — the regime split is the
    statistic that exposes it. On stepped H-bar P&L the sample is small
    (correlation SE ~ 1/sqrt(n)); read point estimates alongside ``n_obs``.
    """
    aligned = pd.concat(
        [strategy_returns.rename("s"), market_returns.rename("m")], axis=1
    ).dropna()

    def _beta(frame):
        if len(frame) < 2:
            return np.nan
        var = frame["m"].var()
        if not np.isfinite(var) or var == 0:
            return np.nan
        return float(frame["s"].cov(frame["m"]) / var)

    up = aligned[aligned["m"] > 0]
    down = aligned[aligned["m"] < 0]
    min_regime = int(min_regime_obs)
    window = int(rolling_window)
    rolling_beta = (
        aligned["s"].rolling(window, min_periods=window).cov(aligned["m"])
        / aligned["m"].rolling(window, min_periods=window).var().replace(0.0, np.nan)
    )
    return {
        "n_obs": int(len(aligned)),
        "beta_full": _beta(aligned),
        "corr_full": (
            float(aligned["s"].corr(aligned["m"])) if len(aligned) >= 2 else np.nan
        ),
        "n_up": int(len(up)),
        "n_down": int(len(down)),
        "beta_up": _beta(up) if len(up) >= min_regime else np.nan,
        "beta_down": _beta(down) if len(down) >= min_regime else np.nan,
        "rolling_beta": rolling_beta.rename("rolling_beta"),
    }


def build_beta_hedged_weights(
    weights,
    betas,
    benchmark_symbol="XBT/USD",
    missing_beta_fill=1.0,
    max_hedge_weight=None,
    normalize_gross=True,
):
    """Overlay a benchmark position that cancels the ex-ante portfolio beta.

    ``hedge[t] = -portfolio_beta[t] / beta_benchmark[t]`` is **added** to the
    ``benchmark_symbol`` column (created if absent), so an existing benchmark
    quantile position and the hedge net into a single position — one turnover
    charge, which is the correct portfolio accounting. The result is
    beta-neutral instead of dollar-neutral (row sums are no longer 0).

    With ``normalize_gross=True`` each hedged row is rescaled to its pre-hedge
    gross exposure, so hedged and unhedged books deploy identical capital and
    drawdowns/returns compare like-for-like; the per-row scalar preserves the
    zero ex-ante beta. ``normalize_gross=False`` keeps the raw overlay, whose
    gross is the pre-hedge gross plus ``|hedge|``.

    Held positions with a missing beta are filled with ``missing_beta_fill``
    (default 1.0 — the sane crypto prior). This fill is deliberate, not
    silent: check ``beta_coverage`` from ``portfolio_beta_series`` before
    trusting the hedge. ``max_hedge_weight`` optionally clips the hedge to
    ``±max_hedge_weight`` (beta noise at short windows can size absurd
    hedges).

    Pass betas estimated on simple returns, matching the arithmetic P&L the
    hedge offsets.
    """
    fill = float(missing_beta_fill)
    betas_aligned = betas.reindex(index=weights.index, columns=weights.columns)
    held = weights.fillna(0.0) != 0.0
    betas_filled = betas_aligned.where(~held | betas_aligned.notna(), fill)
    beta_p = weights.mul(betas_filled).sum(axis=1, min_count=1)

    if benchmark_symbol in betas.columns:
        beta_bench = betas[benchmark_symbol].reindex(weights.index)
    else:
        beta_bench = pd.Series(np.nan, index=weights.index)
    beta_bench = beta_bench.fillna(fill).replace(0.0, np.nan)

    hedge = -beta_p.div(beta_bench)
    if max_hedge_weight is not None:
        cap = float(max_hedge_weight)
        if cap <= 0:
            raise ValueError("max_hedge_weight must be > 0")
        hedge = hedge.clip(-cap, cap)

    hedged = weights.copy()
    if benchmark_symbol not in hedged.columns:
        hedged[benchmark_symbol] = 0.0
    hedged[benchmark_symbol] = hedged[benchmark_symbol].fillna(0.0) + hedge.fillna(0.0)
    if normalize_gross:
        gross_before = weights.abs().sum(axis=1)
        gross_after = hedged.abs().sum(axis=1)
        scale = gross_before.div(gross_after.replace(0.0, np.nan)).fillna(1.0)
        hedged = hedged.mul(scale, axis=0)
    return hedged


# ---------------------------------------------------------------------------
# Selection-bias corrections (multiple testing)
# ---------------------------------------------------------------------------

_EULER_GAMMA = 0.5772156649015329


def max_over_trials_pvalue(t_stat, n_trials):
    """P(max of ``n_trials`` independent N(0,1) draws >= ``t_stat``).

    The honest null for "the best of N features/configurations has t = X" is
    the maximum of N t-stats, not a single one: ``1 - Phi(t)^N``. Features in
    a grid are correlated, so the true trial count lies between 1 and N —
    bracket the p-value by calling this with both, or pass an effective count
    (e.g. the number of distinct feature families).
    """
    from scipy.stats import norm

    n = int(n_trials)
    if n < 1:
        raise ValueError("n_trials must be >= 1")
    if pd.isna(t_stat):
        return np.nan
    return float(1.0 - norm.cdf(t_stat) ** n)


def expected_max_sharpe(n_trials, var_sharpe, mean_sharpe=0.0):
    """Expected maximum Sharpe across ``n_trials`` under the no-skill null.

    Bailey & Lopez de Prado (2014): if trial Sharpes are ~N(mean, var), the
    expected maximum of N trials is approximately::

        mean + sd * ((1 - gamma) * Z(1 - 1/N) + gamma * Z(1 - 1/(N*e)))

    with ``gamma`` the Euler-Mascheroni constant and ``Z`` the standard normal
    quantile. Units are whatever the inputs use — pass *per-period* (non-
    annualized) Sharpes and the cross-trial variance of those same Sharpes.
    """
    from scipy.stats import norm

    n = int(n_trials)
    if n < 1:
        raise ValueError("n_trials must be >= 1")
    if n == 1:
        return float(mean_sharpe)
    sd = float(np.sqrt(var_sharpe))
    return float(
        mean_sharpe
        + sd
        * (
            (1.0 - _EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n)
            + _EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n * np.e))
        )
    )


def probabilistic_sharpe_ratio(sharpe, benchmark_sharpe, n_obs, skew=0.0, kurt=3.0):
    """P(true Sharpe > ``benchmark_sharpe``) given an observed per-period Sharpe.

    Standard PSR (Bailey & Lopez de Prado): the sampling error of the Sharpe
    estimator is widened for skewed / fat-tailed returns via ``skew`` and
    ``kurt`` (Pearson kurtosis, normal = 3). All Sharpes per-period.
    """
    from scipy.stats import norm

    n = int(n_obs)
    if n < 2:
        return np.nan
    var_term = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2
    if not np.isfinite(var_term) or var_term <= 0:
        return np.nan
    z = (sharpe - benchmark_sharpe) * np.sqrt(n - 1.0) / np.sqrt(var_term)
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    sharpe,
    n_obs,
    n_trials,
    var_sharpe,
    skew=0.0,
    kurt=3.0,
    mean_sharpe=0.0,
):
    """Deflated Sharpe ratio: PSR against the expected max of N no-skill trials.

    Converts "the best configuration in the grid has Sharpe X" into the
    probability that its true Sharpe exceeds what pure selection over
    ``n_trials`` correlated-noise trials would deliver. Values near 0.5 mean
    the winner is indistinguishable from the luckiest of N noise strategies;
    conventionally require >= 0.95.

    Parameters use *per-period* units: divide annualized Sharpes by
    ``sqrt(periods_per_year)``, take ``var_sharpe`` as the cross-trial
    variance of those per-period Sharpes, and ``skew``/``kurt`` from the
    winning strategy's per-period returns.
    """
    sr_star = expected_max_sharpe(n_trials, var_sharpe, mean_sharpe=mean_sharpe)
    dsr = probabilistic_sharpe_ratio(sharpe, sr_star, n_obs, skew=skew, kurt=kurt)
    return {
        "expected_max_sharpe": sr_star,
        "deflated_sharpe_prob": dsr,
        "n_trials": int(n_trials),
        "n_obs": int(n_obs),
    }


# ---------------------------------------------------------------------------
# Walk-forward evaluation with embargo
# ---------------------------------------------------------------------------


def walk_forward_splits(index, n_folds=5, embargo_obs=0):
    """Split an ordered index into contiguous folds with an embargo.

    Overlapping signal-lookback / forward-return windows make observations
    near a fold boundary share data with the previous fold, so the first
    ``embargo_obs`` observations of every fold after the first are dropped.
    ``embargo_obs`` is in units of *observations of the series being split* —
    if the IC series is subsampled every D bars, convert bar-overlap
    ``max(H, R) - 1`` to ``(max(H, R) - 1) // D``.

    Returns a list of ``{"fold", "test_index"}`` dicts (folds are 1-based).
    """
    n_folds = int(n_folds)
    embargo = int(embargo_obs)
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if embargo < 0:
        raise ValueError("embargo_obs must be >= 0")
    n = len(index)
    if n < n_folds:
        raise ValueError(f"index has {n} observations, fewer than n_folds={n_folds}")

    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    splits = []
    for k in range(n_folds):
        start, stop = edges[k], edges[k + 1]
        if k > 0:
            start = min(start + embargo, stop)
        splits.append({"fold": k + 1, "test_index": index[start:stop]})
    return splits


def walk_forward_ic_table(ic_series, n_folds=5, embargo_obs=0, nw_lag=0):
    """Per-fold IC stability table for a fixed (pre-selected) signal.

    Splits the IC series into ``n_folds`` contiguous blocks (embargoed as in
    ``walk_forward_splits``) and reports ``ic_summary`` per block. A robust
    signal shows same-sign mean IC of similar magnitude across folds; a
    regime artifact shows one or two dominant folds.
    """
    clean = ic_series.dropna()
    rows = []
    for split in walk_forward_splits(clean.index, n_folds=n_folds, embargo_obs=embargo_obs):
        block = clean.loc[split["test_index"]]
        stats = ic_summary(block, nw_lag=nw_lag)
        rows.append(
            {
                "fold": split["fold"],
                "start": block.index.min() if len(block) else pd.NaT,
                "end": block.index.max() if len(block) else pd.NaT,
                "n_obs": stats["n_obs"],
                "mean_ic": stats["mean_ic"],
                "t_stat_ic_nw": stats["t_stat_ic_nw"],
            }
        )
    return pd.DataFrame(rows).set_index("fold")


def walk_forward_sharpe_table(returns, periods_per_year, n_folds=5, embargo_obs=0):
    """Per-fold annualised Sharpe of a fixed book's period returns.

    Folds and embargo follow ``walk_forward_splits``; ``embargo_obs`` is in
    units of the return series (e.g. rebalance periods).
    """
    ppy = float(periods_per_year)
    rows = []
    for split in walk_forward_splits(returns.index, n_folds=n_folds, embargo_obs=embargo_obs):
        block = returns.loc[split["test_index"]].dropna()
        sd = block.std(ddof=1)
        rows.append(
            {
                "fold": split["fold"],
                "start": block.index.min(),
                "end": block.index.max(),
                "n_obs": len(block),
                "sharpe_net": (
                    float(np.sqrt(ppy) * block.mean() / sd) if (sd and sd > 0) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).set_index("fold")


def walk_forward_selection(
    ic_series_by_config,
    n_folds=5,
    embargo_obs=0,
    nw_lag_by_config=None,
):
    """Honest grid evaluation: select on past folds, score on the next fold.

    For each test fold ``k >= 2``, the configuration with the best mean IC
    over *all observations before the fold* (minus the embargo tail) is
    selected, and its IC over fold ``k`` is recorded. The pooled
    out-of-sample series therefore never uses data that influenced the
    selection — unlike sorting a robustness grid by full-sample IC.

    Parameters
    ----------
    ic_series_by_config : dict[label, pd.Series]
        Precomputed IC series per configuration (as produced inside the
        robustness-grid loops).
    nw_lag_by_config : dict[label, int] or None
        NW lag per configuration for the pooled summary; the maximum over
        the selected configurations is used (conservative). None -> 0.

    Returns
    -------
    (selection_table, pooled_stats): per-fold DataFrame with the selected
    config and its in-selection vs out-of-sample mean IC, and an
    ``ic_summary`` dict over the concatenated out-of-sample ICs.
    """
    if not ic_series_by_config:
        raise ValueError("ic_series_by_config is empty")
    panel = pd.DataFrame({k: v for k, v in ic_series_by_config.items()}).sort_index()

    splits = walk_forward_splits(panel.index, n_folds=n_folds, embargo_obs=embargo_obs)
    embargo = int(embargo_obs)

    rows = []
    oos_segments = []
    used_configs = set()
    for split in splits[1:]:
        test_idx = split["test_index"]
        if len(test_idx) == 0:
            continue
        fold_start_pos = panel.index.get_loc(test_idx[0])
        sel_stop = max(fold_start_pos - embargo, 0)
        selection_window = panel.iloc[:sel_stop]
        sel_means = selection_window.mean()
        if sel_means.isna().all():
            continue
        best = sel_means.idxmax()
        used_configs.add(best)
        oos_block = panel.loc[test_idx, best].dropna()
        oos_segments.append(oos_block)
        rows.append(
            {
                "fold": split["fold"],
                "selected_config": best,
                "selection_mean_ic": float(sel_means[best]),
                "oos_mean_ic": float(oos_block.mean()) if len(oos_block) else np.nan,
                "oos_n_obs": int(len(oos_block)),
            }
        )

    selection_table = pd.DataFrame(rows).set_index("fold") if rows else pd.DataFrame()
    pooled = pd.concat(oos_segments) if oos_segments else pd.Series(dtype=float)
    if nw_lag_by_config:
        pooled_lag = max(int(nw_lag_by_config.get(c, 0)) for c in used_configs) if used_configs else 0
    else:
        pooled_lag = 0
    pooled_stats = ic_summary(pooled, nw_lag=pooled_lag)
    return selection_table, pooled_stats

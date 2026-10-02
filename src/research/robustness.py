"""Robustness-grid evaluation shared by the momentum and mean-reversion studies.

Each grid cell is scored the same way: IC on the decimation grid with a
Newey-West lag covering forward-return and lookback overlap, a train/test IC
split, and a light backtest on the pre-decimation signal. The runners return
the per-cell IC series alongside the summary table so the grid can be
re-evaluated honestly with ``walk_forward_selection``.
"""

import numpy as np
import pandas as pd

from src.research.momentum_eval import (
    compute_ic_series,
    ic_summary,
    run_light_backtest,
)
from src.signals.cs_mean_reversion import build_reversal_family_panel
from src.signals.cs_momentum import (
    build_feature_panels,
    build_momentum_signal,
    build_monthly_universe_mask,
    build_selected_momentum_signal,
    compute_forward_returns,
    cross_sectional_rank_or_zscore,
    select_momentum_feature_panel,
)


def filter_universe_top_n(universe_df, top_n):
    """Monthly universe rows restricted to rank <= ``top_n`` (None passes through)."""
    if universe_df is None:
        return None
    return universe_df[universe_df["rank"] <= int(top_n)].copy()


def train_test_ic_stats(ic_series, split_ts, nw_lag):
    """Mean IC and NW t-stat before (inclusive) and after ``split_ts``."""
    train = ic_series[ic_series.index <= split_ts]
    test = ic_series[ic_series.index > split_ts]
    train_stats = ic_summary(train, nw_lag=nw_lag) if len(train) else None
    test_stats = ic_summary(test, nw_lag=nw_lag) if len(test) else None
    return {
        "train_mean_ic": float(train.mean()) if len(train) else np.nan,
        "test_mean_ic": float(test.mean()) if len(test) else np.nan,
        "train_t_stat_nw": train_stats["t_stat_ic_nw"] if train_stats else np.nan,
        "test_t_stat_nw": test_stats["t_stat_ic_nw"] if test_stats else np.nan,
    }


def evaluate_signal_config(
    signal_wide,
    ic_forward_return_wide,
    backtest_forward_return_wide,
    lookback_bars,
    holding_period_bars,
    ic_decimation_bars=1,
    min_assets=6,
    train_split=None,
    **backtest_kwargs,
):
    """Score one (pre-decimation) signal configuration.

    IC is computed on every ``ic_decimation_bars``-th row against
    ``ic_forward_return_wide``; the NW lag is the last overlapping lag of that
    subsampled series, ``(max(H, lookback) - 1) // D``. The backtest trades
    the full signal at ``holding_period_bars`` cadence on
    ``backtest_forward_return_wide``.

    Returns a dict with ``ic`` (series), ``nw_lag``, ``summary``
    (``ic_summary``), ``train_test`` (``train_test_ic_stats``, empty when
    ``train_split`` is None) and ``backtest``.
    """
    H = int(holding_period_bars)
    D = max(int(ic_decimation_bars), 1)
    overlap = max(H, int(lookback_bars)) - 1
    if D > 1:
        signal_ic = signal_wide.iloc[::D]
        nw_lag = overlap // D
    else:
        signal_ic = signal_wide
        nw_lag = overlap
    ic = compute_ic_series(
        signal_ic, ic_forward_return_wide.reindex(signal_ic.index), min_assets=min_assets
    )
    backtest = run_light_backtest(
        signal_wide,
        backtest_forward_return_wide,
        holding_period_bars=H,
        **backtest_kwargs,
    )
    return {
        "ic": ic,
        "nw_lag": nw_lag,
        "summary": ic_summary(ic, nw_lag=nw_lag),
        "train_test": (
            train_test_ic_stats(ic, train_split, nw_lag) if train_split is not None else {}
        ),
        "backtest": backtest,
    }


def _backtest_kwargs(settings):
    return dict(
        top_quantile=settings["top_quantile"],
        bottom_quantile=settings["bottom_quantile"],
        fee_bps=settings["fee_bps"],
        half_spread_bps=settings["half_spread_bps"],
        execution_delay_bars=settings["execution_delay_bars"],
        returns_are_log=settings["log_returns"],
    )


def run_reversal_grid(
    close_wide,
    universe_df,
    settings,
    family,
    backtest_forward_returns,
    ic_decimation_bars=1,
    transform_grid=None,
):
    """Reversal lookback x holding x transform x universe-size grid.

    ``backtest_forward_returns`` maps each holding period in
    ``settings["holding_period_grid_bars"]`` to the forward-return panel the
    backtest prices on (e.g. execution-priced); IC uses close-to-close returns
    (raw predictive power). Grid axes come from ``momentum_lookback_grid_bars``,
    ``holding_period_grid_bars``, ``transform_grid`` (default
    ``cross_sectional_transform_grid``) and ``universe_top_n_grid``.

    Returns ``(table, ic_by_config, nw_lag_by_config)``; the table is sorted by
    mean IC then net Sharpe.
    """
    log_returns = settings["log_returns"]
    min_assets = settings["min_assets_per_timestamp"]
    bt_kwargs = _backtest_kwargs(settings)
    if transform_grid is None:
        transform_grid = settings["cross_sectional_transform_grid"]
    ic_forward_returns = {
        h: compute_forward_returns(close_wide, holding_period_bars=h, log_returns=log_returns)
        for h in settings["holding_period_grid_bars"]
    }

    rows, ic_by_config, nw_by_config = [], {}, {}
    for lookback in settings["momentum_lookback_grid_bars"]:
        for top_n in settings["universe_top_n_grid"]:
            grid_universe = filter_universe_top_n(universe_df, top_n)
            universe_mask = (
                build_monthly_universe_mask(close_wide.index, close_wide.columns, grid_universe)
                if grid_universe is not None
                else None
            )
            raw = build_reversal_family_panel(
                close_wide,
                family,
                lookback,
                vol_window_bars=settings["vol_window_bars"],
                log_returns=log_returns,
                universe_mask=universe_mask,
            )
            for holding in settings["holding_period_grid_bars"]:
                fwd_ic = ic_forward_returns[holding]
                fwd_bt = backtest_forward_returns[holding]
                if universe_mask is not None:
                    aligned = universe_mask.reindex(
                        index=fwd_ic.index, columns=fwd_ic.columns
                    ).fillna(False)
                    fwd_ic = fwd_ic.where(aligned)
                    fwd_bt = fwd_bt.where(aligned)
                for transform in transform_grid:
                    signal = cross_sectional_rank_or_zscore(
                        raw, method=transform, min_assets_per_timestamp=min_assets
                    )
                    result = evaluate_signal_config(
                        signal,
                        fwd_ic,
                        fwd_bt,
                        lookback_bars=lookback,
                        holding_period_bars=holding,
                        ic_decimation_bars=ic_decimation_bars,
                        min_assets=min_assets,
                        train_split=settings["train_split_date"],
                        **bt_kwargs,
                    )
                    label = f"lb{lookback}_h{holding}_{transform}_top{top_n}"
                    ic_by_config[label] = result["ic"]
                    nw_by_config[label] = result["nw_lag"]
                    summary, metrics = result["summary"], result["backtest"]["metrics"]
                    rows.append(
                        {
                            "lookback": lookback,
                            "holding": holding,
                            "transform": transform,
                            "top_n": top_n,
                            "nw_lag": result["nw_lag"],
                            "mean_ic": summary["mean_ic"],
                            "t_stat_ic": summary["t_stat_ic"],
                            "t_stat_ic_nw": summary["t_stat_ic_nw"],
                            "sharpe_net": metrics["sharpe_net"],
                            "mean_turnover": metrics["mean_turnover"],
                            "annualized_turnover": metrics["annualized_turnover"],
                            "max_dd": metrics["max_drawdown_net"],
                            **result["train_test"],
                        }
                    )

    table = (
        pd.DataFrame(rows)
        .sort_values(["mean_ic", "sharpe_net"], ascending=False)
        .reset_index(drop=True)
    )
    return table, ic_by_config, nw_by_config


def run_momentum_grid(
    df_long,
    universe_df,
    settings,
    feature_family,
    relative_feature,
    vol_feature,
    residual_feature,
    apply_xsec_transform=True,
    apply_rebalance_decimation=True,
    beta_panel=None,
    market_returns=None,
):
    """Signal horizon x JT skip x holding x transform x universe-size grid.

    Rebuilds the momentum signal and feature pack per (skip, universe) cell —
    the skip changes the feature definition and the universe changes the
    mask — then selects ``feature_family`` at each horizon in
    ``settings["feature_horizons_bars"]``. Residual features reuse the
    supplied ``beta_panel``/``market_returns`` for every universe size (the
    universe gates index membership, not the betas).

    Returns ``(table, ic_by_config, nw_lag_by_config)``; the table is sorted by
    mean IC then net Sharpe and keeps the original row order as its index.
    """
    log_returns = settings["log_returns"]
    min_assets = settings["min_assets_per_timestamp"]
    D = int(settings["ic_rebalance_bars"])
    ic_decimation = D if apply_rebalance_decimation else 1
    bt_kwargs = _backtest_kwargs(settings)
    transform_grid = (
        settings["cross_sectional_transform_grid"]
        if apply_xsec_transform
        else [settings["cross_sectional_transform"]]
    )

    packs = {}

    def _packs(lookback, skip_bars, top_n):
        key = (lookback, skip_bars, top_n)
        if key not in packs:
            pack = build_momentum_signal(
                df_long=df_long,
                signal_timeframe=settings["signal_timeframe"],
                momentum_lookback_bars=lookback,
                momentum_skip_bars=skip_bars,
                universe_df=filter_universe_top_n(universe_df, top_n),
                cross_sectional_transform=settings["cross_sectional_transform"],
                min_assets_per_timestamp=min_assets,
                log_returns=log_returns,
                ic_rebalance_bars=settings["ic_rebalance_bars"],
            )
            feature_pack = build_feature_panels(
                close_wide=pack["close_wide"],
                horizons=settings["feature_horizons_bars"],
                benchmark_symbol=settings["benchmark_symbol"],
                vol_window_bars=settings["vol_window_bars"],
                log_returns=log_returns,
                residual_space=settings["residual_space"],
                universe_mask=pack["universe_mask"],
                skip_bars=skip_bars,
                beta_panel=beta_panel,
                market_returns=market_returns,
            )
            packs[key] = (pack, feature_pack)
        return packs[key]

    rows, ic_by_config, nw_by_config = [], {}, {}
    for signal_horizon in settings["feature_horizons_bars"]:
        for skip_bars in settings["momentum_skip_grid_bars"]:
            for hold in settings["holding_period_grid_bars"]:
                for transform in transform_grid:
                    for top_n in settings["universe_top_n_grid"]:
                        lookback = (
                            signal_horizon
                            if feature_family == "baseline"
                            else settings["momentum_lookback_bars"]
                        )
                        pack, feature_pack = _packs(lookback, skip_bars, top_n)
                        raw, signal_label = select_momentum_feature_panel(
                            feature_pack=feature_pack,
                            baseline_signal=pack["raw_signal"],
                            available_horizons=settings["feature_horizons_bars"],
                            feature_family=feature_family,
                            feature_horizon=signal_horizon,
                            relative_feature=relative_feature,
                            vol_feature=vol_feature,
                            residual_feature=residual_feature,
                        )
                        signal = build_selected_momentum_signal(
                            raw_panel=raw,
                            universe_mask=pack["universe_mask"],
                            cross_sectional_transform=transform,
                            min_assets_per_timestamp=min_assets,
                            apply_xsec_transform=apply_xsec_transform,
                            apply_rebalance_decimation_flag=False,
                        )
                        fwd = compute_forward_returns(
                            pack["close_wide"], holding_period_bars=hold, log_returns=log_returns
                        )
                        result = evaluate_signal_config(
                            signal,
                            fwd,
                            fwd,
                            lookback_bars=signal_horizon,
                            holding_period_bars=hold,
                            ic_decimation_bars=ic_decimation,
                            min_assets=min_assets,
                            train_split=settings["train_split_date"],
                            **bt_kwargs,
                        )
                        label = (
                            f"R{int(signal_horizon)}_s{int(skip_bars)}_h{int(hold)}"
                            f"_{transform}_top{int(top_n)}"
                        )
                        ic_by_config[label] = result["ic"]
                        nw_by_config[label] = result["nw_lag"]
                        summary, metrics = result["summary"], result["backtest"]["metrics"]
                        rows.append(
                            {
                                "signal_horizon": int(signal_horizon),
                                "signal_label": signal_label,
                                "relative_feature": relative_feature,
                                "vol_feature": vol_feature,
                                "skip_bars": int(skip_bars),
                                "apply_xsec_transform": bool(apply_xsec_transform),
                                "apply_rebalance_decimation": bool(apply_rebalance_decimation),
                                "holding": hold,
                                "top_n": top_n,
                                "nw_lag": result["nw_lag"],
                                "mean_ic": summary["mean_ic"],
                                "t_stat_ic": summary["t_stat_ic"],
                                "t_stat_ic_nw": summary["t_stat_ic_nw"],
                                "sharpe_net": metrics["sharpe_net"],
                                "mean_turnover": metrics["mean_turnover"],
                                "annualized_turnover": metrics["annualized_turnover"],
                                **result["train_test"],
                            }
                        )

    table = pd.DataFrame(rows).sort_values(["mean_ic", "sharpe_net"], ascending=False)
    return table, ic_by_config, nw_by_config

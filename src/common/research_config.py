"""Load research-specific settings while reusing project defaults."""

from pathlib import Path

import pandas as pd
import yaml

from src.common.config import load_settings


def load_research_settings(config_path, base_config_path=None):
    """Load research settings and merge with base settings.

    Handles both momentum and mean-reversion configs.  Every supported YAML key
    has an explicit entry in the ``settings.update()`` block below with a typed
    default, so the returned dict is always fully populated regardless of which
    keys are present in the config file.  To add a new knob, add it to that
    block and to the relevant YAML.
    """
    with open(Path(config_path)) as f:
        cfg = yaml.safe_load(f) or {}

    base_path = base_config_path or cfg.get("reference_base_config")
    base_settings = load_settings(base_path) if base_path else load_settings()

    settings = dict(base_settings)
    settings.update(
        {
            "signal_timeframe": str(cfg.get("signal_timeframe", "4h")),
            # Decimation cadence for IC analysis only — backtests always use
            # the pre-decimation signal with holding_period_bars as cadence.
            "ic_rebalance_bars": int(cfg.get("ic_rebalance_bars", 1)),
            "holding_period_bars": int(cfg.get("holding_period_bars", 1)),
            "execution_delay_bars": int(cfg.get("execution_delay_bars", 0)),
            # Minute-level execution modelling (entry/exit at signal close
            # + delta minutes, priced from the sparse ohlcv_1m table).
            "use_minute_execution": bool(cfg.get("use_minute_execution", False)),
            "execution_delay_minutes": int(cfg.get("execution_delay_minutes", 1)),
            "execution_delay_minutes_grid": list(
                cfg.get("execution_delay_minutes_grid", [0, 1, 5, 15, 30, 60])
            ),
            # Market-beta neutrality: market-index construction, rolling beta
            # estimation, and the ex-ante beta-hedge overlay.
            "market_index_mode": str(
                cfg.get("market_index_mode", "equal_weight")
            ).lower(),
            "beta_window_bars": int(cfg.get("beta_window_bars", 90)),
            "beta_min_periods_bars": cfg.get("beta_min_periods_bars"),
            "beta_shrinkage": float(cfg.get("beta_shrinkage", 0.2)),
            "beta_shrink_target": float(cfg.get("beta_shrink_target", 1.0)),
            "beta_hedge_enabled": bool(cfg.get("beta_hedge_enabled", False)),
            "max_hedge_weight": cfg.get("max_hedge_weight"),
            "momentum_lookback_bars": int(cfg.get("momentum_lookback_bars", 6)),
            "momentum_skip_bars": int(cfg.get("momentum_skip_bars", 0)),
            "log_returns": bool(cfg.get("log_returns", True)),
            "vol_window_bars": int(cfg.get("vol_window_bars", 24)),
            "benchmark_symbol": str(cfg.get("benchmark_symbol", "XBT/USD")),
            "residual_space": str(cfg.get("residual_space", "log")).lower(),
            "use_monthly_universe": bool(cfg.get("use_monthly_universe", True)),
            "max_assets": cfg.get("max_assets"),
            "cross_sectional_transform": str(
                cfg.get("cross_sectional_transform", "zscore")
            ),
            "feature_horizons_bars": list(cfg.get("feature_horizons_bars", [1, 3, 6, 12])),
            "momentum_lookback_grid_bars": list(
                cfg.get("momentum_lookback_grid_bars", [1, 3, 6, 12])
            ),
            "momentum_skip_grid_bars": list(cfg.get("momentum_skip_grid_bars", [0])),
            "holding_period_grid_bars": list(cfg.get("holding_period_grid_bars", [1, 3])),
            "cross_sectional_transform_grid": list(
                cfg.get("cross_sectional_transform_grid", ["rank", "zscore"])
            ),
            "universe_top_n_grid": list(cfg.get("universe_top_n_grid", [10, 20])),
            "min_assets_per_timestamp": int(cfg.get("min_assets_per_timestamp", 6)),
            "top_quantile": float(cfg.get("top_quantile", 0.2)),
            "bottom_quantile": float(cfg.get("bottom_quantile", 0.2)),
            "fee_bps": float(cfg.get("fee_bps", 2.0)),
            "half_spread_bps": float(cfg.get("half_spread_bps", 1.0)),
            "fee_tier_grid_bps": list(cfg.get("fee_tier_grid_bps", [])),
            "rolling_window_bars": int(cfg.get("rolling_window_bars", 60)),
            "extreme_event_quantile": float(cfg.get("extreme_event_quantile", 0.99)),
            "data_start_date": pd.Timestamp(
                cfg.get("data_start_date", base_settings["data_start_date"])
            ),
            "data_end_date": pd.Timestamp(
                cfg.get("data_end_date", base_settings["data_end_date"])
            ),
            "train_split_date": pd.Timestamp(
                cfg.get("train_split_date", cfg.get("data_end_date", base_settings["data_end_date"]))
            ),
            # Mean-reversion specific knobs (no-ops for momentum configs that omit them).
            "reversal_lookback_bars": int(cfg.get("reversal_lookback_bars", 2)),
            "price_zscore_window_bars": int(cfg.get("price_zscore_window_bars", 24)),
            "bollinger_window_bars": int(cfg.get("bollinger_window_bars", 20)),
            "vwap_window_bars": int(cfg.get("vwap_window_bars", 24)),
            "volume_window_bars": int(cfg.get("volume_window_bars", 24)),
            "rsi_window_bars": int(cfg.get("rsi_window_bars", 14)),
            "range_position_window_bars": int(cfg.get("range_position_window_bars", 12)),
            "extreme_move_threshold_sigma": float(cfg.get("extreme_move_threshold_sigma", 2.0)),
        }
    )

    if settings["residual_space"] not in ("log", "simple"):
        raise ValueError(
            "residual_space must be 'log' or 'simple', "
            f"got {settings['residual_space']!r}"
        )

    if settings["use_minute_execution"] and settings["execution_delay_bars"] > 0:
        raise ValueError(
            "use_minute_execution embeds the delay in the execution close "
            "panel; execution_delay_bars must stay 0 to avoid a double delay "
            f"(got execution_delay_bars={settings['execution_delay_bars']})"
        )

    if settings["market_index_mode"] not in ("equal_weight", "benchmark"):
        raise ValueError(
            "market_index_mode must be 'equal_weight' or 'benchmark', "
            f"got {settings['market_index_mode']!r}"
        )
    if not 0.0 <= settings["beta_shrinkage"] <= 1.0:
        raise ValueError(
            f"beta_shrinkage must be in [0, 1], got {settings['beta_shrinkage']}"
        )
    if settings["beta_min_periods_bars"] is not None:
        settings["beta_min_periods_bars"] = int(settings["beta_min_periods_bars"])
    if settings["max_hedge_weight"] is not None:
        settings["max_hedge_weight"] = float(settings["max_hedge_weight"])

    if settings["max_assets"] is not None:
        settings["max_assets"] = int(settings["max_assets"])
    settings["feature_horizons_bars"] = [int(x) for x in settings["feature_horizons_bars"]]
    settings["momentum_lookback_grid_bars"] = [
        int(x) for x in settings["momentum_lookback_grid_bars"]
    ]
    settings["momentum_skip_grid_bars"] = [int(x) for x in settings["momentum_skip_grid_bars"]]
    settings["holding_period_grid_bars"] = [int(x) for x in settings["holding_period_grid_bars"]]
    settings["universe_top_n_grid"] = [int(x) for x in settings["universe_top_n_grid"]]
    # One-way taker fee levels for the cost sensitivity; always includes the
    # headline fee_bps so the base case is one of the reported tiers.
    settings["fee_tier_grid_bps"] = sorted(
        {float(x) for x in settings["fee_tier_grid_bps"]} | {settings["fee_bps"]}
    )
    settings["execution_delay_minutes_grid"] = [
        int(x) for x in settings["execution_delay_minutes_grid"]
    ]

    return settings

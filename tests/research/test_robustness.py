"""Unit tests for src/research/robustness.py (grid-cell scoring and runners)."""

import numpy as np
import pandas as pd
import pytest

from src.research.momentum_eval import compute_ic_series, ic_summary, run_light_backtest
from src.research.robustness import (
    evaluate_signal_config,
    filter_universe_top_n,
    run_reversal_grid,
    train_test_ic_stats,
)
from src.signals.cs_momentum import compute_forward_returns


def _panel(n=60, m=8, seed=0, scale=1.0):
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, scale, size=(n, m)), index=idx, columns=[f"S{i}" for i in range(m)]
    )


class TestFilterUniverseTopN:
    def test_keeps_ranks_at_or_below_n(self):
        u = pd.DataFrame({"symbol": list("ABCD"), "rank": [1, 2, 3, 4]})
        assert filter_universe_top_n(u, 2)["symbol"].tolist() == ["A", "B"]

    def test_none_passes_through(self):
        assert filter_universe_top_n(None, 5) is None


class TestTrainTestICStats:
    def test_split_is_inclusive_on_train_side(self):
        ic = pd.Series([0.1, 0.2, 0.3, 0.4], index=pd.date_range("2023-01-01", periods=4, freq="D"))
        out = train_test_ic_stats(ic, pd.Timestamp("2023-01-02"), nw_lag=0)
        assert np.isclose(out["train_mean_ic"], 0.15)
        assert np.isclose(out["test_mean_ic"], 0.35)

    def test_empty_side_is_nan(self):
        ic = pd.Series([0.1, 0.2], index=pd.date_range("2023-01-01", periods=2, freq="D"))
        out = train_test_ic_stats(ic, pd.Timestamp("2024-01-01"), nw_lag=0)
        assert np.isnan(out["test_mean_ic"]) and np.isnan(out["test_t_stat_nw"])


class TestEvaluateSignalConfig:
    def test_matches_direct_computation_without_decimation(self):
        signal, fwd = _panel(seed=1), _panel(seed=2, scale=0.01)
        out = evaluate_signal_config(
            signal, fwd, fwd, lookback_bars=3, holding_period_bars=2,
            ic_decimation_bars=1, min_assets=4, fee_bps=5.0, half_spread_bps=1.0,
        )
        assert out["nw_lag"] == 2  # max(H=2, lookback=3) - 1
        pd.testing.assert_series_equal(out["ic"], compute_ic_series(signal, fwd, min_assets=4))
        direct_bt = run_light_backtest(
            signal, fwd, holding_period_bars=2, fee_bps=5.0, half_spread_bps=1.0
        )
        pd.testing.assert_series_equal(out["backtest"]["net_returns"], direct_bt["net_returns"])
        assert out["train_test"] == {}

    def test_decimation_subsamples_ic_and_scales_nw_lag(self):
        signal, fwd = _panel(seed=3), _panel(seed=4, scale=0.01)
        out = evaluate_signal_config(
            signal, fwd, fwd, lookback_bars=5, holding_period_bars=7,
            ic_decimation_bars=3, min_assets=4,
        )
        assert out["nw_lag"] == (7 - 1) // 3
        assert out["ic"].index.equals(signal.index[::3])
        assert out["summary"] == ic_summary(out["ic"], nw_lag=out["nw_lag"])

    def test_backtest_uses_full_signal_not_decimated(self):
        signal, fwd = _panel(seed=5), _panel(seed=6, scale=0.01)
        a = evaluate_signal_config(signal, fwd, fwd, 1, 1, ic_decimation_bars=4, min_assets=4)
        b = evaluate_signal_config(signal, fwd, fwd, 1, 1, ic_decimation_bars=1, min_assets=4)
        pd.testing.assert_series_equal(a["backtest"]["net_returns"], b["backtest"]["net_returns"])


class TestRunReversalGrid:
    def _settings(self):
        return {
            "log_returns": True,
            "min_assets_per_timestamp": 4,
            "top_quantile": 0.25,
            "bottom_quantile": 0.25,
            "fee_bps": 0.0,
            "half_spread_bps": 0.0,
            "execution_delay_bars": 0,
            "vol_window_bars": 5,
            "momentum_lookback_grid_bars": [1, 2],
            "holding_period_grid_bars": [1, 3],
            "cross_sectional_transform_grid": ["rank", "zscore"],
            "universe_top_n_grid": [8],
            "train_split_date": pd.Timestamp("2023-01-02", tz="UTC"),
        }

    def test_grid_shape_labels_and_ic_alignment(self):
        settings = self._settings()
        close = 100.0 * np.exp(_panel(n=80, seed=7, scale=0.01).cumsum())
        fwd_bt = {
            h: compute_forward_returns(close, holding_period_bars=h, log_returns=True)
            for h in settings["holding_period_grid_bars"]
        }
        table, ic_by, nw_by = run_reversal_grid(
            close, None, settings, family="reversal", backtest_forward_returns=fwd_bt
        )
        assert len(table) == 2 * 2 * 2 * 1
        assert "lb2_h3_zscore_top8" in ic_by
        assert nw_by["lb2_h3_zscore_top8"] == 2
        assert table["mean_ic"].is_monotonic_decreasing
        assert set(ic_by) == set(nw_by)

"""Unit tests for src/research/spreads.py.

The Corwin-Schultz estimator has a clean closed-form property used here: with
a constant mid-price and every bar printing both the ask (high) and the bid
(low), alpha reduces exactly to log(high/low) and the estimate recovers the
true proportional spread exactly. The Roll estimator is checked on simulated
Roll-model data (random bid/ask bounce around a martingale mid).
"""

import numpy as np
import pandas as pd
import pytest

from src.research.momentum_eval import run_light_backtest
from src.research.spreads import (
    corwin_schultz_spread,
    estimate_half_spread_bps_from_long,
    half_spread_bps,
    roll_spread,
)


def _utc_index(n, freq="1h", start="2023-01-01"):
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC")


def _pure_bounce_hl(spread, n=50, mid=100.0):
    """High/low panels for a constant mid with proportional spread `spread`."""
    ask = mid * (1.0 + spread / 2.0)
    bid = mid * (1.0 - spread / 2.0)
    idx = _utc_index(n)
    high = pd.DataFrame({"AAA": ask}, index=idx)
    low = pd.DataFrame({"AAA": bid}, index=idx)
    return high, low


class TestCorwinSchultz:
    def test_recovers_spread_exactly_under_pure_bounce(self):
        # With constant mid, alpha == log(high/low) and S == 2(e^a-1)/(e^a+1),
        # which equals (ask-bid)/mid exactly.
        spread = 0.01
        high, low = _pure_bounce_hl(spread)
        est = corwin_schultz_spread(high, low)
        valid = est["AAA"].dropna()
        assert len(valid) == len(high) - 1
        np.testing.assert_allclose(valid.values, spread, rtol=1e-10)

    def test_zero_range_gives_zero_spread(self):
        idx = _utc_index(10)
        flat = pd.DataFrame({"AAA": 100.0}, index=idx)
        est = corwin_schultz_spread(flat, flat)
        assert (est["AAA"].dropna() == 0.0).all()

    def test_negative_estimates_clipped_by_default(self):
        # Large two-bar range vs single-bar ranges (vol >> spread) drives the
        # raw estimator negative.
        idx = _utc_index(3)
        high = pd.DataFrame({"AAA": [100.1, 120.1, 100.1]}, index=idx)
        low = pd.DataFrame({"AAA": [99.9, 119.9, 99.9]}, index=idx)
        clipped = corwin_schultz_spread(high, low)
        raw = corwin_schultz_spread(high, low, clip_negative=False)
        assert (raw["AAA"].dropna() < 0).any()
        assert (clipped["AAA"].dropna() >= 0).all()

    def test_invalid_bars_are_nan(self):
        idx = _utc_index(4)
        high = pd.DataFrame({"AAA": [101.0, 99.0, 101.0, np.nan]}, index=idx)
        low = pd.DataFrame({"AAA": [100.0, 100.0, 100.0, 100.0]}, index=idx)
        est = corwin_schultz_spread(high, low)
        # Bar 1 has high < low -> pairs (0,1), (1,2) invalid; bar 3 NaN -> pair (2,3) invalid.
        assert est["AAA"].isna().all()


class TestRollSpread:
    def test_recovers_spread_on_roll_model(self):
        rng = np.random.default_rng(7)
        n = 20000
        spread = 0.01
        mid = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
        side = rng.choice([-1.0, 1.0], size=n)
        close = mid * (1.0 + side * spread / 2.0)
        panel = pd.DataFrame({"AAA": close}, index=_utc_index(n))
        est = roll_spread(panel)
        assert est["AAA"] == pytest.approx(spread, rel=0.10)

    def test_nan_when_autocovariance_positive(self):
        # Smoothly accelerating returns -> positive autocovariance -> model
        # uninformative.
        rets = np.linspace(0.001, 0.02, 100)
        panel = pd.DataFrame({"AAA": np.exp(np.cumsum(rets))}, index=_utc_index(100))
        assert np.isnan(roll_spread(panel)["AAA"])


class TestHalfSpreadBps:
    def test_median_and_units(self):
        idx = _utc_index(40)
        panel = pd.DataFrame({"AAA": 0.002, "BBB": 0.02}, index=idx)
        out = half_spread_bps(panel, min_obs=30)
        assert out["AAA"] == pytest.approx(10.0)  # 20 bps spread -> 10 bps half
        assert out["BBB"] == pytest.approx(100.0)

    def test_min_obs_gate(self):
        idx = _utc_index(40)
        panel = pd.DataFrame({"AAA": 0.002}, index=idx)
        panel.iloc[20:, 0] = np.nan
        assert np.isnan(half_spread_bps(panel, min_obs=30)["AAA"])
        assert half_spread_bps(panel, min_obs=10)["AAA"] == pytest.approx(10.0)


class TestEstimateFromLong:
    def test_stale_hours_excluded_and_pipeline_runs(self):
        # 40 days of hourly bars; stale (trades=0) hours carry high==low which
        # would drag the estimate toward zero if not excluded.
        spread = 0.01
        hours = _utc_index(40 * 24)
        rows = []
        for i, ts in enumerate(hours):
            stale = i % 3 == 0
            rows.append(
                {
                    "ts": ts,
                    "symbol": "AAA",
                    "high": 100.0 if stale else 100.0 * (1 + spread / 2),
                    "low": 100.0 if stale else 100.0 * (1 - spread / 2),
                    "trades": 0 if stale else 5,
                }
            )
        est, panel = estimate_half_spread_bps_from_long(
            pd.DataFrame(rows), estimation_timeframe="24h", min_obs=30
        )
        # Pure bounce at daily aggregation -> exact recovery: half of 100 bps.
        assert est["AAA"] == pytest.approx(50.0, rel=1e-6)

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError, match="missing required columns"):
            estimate_half_spread_bps_from_long(pd.DataFrame({"ts": [], "symbol": []}))


class TestPerAssetCosts:
    def _setup(self):
        # AAA/CCC swap book sides every bar so per-asset turnover is nonzero
        # (a constant book never rebalances and all cost paths degenerate to 0).
        idx = _utc_index(6)
        signal = pd.DataFrame(
            {"AAA": [1, -1, 1, -1, 1, -1], "BBB": [0, 0, 0, 0, 0, 0],
             "CCC": [-1, 1, -1, 1, -1, 1]},
            index=idx, dtype=float,
        )
        fwd = pd.DataFrame(0.01, index=idx, columns=signal.columns)
        return signal, fwd

    def test_uniform_series_matches_scalar(self):
        signal, fwd = self._setup()
        common = dict(
            signal_wide=signal, forward_return_wide=fwd,
            top_quantile=0.34, bottom_quantile=0.34,
            fee_bps=10.0, holding_period_bars=1,
        )
        scalar_bt = run_light_backtest(half_spread_bps=5.0, **common)
        series_bt = run_light_backtest(
            half_spread_bps=pd.Series(5.0, index=signal.columns), **common
        )
        pd.testing.assert_series_equal(scalar_bt["net_returns"], series_bt["net_returns"])

    def test_wider_spread_on_traded_asset_raises_costs(self):
        signal, fwd = self._setup()
        common = dict(
            signal_wide=signal, forward_return_wide=fwd,
            top_quantile=0.34, bottom_quantile=0.34,
            fee_bps=10.0, holding_period_bars=1,
        )
        flat = run_light_backtest(half_spread_bps=pd.Series(5.0, index=signal.columns), **common)
        wide = run_light_backtest(
            half_spread_bps=pd.Series({"AAA": 50.0, "BBB": 5.0, "CCC": 5.0}), **common
        )
        # AAA is in the long book, so widening its spread must increase costs.
        assert wide["costs"].sum() > flat["costs"].sum()

    def test_missing_symbols_filled_with_median(self):
        signal, fwd = self._setup()
        common = dict(
            signal_wide=signal, forward_return_wide=fwd,
            top_quantile=0.34, bottom_quantile=0.34,
            fee_bps=10.0, holding_period_bars=1,
        )
        # CCC missing -> filled with median(5, 5) = 5 -> identical to uniform 5.
        partial = run_light_backtest(
            half_spread_bps=pd.Series({"AAA": 5.0, "BBB": 5.0}), **common
        )
        uniform = run_light_backtest(
            half_spread_bps=pd.Series(5.0, index=signal.columns), **common
        )
        pd.testing.assert_series_equal(partial["net_returns"], uniform["net_returns"])

"""Unit tests for the traded-bar (stale-price) mask in cs_momentum.

The hourly loader forward-fills no-trade hours with trades=0/volume=0; these
tests verify that build_traded_mask flags those bars and apply_traded_mask
removes cells whose defining price endpoints are stale.
"""

import numpy as np
import pandas as pd
import pytest

from src.signals.cs_momentum import apply_traded_mask, build_traded_mask


def _utc_index(n, freq="1h", start="2023-01-01"):
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC")


def _long_with_trades(trades_by_symbol, freq="1h"):
    """Build a long OHLCV-shaped frame from {symbol: [trades per bar]}."""
    n = len(next(iter(trades_by_symbol.values())))
    idx = _utc_index(n, freq=freq)
    rows = []
    for sym, trades in trades_by_symbol.items():
        for ts, tr in zip(idx, trades):
            rows.append({"ts": ts, "symbol": sym, "close": 100.0, "trades": tr})
    return pd.DataFrame(rows)


class TestBuildTradedMask:
    def test_flags_zero_trade_bars(self):
        df_long = _long_with_trades({"AAA": [3, 0, 1, 0], "BBB": [1, 1, 1, 1]})
        mask = build_traded_mask(df_long, "1h")
        assert mask["AAA"].tolist() == [True, False, True, False]
        assert mask["BBB"].tolist() == [True, True, True, True]

    def test_resample_aggregates_trades(self):
        # At 2h cadence a bar is traded if either hourly row traded.
        df_long = _long_with_trades({"AAA": [0, 2, 0, 0], "BBB": [1, 0, 0, 3]})
        mask = build_traded_mask(df_long, "2h")
        assert mask["AAA"].tolist() == [True, False]
        assert mask["BBB"].tolist() == [True, True]

    def test_missing_bins_are_false(self):
        # BBB only exists for the first two hours; its later bins have no rows.
        idx = _utc_index(4)
        rows = [{"ts": ts, "symbol": "AAA", "close": 1.0, "trades": 1} for ts in idx]
        rows += [
            {"ts": idx[0], "symbol": "BBB", "close": 1.0, "trades": 1},
            {"ts": idx[1], "symbol": "BBB", "close": 1.0, "trades": 1},
        ]
        mask = build_traded_mask(pd.DataFrame(rows), "1h")
        assert mask["BBB"].tolist() == [True, True, False, False]

    def test_raises_without_trades_column(self):
        df_long = _long_with_trades({"AAA": [1, 1]}).drop(columns=["trades"])
        with pytest.raises(ValueError, match="trades"):
            build_traded_mask(df_long, "1h")


class TestApplyTradedMask:
    @pytest.fixture
    def panel_and_mask(self):
        idx = _utc_index(5)
        panel = pd.DataFrame(
            {"AAA": np.arange(5.0), "BBB": np.arange(5.0) + 10.0}, index=idx
        )
        traded = pd.DataFrame(
            {
                "AAA": [True, True, False, True, True],
                "BBB": [True, True, True, True, True],
            },
            index=idx,
        )
        return panel, traded

    def test_masks_current_bar_only_by_default(self, panel_and_mask):
        panel, traded = panel_and_mask
        out = apply_traded_mask(panel, traded)
        assert np.isnan(out.loc[out.index[2], "AAA"])
        assert out["BBB"].notna().all()
        assert out["AAA"].notna().sum() == 4

    def test_lookback_endpoint_masked(self, panel_and_mask):
        panel, traded = panel_and_mask
        # 2-bar feature at t needs a real print at t AND t-2. AAA stale at bar 2
        # -> bars 2 (current) and 4 (lookback endpoint) both invalid.
        out = apply_traded_mask(panel, traded, lookback_bars=2)
        assert np.isnan(out.loc[out.index[2], "AAA"])
        assert np.isnan(out.loc[out.index[4], "AAA"])
        # Leading bars have no t-2 observation -> masked too (shift fills False).
        assert np.isnan(out.loc[out.index[0], "AAA"])
        assert out["AAA"].notna().tolist() == [False, False, False, True, False]

    def test_forward_endpoint_masked(self, panel_and_mask):
        panel, traded = panel_and_mask
        # H=1 forward return at t needs a real print at t AND t+1.
        out = apply_traded_mask(panel, traded, forward_bars=1)
        assert np.isnan(out.loc[out.index[1], "AAA"])  # exit at stale bar 2
        assert np.isnan(out.loc[out.index[2], "AAA"])  # entry at stale bar 2
        # Last bar has no t+1 observation -> masked.
        assert np.isnan(out.loc[out.index[4], "AAA"])
        assert out["AAA"].notna().tolist() == [True, False, False, True, False]

    def test_alignment_fills_missing_symbols_false(self, panel_and_mask):
        panel, traded = panel_and_mask
        out = apply_traded_mask(panel, traded[["AAA"]])
        assert out["BBB"].isna().all()

    def test_negative_offsets_raise(self, panel_and_mask):
        panel, traded = panel_and_mask
        with pytest.raises(ValueError, match=">= 0"):
            apply_traded_mask(panel, traded, lookback_bars=-1)

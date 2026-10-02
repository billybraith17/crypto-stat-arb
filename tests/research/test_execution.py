"""Unit tests for the minute-level execution panel (src/research/execution.py).

The load-bearing properties: alignment to the signal index, LOCF across empty
buckets without lookahead, fallback accounting, and the delta=0 equivalence
with plain close-to-close forward returns.
"""

import numpy as np
import pandas as pd
import pytest

from src.research.execution import (
    _zero_delay_mismatch,
    build_execution_close_panel,
    compute_execution_forward_returns,
)
from src.signals.cs_momentum import compute_forward_returns


def _hourly_index(n, start="2023-01-01"):
    return pd.date_range(start=start, periods=n, freq="1h", tz="UTC")


class TestBuildExecutionClosePanel:
    def test_aligned_to_signal_index(self, minute_exec_closes_long):
        idx = _hourly_index(6)
        panel, _ = build_execution_close_panel(minute_exec_closes_long, idx)
        assert panel.index.equals(idx)
        assert set(panel.columns) == {"AAA", "BBB"}

    def test_direct_values_pass_through(self, minute_exec_closes_long):
        idx = _hourly_index(6)
        panel, _ = build_execution_close_panel(minute_exec_closes_long, idx)
        assert np.allclose(panel["AAA"].values, [100.0, 101.0, 102.0, 103.0, 104.0, 105.0])

    def test_locf_fills_empty_bucket(self, minute_exec_closes_long):
        """BBB has no trades in the 02:00 bucket → carried from 01:00."""
        idx = _hourly_index(6)
        panel, diags = build_execution_close_panel(minute_exec_closes_long, idx)
        assert panel.loc[idx[2], "BBB"] == 51.0  # last close <= exec time, no lookahead
        assert diags["n_locf_fallback"] == 1
        assert diags["fallback_by_symbol"]["BBB"] == 1

    def test_leading_nan_preserved_without_fallback(self, minute_exec_closes_long):
        idx = _hourly_index(6)
        panel, diags = build_execution_close_panel(minute_exec_closes_long, idx)
        assert np.isnan(panel.loc[idx[0], "BBB"])  # BBB not listed yet
        assert diags["n_unpriceable"] == 0

    def test_signal_close_fallback_fills_leading_nan(self, minute_exec_closes_long):
        idx = _hourly_index(6)
        signal_close = pd.DataFrame(
            {"AAA": 1.0, "BBB": 42.0}, index=idx
        )
        panel, diags = build_execution_close_panel(
            minute_exec_closes_long, idx, signal_close_wide=signal_close
        )
        assert panel.loc[idx[0], "BBB"] == 42.0
        assert diags["n_signal_close_fallback"] == 1
        # direct cells must NOT be overwritten by the fallback
        assert panel.loc[idx[0], "AAA"] == 100.0

    def test_no_lookahead(self, minute_exec_closes_long):
        """Deleting all buckets after T must not change values at labels <= T."""
        idx = _hourly_index(6)
        full_panel, _ = build_execution_close_panel(minute_exec_closes_long, idx)

        cutoff = idx[3]
        truncated = minute_exec_closes_long[
            minute_exec_closes_long["bucket_ts"] <= cutoff
        ]
        trunc_panel, _ = build_execution_close_panel(truncated, idx[idx <= cutoff])

        pd.testing.assert_frame_equal(
            full_panel.loc[:cutoff], trunc_panel, check_freq=False
        )

    def test_empty_input_raises(self):
        idx = _hourly_index(3)
        empty = pd.DataFrame(columns=["bucket_ts", "symbol", "exec_close"])
        with pytest.raises(ValueError, match="empty"):
            build_execution_close_panel(empty, idx)

    def test_diagnostics_counts_sum(self, minute_exec_closes_long):
        idx = _hourly_index(6)
        _, diags = build_execution_close_panel(minute_exec_closes_long, idx)
        # 12 cells: 10 direct (6 AAA + 4 BBB), 1 LOCF (BBB 02:00),
        # 0 signal fallback, 1 leading NaN (BBB 00:00, expected, not counted).
        assert diags["n_cells"] == 12
        assert diags["n_direct"] == 10
        assert diags["n_locf_fallback"] == 1
        assert diags["n_signal_close_fallback"] == 0

    def test_direct_fill_mask_flags_fallback_cells(self, minute_exec_closes_long):
        idx = _hourly_index(6)
        _, diags = build_execution_close_panel(minute_exec_closes_long, idx)
        mask = diags["direct_fill_mask"]
        assert mask["AAA"].all()
        assert not mask.loc[idx[0], "BBB"]  # not listed yet
        assert not mask.loc[idx[2], "BBB"]  # LOCF-filled bucket
        assert mask["BBB"].sum() == 4
        # Count consistency with the scalar diagnostics.
        assert int(mask.sum().sum()) == diags["n_direct"]


class TestComputeExecutionForwardReturns:
    def test_hand_computed_log_returns(self, minute_exec_closes_long):
        idx = _hourly_index(6)
        panel, _ = build_execution_close_panel(minute_exec_closes_long, idx)
        fwd = compute_execution_forward_returns(panel, holding_period_bars=2)
        expected = np.log(103.0 / 101.0)
        assert abs(fwd.loc[idx[1], "AAA"] - expected) < 1e-12

    def test_simple_returns(self, minute_exec_closes_long):
        idx = _hourly_index(6)
        panel, _ = build_execution_close_panel(minute_exec_closes_long, idx)
        fwd = compute_execution_forward_returns(panel, holding_period_bars=1, log_returns=False)
        assert abs(fwd.loc[idx[0], "AAA"] - 0.01) < 1e-12

    def test_nan_when_exit_unpriceable(self, minute_exec_closes_long):
        idx = _hourly_index(6)
        panel, _ = build_execution_close_panel(minute_exec_closes_long, idx)
        fwd = compute_execution_forward_returns(panel, holding_period_bars=2)
        assert fwd.loc[idx[4:], "AAA"].isna().all()  # exit beyond panel end

    def test_nan_when_entry_unpriceable(self, minute_exec_closes_long):
        idx = _hourly_index(6)
        panel, _ = build_execution_close_panel(minute_exec_closes_long, idx)
        fwd = compute_execution_forward_returns(panel, holding_period_bars=1)
        assert np.isnan(fwd.loc[idx[0], "BBB"])  # BBB not listed at entry

    def test_direct_fill_mask_excludes_stale_fill_endpoints(self, minute_exec_closes_long):
        """With the mask, a return is NaN when its entry OR exit fill was
        LOCF-carried; without it, the same cells are priced."""
        idx = _hourly_index(6)
        panel, diags = build_execution_close_panel(minute_exec_closes_long, idx)
        unmasked = compute_execution_forward_returns(panel, holding_period_bars=1)
        masked = compute_execution_forward_returns(
            panel, holding_period_bars=1, direct_fill_mask=diags["direct_fill_mask"]
        )
        # BBB 02:00 is an LOCF fill -> both the return exiting into it (01:00)
        # and the return entering at it (02:00) must be excluded.
        assert not np.isnan(unmasked.loc[idx[1], "BBB"])
        assert not np.isnan(unmasked.loc[idx[2], "BBB"])
        assert np.isnan(masked.loc[idx[1], "BBB"])
        assert np.isnan(masked.loc[idx[2], "BBB"])
        # Direct-fill cells are untouched.
        pd.testing.assert_series_equal(masked["AAA"], unmasked["AAA"])

    def test_delta_zero_equals_close_to_close(self, simple_close_wide):
        """A delta=0 exec panel on fully traded data reproduces
        compute_forward_returns on the close panel exactly."""
        long = (
            simple_close_wide.stack()
            .rename("exec_close")
            .reset_index()
            .rename(columns={"level_0": "bucket_ts", "level_1": "symbol"})
        )
        long.columns = ["bucket_ts", "symbol", "exec_close"]
        panel, diags = build_execution_close_panel(long, simple_close_wide.index)
        assert diags["n_locf_fallback"] == 0
        fwd_exec = compute_execution_forward_returns(panel, holding_period_bars=1)
        fwd_close = compute_forward_returns(simple_close_wide, holding_period_bars=1)
        pd.testing.assert_frame_equal(
            fwd_exec, fwd_close, check_freq=False, check_names=False
        )


class TestZeroDelayMismatch:
    def _frames(self):
        idx = _hourly_index(3)
        cols = ["AAA", "BBB"]
        close = pd.DataFrame(
            {"AAA": [100.0, 101.0, 102.0], "BBB": [50.0, 50.5, 51.0]}, index=idx
        )
        volume = pd.DataFrame(
            {"AAA": [1.0, 1.0, 1.0], "BBB": [1.0, 1.0, 1.0]}, index=idx, columns=cols
        )
        return idx, cols, close, volume

    def test_clean_match(self):
        idx, cols, close, volume = self._frames()
        direct0 = close.copy()  # δ=0 exec close == hourly close everywhere
        s = _zero_delay_mismatch(direct0, close, volume)
        assert s["n_cells"] == 6
        assert s["n_synth_with_1m"] == 0
        assert s["max_rel_diff"] == 0.0

    def test_synthetic_bar_excluded_not_flagged(self):
        idx, cols, close, volume = self._frames()
        # BBB bar 1 is a forward-filled hourly bar (volume 0) but 1m has a
        # (different) real print — must be counted as synthetic, not drift.
        volume.loc[idx[1], "BBB"] = 0.0
        direct0 = close.copy()
        direct0.loc[idx[1], "BBB"] = 999.0
        s = _zero_delay_mismatch(direct0, close, volume)
        assert s["n_synth_with_1m"] == 1
        assert s["n_cells"] == 5
        assert s["max_rel_diff"] == 0.0  # the 999 bar is excluded from comparison

    def test_real_drift_is_caught(self):
        idx, cols, close, volume = self._frames()
        direct0 = close.copy()
        direct0.loc[idx[0], "AAA"] = 110.0  # 10% drift on a real-trade bar
        s = _zero_delay_mismatch(direct0, close, volume)
        assert abs(s["max_rel_diff"] - 0.1) < 1e-12

    def test_missing_1m_cell_ignored(self):
        idx, cols, close, volume = self._frames()
        direct0 = close.copy()
        direct0.loc[idx[2], "AAA"] = np.nan  # no 1m print this bar
        s = _zero_delay_mismatch(direct0, close, volume)
        assert s["n_cells"] == 5
        assert s["max_rel_diff"] == 0.0


class TestZeroDelayMismatchEdge:
    def test_no_comparable_cells_is_not_drift(self):
        """When no real-trade cell has a 1m print, n_cells=0 and max_rel_diff
        must be 0.0 (not NaN), so the downstream assert does not spuriously
        fire on an empty comparison."""
        idx = _hourly_index(2)
        close = pd.DataFrame({"AAA": [100.0, 101.0]}, index=idx)
        volume = pd.DataFrame({"AAA": [0.0, 0.0]}, index=idx)  # all synthetic bars
        direct0 = pd.DataFrame({"AAA": [100.0, 101.0]}, index=idx)
        s = _zero_delay_mismatch(direct0, close, volume)
        assert s["n_cells"] == 0
        assert s["max_rel_diff"] == 0.0
        assert s["n_synth_with_1m"] == 2


class TestExecutionPriceCache:
    def _cache(self, monkeypatch, minute_exec_closes_long, calls):
        import src.research.execution as execution

        def fake_fetch(engine, start_ts, end_ts, delay_minutes, symbols):
            calls.append(delay_minutes)
            return minute_exec_closes_long

        monkeypatch.setattr(execution, "fetch_minute_exec_closes", fake_fetch)
        idx = _hourly_index(6)
        close_wide = pd.DataFrame(
            {"AAA": np.arange(100.0, 106.0), "BBB": np.arange(50.0, 56.0)}, index=idx
        )
        return execution.ExecutionPriceCache(engine=None, close_wide=close_wide)

    def test_fetches_once_per_delay(self, monkeypatch, minute_exec_closes_long):
        calls = []
        cache = self._cache(monkeypatch, minute_exec_closes_long, calls)
        cache.close_panel(1)
        cache.diagnostics(1)
        cache.forward_returns(1, holding_period_bars=2)
        cache.close_panel(5)
        assert calls == [1, 5]

    def test_forward_returns_match_direct_computation(
        self, monkeypatch, minute_exec_closes_long
    ):
        cache = self._cache(monkeypatch, minute_exec_closes_long, [])
        expected = compute_execution_forward_returns(
            cache.close_panel(1),
            holding_period_bars=2,
            log_returns=True,
            direct_fill_mask=cache.diagnostics(1)["direct_fill_mask"],
        )
        pd.testing.assert_frame_equal(
            cache.forward_returns(1, holding_period_bars=2, log_returns=True), expected
        )


class TestRunDelaySweep:
    def test_each_delay_backtests_its_own_prices(self):
        from src.research.execution import run_delay_sweep
        from src.research.momentum_eval import run_light_backtest

        idx = _hourly_index(8)
        cols = [f"S{i}" for i in range(6)]
        rng = np.random.default_rng(3)
        signal = pd.DataFrame(rng.normal(size=(8, 6)), index=idx, columns=cols)
        fwd_by_delay = {
            d: pd.DataFrame(rng.normal(0, 0.01, size=(8, 6)), index=idx, columns=cols)
            for d in (0, 5)
        }

        class _Prices:
            def forward_returns(self, delay, holding_period_bars, log_returns=True):
                return fwd_by_delay[delay]

        out = run_delay_sweep(
            signal, _Prices(), [0, 5], holding_period_bars=2, log_returns=False,
            fee_bps=10.0, half_spread_bps=1.0,
        )
        assert list(out) == [0, 5]
        for d in (0, 5):
            direct = run_light_backtest(
                signal, fwd_by_delay[d], fee_bps=10.0, half_spread_bps=1.0,
                returns_are_log=False, holding_period_bars=2,
            )
            pd.testing.assert_series_equal(out[d]["net_returns"], direct["net_returns"])

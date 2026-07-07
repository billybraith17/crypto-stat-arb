"""Unit tests for the minute-level execution panel (src/research/execution.py).

The load-bearing properties: alignment to the signal index, LOCF across empty
buckets without lookahead, fallback accounting, and the delta=0 equivalence
with plain close-to-close forward returns.
"""

import numpy as np
import pandas as pd
import pytest

from src.research.execution import (
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

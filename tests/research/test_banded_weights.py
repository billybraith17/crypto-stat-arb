"""Unit tests for build_banded_quantile_weights (hysteresis no-trade bands)."""

import numpy as np
import pandas as pd
import pytest

from src.research.momentum_eval import (
    build_banded_quantile_weights,
    build_quantile_weights,
    run_light_backtest,
)


def _utc_index(n, freq="1h", start="2023-01-01"):
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC")


class TestBandZeroEquivalence:
    def test_band_zero_reproduces_memoryless_weights(self, multi_symbol_close_wide):
        signal = multi_symbol_close_wide.pct_change().dropna(how="all")
        banded = build_banded_quantile_weights(signal, 0.2, 0.2, band=0.0)
        plain = build_quantile_weights(signal, 0.2, 0.2)
        pd.testing.assert_frame_equal(banded, plain)


class TestHysteresis:
    @pytest.fixture
    def path_signal(self):
        """5 symbols; A's rank walks: top -> inside band -> below band.

        Ranks (pct of 5): values 1..5 map to 0.2..1.0. top_q=0.2 -> enter at
        rank >= 0.8, band=0.4 -> exit below 0.4.
        """
        idx = _utc_index(4)
        #        A    B    C    D    E
        data = [
            [5.0, 4.0, 3.0, 2.0, 1.0],   # A rank 1.0 -> enters long
            [3.0, 5.0, 4.0, 2.0, 1.0],   # A rank 0.6 -> inside band, stays long
            [2.0, 5.0, 4.0, 3.0, 1.0],   # A rank 0.4 -> still >= 0.4, stays
            [1.0, 5.0, 4.0, 3.0, 2.0],   # A rank 0.2 -> below 0.4 -> exits (enters short)
        ]
        return pd.DataFrame(data, index=idx, columns=list("ABCDE"))

    def test_membership_persists_inside_band(self, path_signal):
        w = build_banded_quantile_weights(path_signal, 0.2, 0.2, band=0.4)
        assert w.iloc[0]["A"] > 0
        assert w.iloc[1]["A"] > 0  # rank 0.6: below entry, above exit -> held
        assert w.iloc[2]["A"] > 0  # rank 0.4: exactly at exit threshold -> held
        assert w.iloc[3]["A"] < 0  # rank 0.2: below exit AND at short entry

    def test_without_band_same_path_churns(self, path_signal):
        w = build_banded_quantile_weights(path_signal, 0.2, 0.2, band=0.0)
        assert w.iloc[0]["A"] > 0
        assert w.iloc[1]["A"] == 0.0  # dropped immediately without hysteresis
        turnover_banded = (
            build_banded_quantile_weights(path_signal, 0.2, 0.2, band=0.4)
            .diff().abs().sum().sum()
        )
        turnover_plain = w.diff().abs().sum().sum()
        assert turnover_banded < turnover_plain

    def test_long_book_shares_renormalise(self, path_signal):
        w = build_banded_quantile_weights(path_signal, 0.2, 0.2, band=0.4)
        # Row 1 ranks: B=1.0 and C=0.8 enter long (entry threshold is >= 0.8);
        # A (rank 0.6) is held by the band -> three longs at 1/3 each.
        assert w.iloc[1]["A"] == pytest.approx(1 / 3)
        assert w.iloc[1]["B"] == pytest.approx(1 / 3)
        assert w.iloc[1]["C"] == pytest.approx(1 / 3)
        # Every row is dollar-neutral.
        np.testing.assert_allclose(w.sum(axis=1).to_numpy(), 0.0, atol=1e-12)


class TestNanHandling:
    def test_nan_name_is_forced_out(self):
        idx = _utc_index(3)
        sig = pd.DataFrame(
            {"A": [5.0, np.nan, np.nan], "B": [4.0, 4.0, 4.0], "C": [3.0, 3.0, 3.0],
             "D": [2.0, 2.0, 2.0], "E": [1.0, 1.0, 1.0]},
            index=idx,
        )
        w = build_banded_quantile_weights(sig, 0.2, 0.2, band=0.4)
        assert w.iloc[0]["A"] > 0
        assert w.iloc[1]["A"] == 0.0  # NaN -> out, despite the band

    def test_all_nan_row_carries_previous_weights(self):
        idx = _utc_index(3)
        sig = pd.DataFrame(
            {"A": [5.0, np.nan, 5.0], "B": [4.0, np.nan, 4.0], "C": [3.0, np.nan, 3.0],
             "D": [2.0, np.nan, 2.0], "E": [1.0, np.nan, 1.0]},
            index=idx,
        )
        w = build_banded_quantile_weights(sig, 0.2, 0.2, band=0.0)
        pd.testing.assert_series_equal(w.iloc[1], w.iloc[0], check_names=False)
        pd.testing.assert_series_equal(w.iloc[2], w.iloc[0], check_names=False)


class TestTurnoverReductionAndIntegration:
    def test_band_reduces_turnover_on_noisy_signal(self):
        # Random-walk signals give persistent ranks with boundary noise — the
        # regime hysteresis is built for (iid ranks churn regardless of band).
        rng = np.random.default_rng(5)
        idx = _utc_index(300)
        sig = pd.DataFrame(np.cumsum(rng.normal(0, 1, (300, 10)), axis=0), index=idx,
                           columns=[f"S{i}" for i in range(10)])
        t0 = build_banded_quantile_weights(sig, 0.2, 0.2, band=0.0).diff().abs().sum().sum()
        t1 = build_banded_quantile_weights(sig, 0.2, 0.2, band=0.2).diff().abs().sum().sum()
        assert t1 < 0.7 * t0

    def test_weights_wide_injection_runs(self):
        rng = np.random.default_rng(6)
        idx = _utc_index(60)
        sig = pd.DataFrame(rng.normal(0, 1, (60, 6)), index=idx,
                           columns=list("ABCDEF"))
        fwd = pd.DataFrame(rng.normal(0, 0.01, (60, 6)), index=idx,
                           columns=list("ABCDEF"))
        w = build_banded_quantile_weights(sig.iloc[::4], 0.2, 0.2, band=0.1)
        w_full = w.reindex(idx, method="ffill")
        bt = run_light_backtest(
            signal_wide=None, forward_return_wide=fwd, weights_wide=w_full,
            fee_bps=10.0, half_spread_bps=5.0, holding_period_bars=4,
        )
        assert bt["metrics"]["n_periods"] > 0
        assert np.isfinite(bt["metrics"]["sharpe_net"])

    def test_validation(self):
        idx = _utc_index(3)
        sig = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [3.0, 2.0, 1.0]}, index=idx)
        with pytest.raises(ValueError, match="band"):
            build_banded_quantile_weights(sig, 0.2, 0.2, band=-0.1)
        with pytest.raises(ValueError, match="quantile"):
            build_banded_quantile_weights(sig, 0.0, 0.2)

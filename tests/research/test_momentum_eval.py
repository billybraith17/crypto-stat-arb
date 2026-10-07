"""Unit tests for src/research/momentum_eval.py.

Each test constructs a small, controlled dataset so expected outputs are
derivable analytically. No database or filesystem access is required.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.research.momentum_eval import (
    _infer_periods_per_year,
    _newey_west_se,
    build_banded_book,
    build_beta_hedged_weights,
    build_quantile_weights,
    compute_ic_series,
    feature_ic_row,
    gross_sharpe,
    ic_summary,
    long_short_leg_returns,
    market_correlation,
    max_drawdown,
    portfolio_beta_series,
    quantile_analysis,
    realized_beta_diagnostics,
    rolling_mean,
    rolling_sharpe,
    run_light_backtest,
    walk_forward_sharpe_table,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_index(n, freq="4h", start="2023-01-01"):
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC")


def _wide(data: dict, n=10):
    return pd.DataFrame(data, index=_utc_index(n))


# ---------------------------------------------------------------------------
# _infer_periods_per_year
# ---------------------------------------------------------------------------

class TestInferPeriodsPerYear:
    def test_daily_index(self):
        idx = pd.date_range("2023-01-01", periods=100, freq="D", tz="UTC")
        ppy = _infer_periods_per_year(idx)
        assert abs(ppy - 365.0) < 1.0

    def test_hourly_index(self):
        idx = pd.date_range("2023-01-01", periods=200, freq="h", tz="UTC")
        ppy = _infer_periods_per_year(idx)
        assert abs(ppy - 365 * 24) < 1.0

    def test_short_index_returns_fallback(self):
        idx = pd.date_range("2023-01-01", periods=1, freq="D", tz="UTC")
        ppy = _infer_periods_per_year(idx)
        assert ppy == 365.0

    def test_irregular_index_emits_warning(self):
        """A highly irregular index should trigger a UserWarning."""
        idx = pd.DatetimeIndex(
            pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-10", "2023-01-11",
                            "2023-01-20", "2023-02-01", "2023-03-01"]).tz_localize("UTC")
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _infer_periods_per_year(idx)
            assert any(issubclass(warning.category, UserWarning) for warning in w)


# ---------------------------------------------------------------------------
# _newey_west_se
# ---------------------------------------------------------------------------

class TestNeweyWestSE:
    def test_lag0_matches_iid_se(self):
        """With lag=0, NW SE equals population std / sqrt(n) (NW uses ddof=0 internally)."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=200)
        n = len(x)
        x_dm = x - x.mean()
        # NW variance formula: sum(x^2) / n (biased, ddof=0)
        population_se = np.sqrt(np.dot(x_dm, x_dm) / n / n)
        nw_se = _newey_west_se(x, lag=0)
        assert abs(nw_se - population_se) < 1e-12

    def test_positive_autocorrelation_inflates_se(self):
        """Positive autocorrelation should make NW SE > iid SE."""
        rng = np.random.default_rng(1)
        innov = rng.normal(size=500)
        ar = np.zeros(500)
        ar[0] = innov[0]
        for i in range(1, 500):
            ar[i] = 0.8 * ar[i - 1] + innov[i]
        iid_se = ar.std(ddof=1) / np.sqrt(len(ar))
        nw_se = _newey_west_se(ar, lag=10)
        assert nw_se > iid_se

    def test_returns_nan_for_single_observation(self):
        assert np.isnan(_newey_west_se(np.array([1.0]), lag=0))

    def test_ignores_nan_values(self):
        x = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        result = _newey_west_se(x, lag=0)
        x_clean = np.array([1.0, 2.0, 3.0, 4.0])
        expected = _newey_west_se(x_clean, lag=0)
        assert abs(result - expected) < 1e-12


# ---------------------------------------------------------------------------
# compute_ic_series
# ---------------------------------------------------------------------------

class TestComputeICSeries:
    def _make_signal_and_fwd(self, n_bars=20, n_assets=8):
        """Construct a perfectly rank-correlated signal and forward return."""
        idx = _utc_index(n_bars)
        cols = [f"A{i}" for i in range(n_assets)]
        # Same ordering every bar: assets ranked 0..7
        signal = pd.DataFrame(
            np.tile(np.arange(n_assets, dtype=float), (n_bars, 1)),
            index=idx, columns=cols
        )
        fwd_ret = signal.copy()  # identical ordering → perfect correlation
        return signal, fwd_ret

    def test_perfect_rank_correlation_gives_ic_one(self):
        signal, fwd_ret = self._make_signal_and_fwd()
        ic = compute_ic_series(signal, fwd_ret, method="spearman", min_assets=6)
        assert np.allclose(ic.dropna().values, 1.0, atol=1e-10)

    def test_perfectly_inverse_signal_gives_ic_minus_one(self):
        signal, fwd_ret = self._make_signal_and_fwd()
        ic = compute_ic_series(-signal, fwd_ret, method="spearman", min_assets=6)
        assert np.allclose(ic.dropna().values, -1.0, atol=1e-10)

    def test_rows_below_min_assets_are_nan(self):
        idx = _utc_index(3)
        cols = ["A", "B", "C", "D", "E", "F"]
        # Row 0 has 6 valid pairs; row 1 has only 3 (below min_assets=6)
        sig = pd.DataFrame(
            [[1, 2, 3, 4, 5, 6], [1, 2, 3, np.nan, np.nan, np.nan], [1, 2, 3, 4, 5, 6]],
            index=idx, columns=cols, dtype=float
        )
        fwd = sig.copy()
        ic = compute_ic_series(sig, fwd, min_assets=6)
        assert np.isnan(ic.iloc[1])
        assert not np.isnan(ic.iloc[0])

    def test_constant_signal_gives_nan_ic(self):
        """All-identical signal has zero variance → IC is undefined (NaN)."""
        idx = _utc_index(5)
        cols = list("ABCDEFGH")
        sig = pd.DataFrame(1.0, index=idx, columns=cols)
        fwd = pd.DataFrame(
            np.random.default_rng(0).normal(size=(5, 8)), index=idx, columns=cols
        )
        ic = compute_ic_series(sig, fwd, min_assets=2)
        assert ic.isna().all()

    def test_raises_on_min_assets_below_2(self):
        sig = pd.DataFrame({"A": [1.0], "B": [2.0]},
                           index=_utc_index(1))
        with pytest.raises(ValueError, match="min_assets must be >= 2"):
            compute_ic_series(sig, sig, min_assets=1)

    @pytest.mark.parametrize("method", ["spearman", "pearson"])
    def test_matches_per_timestamp_pandas_correlation(self, method):
        """Row-wise result equals Series.corr on each timestamp's valid pairs."""
        rng = np.random.default_rng(3)
        idx = _utc_index(200)
        cols = [f"S{i}" for i in range(12)]
        sig = pd.DataFrame(rng.normal(size=(200, 12)), index=idx, columns=cols).round(1)
        fwd = pd.DataFrame(rng.normal(size=(200, 12)), index=idx, columns=cols)
        sig = sig.where(rng.random((200, 12)) < 0.7)
        fwd = fwd.where(rng.random((200, 12)) < 0.9)
        sig.iloc[5:8] = np.nan
        sig.iloc[10] = 1.0

        ic = compute_ic_series(sig, fwd, method=method, min_assets=6)

        expected = {}
        for ts in idx:
            pair = pd.concat([sig.loc[ts], fwd.loc[ts]], axis=1, keys=["s", "f"]).dropna()
            if pair.empty:
                continue
            if len(pair) < 6 or pair["s"].nunique() < 2 or pair["f"].nunique() < 2:
                expected[ts] = np.nan
            else:
                expected[ts] = pair["s"].corr(pair["f"], method=method)
        expected = pd.Series(expected)

        assert ic.index.equals(expected.index)
        assert ic.isna().sum() == expected.isna().sum() > 0
        np.testing.assert_allclose(ic.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12)

    def test_raises_on_unknown_method(self):
        sig = pd.DataFrame({"A": [1.0], "B": [2.0]}, index=_utc_index(1))
        with pytest.raises(ValueError, match="method must be"):
            compute_ic_series(sig, sig, method="kendall")


# ---------------------------------------------------------------------------
# ic_summary
# ---------------------------------------------------------------------------

class TestICSummary:
    def _ic(self, mean=0.05, std=0.10, n=100):
        rng = np.random.default_rng(42)
        vals = rng.normal(loc=mean, scale=std, size=n)
        return pd.Series(vals)

    def test_returns_correct_keys(self):
        ic = self._ic()
        result = ic_summary(ic)
        expected_keys = {
            "n_obs", "mean_ic", "median_ic", "std_ic",
            "t_stat_ic", "t_stat_ic_nw", "nw_lag", "p05_ic", "p95_ic"
        }
        assert set(result.keys()) == expected_keys

    def test_t_stat_formula(self):
        """t = mean / (std / sqrt(n)) with lag=0."""
        ic = self._ic(mean=0.05, std=0.10, n=100)
        result = ic_summary(ic, nw_lag=0)
        clean = ic.dropna()
        expected_t = clean.mean() / (clean.std(ddof=1) / np.sqrt(len(clean)))
        assert abs(result["t_stat_ic"] - expected_t) < 1e-10

    def test_nw_lag0_t_stats_are_equal(self):
        """When nw_lag=0, iid and NW t-stats must be identical."""
        ic = self._ic()
        result = ic_summary(ic, nw_lag=0)
        assert abs(result["t_stat_ic"] - result["t_stat_ic_nw"]) < 1e-10

    def test_nw_positive_lag_inflates_tstat(self):
        """Positive autocorrelation + NW lag should produce a lower |t| than iid."""
        rng = np.random.default_rng(7)
        innov = rng.normal(0.05, 0.10, size=300)
        ar = np.zeros(300)
        ar[0] = innov[0]
        for i in range(1, 300):
            ar[i] = 0.6 * ar[i - 1] + innov[i]
        ic = pd.Series(ar)
        iid = ic_summary(ic, nw_lag=0)
        nw = ic_summary(ic, nw_lag=5)
        assert abs(nw["t_stat_ic_nw"]) < abs(iid["t_stat_ic"])

    def test_empty_series_returns_nan(self):
        result = ic_summary(pd.Series(dtype=float))
        assert result["n_obs"] == 0
        assert np.isnan(result["mean_ic"])

    def test_percentiles_are_ordered(self):
        ic = self._ic()
        result = ic_summary(ic)
        assert result["p05_ic"] < result["p95_ic"]


# ---------------------------------------------------------------------------
# rolling_mean / rolling_sharpe
# ---------------------------------------------------------------------------

class TestRollingHelpers:
    def test_rolling_mean_window(self):
        # rolling_mean uses min_periods=int(0.9*window): with window=3 that is
        # 2 observations, so values appear from position 1 onwards.
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        rm = rolling_mean(s, window=3)
        assert np.isnan(rm.iloc[0])
        assert abs(rm.iloc[1] - 1.5) < 1e-12
        assert abs(rm.iloc[2] - 2.0) < 1e-12

    def test_rolling_sharpe_positive_for_upward_drift(self):
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0.01, 0.02, 200))
        rs = rolling_sharpe(returns, window=50, periods_per_year=252)
        assert rs.dropna().mean() > 0


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_monotonically_rising_series(self):
        cum = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
        mdd, dd = max_drawdown(cum)
        assert abs(mdd) < 1e-12

    def test_series_that_halves(self):
        cum = pd.Series([1.0, 0.5])
        mdd, _ = max_drawdown(cum)
        assert abs(mdd - (-0.5)) < 1e-12

    def test_drawdown_series_always_non_positive(self):
        cum = pd.Series([1.0, 1.2, 0.8, 1.1, 0.9, 1.3])
        _, dd = max_drawdown(cum)
        assert (dd <= 1e-12).all()


# ---------------------------------------------------------------------------
# build_quantile_weights
# ---------------------------------------------------------------------------

class TestBuildQuantileWeights:
    def _signal(self, n=20, m=10):
        """Random signal with n bars and m assets."""
        idx = _utc_index(n)
        return pd.DataFrame(
            np.random.default_rng(5).normal(size=(n, m)),
            index=idx,
            columns=[f"A{i}" for i in range(m)]
        )

    def test_dollar_neutral(self):
        """Long and short legs must sum to zero per row."""
        w = build_quantile_weights(self._signal(), top_quantile=0.2, bottom_quantile=0.2)
        row_sums = w.sum(axis=1)
        assert np.allclose(row_sums.values, 0.0, atol=1e-10)

    def test_long_weights_positive_short_negative(self):
        w = build_quantile_weights(self._signal())
        assert (w.max(axis=1) > 0).all()  # at least one long
        assert (w.min(axis=1) < 0).all()  # at least one short

    def test_long_sleeve_sums_to_one(self):
        w = build_quantile_weights(self._signal())
        long_sum = w.clip(lower=0).sum(axis=1)
        assert np.allclose(long_sum.values, 1.0, atol=1e-10)

    def test_short_sleeve_sums_to_minus_one(self):
        w = build_quantile_weights(self._signal())
        short_sum = w.clip(upper=0).sum(axis=1)
        assert np.allclose(short_sum.values, -1.0, atol=1e-10)

    def test_long_short_are_mutually_exclusive(self):
        """No asset can be both long and short simultaneously."""
        w = build_quantile_weights(self._signal())
        assert not ((w > 0) & (w < 0)).any().any()

    def test_raises_on_invalid_quantile(self):
        sig = self._signal()
        with pytest.raises(ValueError):
            build_quantile_weights(sig, top_quantile=0.0)
        with pytest.raises(ValueError):
            build_quantile_weights(sig, bottom_quantile=1.5)


# ---------------------------------------------------------------------------
# run_light_backtest
# ---------------------------------------------------------------------------

class TestRunLightBacktest:
    def _perfect_setup(self, n=30, m=6):
        """Signal and forward returns constructed so long leg always profits."""
        idx = _utc_index(n)
        cols = [f"A{i}" for i in range(m)]
        rng = np.random.default_rng(99)
        fwd = pd.DataFrame(rng.normal(0.01, 0.02, (n, m)), index=idx, columns=cols)
        # Signal = rank of fwd return → perfect foresight (sanity check only)
        signal = fwd.rank(axis=1, pct=True)
        return signal, fwd

    def test_returns_expected_metric_keys(self):
        sig, fwd = self._perfect_setup()
        result = run_light_backtest(sig, fwd, fee_bps=0, half_spread_bps=0)
        expected_keys = {
            "weights", "gross_returns", "net_returns", "turnover", "costs",
            "cum_gross", "cum_net", "drawdown", "metrics"
        }
        assert set(result.keys()) == expected_keys

    def test_zero_fees_gross_equals_net(self):
        sig, fwd = self._perfect_setup()
        result = run_light_backtest(sig, fwd, fee_bps=0, half_spread_bps=0)
        assert np.allclose(
            result["gross_returns"].values,
            result["net_returns"].values,
            atol=1e-12
        )

    def test_positive_costs_reduce_net(self):
        sig, fwd = self._perfect_setup()
        gross = run_light_backtest(sig, fwd, fee_bps=0, half_spread_bps=0)
        costly = run_light_backtest(sig, fwd, fee_bps=10, half_spread_bps=5)
        assert costly["metrics"]["mean_cost_per_period"] > 0
        assert costly["net_returns"].mean() < gross["net_returns"].mean()

    def test_execution_delay_shifts_weights(self):
        sig, fwd = self._perfect_setup()
        no_delay = run_light_backtest(sig, fwd, execution_delay_bars=0, fee_bps=0, half_spread_bps=0)
        delayed = run_light_backtest(sig, fwd, execution_delay_bars=1, fee_bps=0, half_spread_bps=0)
        # Delayed weights offset by one bar → different P&L; align on shared index to compare
        shared = no_delay["net_returns"].index.intersection(delayed["net_returns"].index)
        assert len(shared) > 0
        assert not np.allclose(
            no_delay["net_returns"].reindex(shared).fillna(0).values,
            delayed["net_returns"].reindex(shared).fillna(0).values,
        )

    def test_holding_period_h_reduces_period_count(self):
        sig, fwd = self._perfect_setup(n=30)
        h1 = run_light_backtest(sig, fwd, holding_period_bars=1, fee_bps=0, half_spread_bps=0)
        h3 = run_light_backtest(sig, fwd, holding_period_bars=3, fee_bps=0, half_spread_bps=0)
        assert h3["metrics"]["n_periods"] < h1["metrics"]["n_periods"]

    def test_cumulative_series_is_monotone_for_perfect_signal(self):
        """With a perfect-foresight signal and zero costs the book should never lose."""
        sig, fwd = self._perfect_setup()
        result = run_light_backtest(sig, fwd, fee_bps=0, half_spread_bps=0)
        # Cumulative product starts from 1+first_return; check it's always positive
        assert (result["cum_net"] > 0).all()
        # And final cum_net >= first element (monotone for a consistently positive signal)
        assert result["cum_net"].iloc[-1] >= result["cum_net"].iloc[0]

    def test_log_returns_flag_converts_inputs(self):
        """Passing log returns with the flag set should give different P&L than raw."""
        idx = _utc_index(20)
        cols = ["A", "B", "C", "D"]
        rng = np.random.default_rng(1)
        log_fwd = pd.DataFrame(rng.normal(0.01, 0.02, (20, 4)), index=idx, columns=cols)
        signal = log_fwd.rank(axis=1, pct=True)
        log_result = run_light_backtest(signal, log_fwd, returns_are_log=True, fee_bps=0, half_spread_bps=0)
        raw_result = run_light_backtest(signal, log_fwd, returns_are_log=False, fee_bps=0, half_spread_bps=0)
        assert not np.allclose(
            log_result["net_returns"].fillna(0).values,
            raw_result["net_returns"].fillna(0).values,
        )

    def test_sharpe_is_float(self):
        sig, fwd = self._perfect_setup()
        result = run_light_backtest(sig, fwd)
        assert isinstance(result["metrics"]["sharpe_net"], float)


# ---------------------------------------------------------------------------
# quantile_analysis
# ---------------------------------------------------------------------------

class TestQuantileAnalysis:
    def _monotone_setup(self, n=20, m=8):
        """Signal with a deterministic monotone cross-section so Q5 > Q1."""
        idx = _utc_index(n)
        cols = [f"A{i}" for i in range(m)]
        # Signal: clear ordering; fwd_ret scales with signal
        signal = pd.DataFrame(
            np.tile(np.arange(m, dtype=float), (n, 1)), index=idx, columns=cols
        )
        fwd_ret = signal / m  # positive and monotone
        return signal, fwd_ret

    def test_top_quantile_return_exceeds_bottom(self):
        sig, fwd = self._monotone_setup()
        qmeans, spread = quantile_analysis(sig, fwd, n_quantiles=5)
        top_ret = qmeans.loc[qmeans["quantile"] == 5, "mean_fwd_ret"].values[0]
        bot_ret = qmeans.loc[qmeans["quantile"] == 1, "mean_fwd_ret"].values[0]
        assert top_ret > bot_ret

    def test_spread_is_positive_for_monotone_signal(self):
        sig, fwd = self._monotone_setup()
        _, spread = quantile_analysis(sig, fwd, n_quantiles=5)
        assert spread.mean() > 0

    def test_returns_dataframe_and_series(self):
        sig, fwd = self._monotone_setup()
        qmeans, spread = quantile_analysis(sig, fwd)
        assert isinstance(qmeans, pd.DataFrame)
        assert isinstance(spread, pd.Series)

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame()
        qmeans, spread = quantile_analysis(empty, empty)
        assert qmeans.empty
        assert spread.empty

    def test_raises_on_single_quantile(self):
        sig, fwd = self._monotone_setup()
        with pytest.raises(ValueError):
            quantile_analysis(sig, fwd, n_quantiles=1)

    def test_log_returns_flag_converts_inputs(self):
        sig, fwd = self._monotone_setup()
        r_simple = quantile_analysis(sig, fwd, n_quantiles=3, returns_are_log=False)
        r_log = quantile_analysis(sig, np.log(1.0 + fwd), n_quantiles=3, returns_are_log=True)
        assert np.allclose(
            r_simple[0]["mean_fwd_ret"].values,
            r_log[0]["mean_fwd_ret"].values,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# long_short_leg_returns
# ---------------------------------------------------------------------------

class TestLongShortLegReturns:
    def test_sum_equals_combined_return(self):
        """long_ret + short_ret should equal the combined book return."""
        idx = _utc_index(10)
        cols = ["A", "B", "C", "D"]
        rng = np.random.default_rng(3)
        fwd = pd.DataFrame(rng.normal(0.01, 0.02, (10, 4)), index=idx, columns=cols)
        weights = pd.DataFrame(
            [[0.5, 0.5, -0.5, -0.5]] * 10, index=idx, columns=cols
        )
        long_ret, short_ret = long_short_leg_returns(weights, fwd)
        combined = (weights * fwd).sum(axis=1)
        assert np.allclose((long_ret + short_ret).values, combined.values, atol=1e-12)

    def test_long_leg_non_negative_for_positive_fwd(self):
        idx = _utc_index(5)
        cols = ["A", "B"]
        weights = pd.DataFrame({"A": [1.0] * 5, "B": [0.0] * 5}, index=idx)
        fwd = pd.DataFrame({"A": [0.01] * 5, "B": [-0.01] * 5}, index=idx)
        long_ret, _ = long_short_leg_returns(weights, fwd)
        assert (long_ret >= 0).all()

    def test_log_conversion_changes_values(self):
        idx = _utc_index(5)
        cols = ["A", "B"]
        fwd = pd.DataFrame({"A": [0.05] * 5, "B": [-0.03] * 5}, index=idx)
        weights = pd.DataFrame({"A": [1.0] * 5, "B": [-1.0] * 5}, index=idx)
        l_log, _ = long_short_leg_returns(weights, fwd, returns_are_log=True)
        l_raw, _ = long_short_leg_returns(weights, fwd, returns_are_log=False)
        assert not np.allclose(l_log.values, l_raw.values)


# ---------------------------------------------------------------------------
# market_correlation
# ---------------------------------------------------------------------------

class TestMarketCorrelation:
    def test_perfect_correlation(self):
        idx = _utc_index(20)
        s = pd.Series(np.arange(20.0), index=idx)
        assert abs(market_correlation(s, s) - 1.0) < 1e-10

    def test_perfect_negative_correlation(self):
        idx = _utc_index(20)
        s = pd.Series(np.arange(20.0), index=idx)
        assert abs(market_correlation(s, -s) - (-1.0)) < 1e-10

    def test_returns_nan_on_empty_overlap(self):
        idx_a = _utc_index(5, start="2023-01-01")
        idx_b = _utc_index(5, start="2023-02-01")
        a = pd.Series(np.ones(5), index=idx_a)
        b = pd.Series(np.ones(5), index=idx_b)
        assert np.isnan(market_correlation(a, b))


# ---------------------------------------------------------------------------
# Trading cadence: decimation is an IC device, holding_period_bars is the
# single backtest cadence mechanism
# ---------------------------------------------------------------------------

class TestCadenceSemantics:
    def _random_signal(self, index, columns, seed=7):
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            rng.normal(size=(len(index), len(columns))),
            index=index,
            columns=columns,
        )

    def test_decimation_redundant_when_phase_aligned(self, multi_symbol_close_wide):
        """With D == H and aligned phase, feeding the decimated signal to the
        backtest changes nothing — so the fresh signal is always safe."""
        from src.signals.cs_momentum import (
            apply_rebalance_decimation,
            compute_forward_returns,
        )

        close = multi_symbol_close_wide
        D = 2
        sig_fresh = self._random_signal(close.index, close.columns)
        sig_dec = apply_rebalance_decimation(sig_fresh, D)
        fwd = compute_forward_returns(close, holding_period_bars=D)

        kwargs = dict(fee_bps=0.0, half_spread_bps=0.0, holding_period_bars=D)
        bt_fresh = run_light_backtest(sig_fresh, fwd, **kwargs)
        bt_dec = run_light_backtest(sig_dec, fwd, **kwargs)

        pd.testing.assert_frame_equal(bt_fresh["weights"], bt_dec["weights"])
        pd.testing.assert_series_equal(bt_fresh["net_returns"], bt_dec["net_returns"])

    def test_mismatched_decimation_trades_stale_signal(self, multi_symbol_close_wide):
        """With D=2 decimation under H=3 stepping, the step at bar 3 trades the
        bar-2 signal — the phase-dependent staleness that motivated moving all
        backtests to the pre-decimation signal."""
        from src.signals.cs_momentum import (
            apply_rebalance_decimation,
            compute_forward_returns,
        )

        close = multi_symbol_close_wide
        D, H = 2, 3
        sig_fresh = self._random_signal(close.index, close.columns)
        sig_dec = apply_rebalance_decimation(sig_fresh, D)
        fwd = compute_forward_returns(close, holding_period_bars=H)

        bt_dec = run_light_backtest(
            sig_dec, fwd, fee_bps=0.0, half_spread_bps=0.0, holding_period_bars=H
        )
        fresh_weights = build_quantile_weights(sig_fresh)
        step_bar = close.index[3]

        stale = bt_dec["weights"].loc[step_bar].fillna(0.0)
        assert np.allclose(stale, fresh_weights.iloc[2].fillna(0.0))
        assert not np.allclose(stale, fresh_weights.iloc[3].fillna(0.0))


# ---------------------------------------------------------------------------
# portfolio_beta_series
# ---------------------------------------------------------------------------

def _const_panel(values: dict, n=4):
    idx = _utc_index(n)
    return pd.DataFrame({k: [v] * n for k, v in values.items()}, index=idx)


class TestPortfolioBetaSeries:
    def test_hand_computed_beta_and_full_coverage(self):
        weights = _const_panel({"AAA": 0.5, "BBB": 0.5, "CCC": -1.0})
        betas = _const_panel({"AAA": 1.0, "BBB": 2.0, "CCC": 1.0})
        out = portfolio_beta_series(weights, betas)
        # 0.5*1 + 0.5*2 - 1*1 = 0.5, measured on the full gross book.
        assert np.allclose(out["portfolio_beta"].values, 0.5, atol=1e-12)
        assert np.allclose(out["beta_coverage"].values, 1.0, atol=1e-12)

    def test_missing_beta_reduces_coverage(self):
        weights = _const_panel({"AAA": 0.5, "BBB": 0.5, "CCC": -1.0})
        betas = _const_panel({"AAA": 1.0, "BBB": np.nan, "CCC": 1.0})
        out = portfolio_beta_series(weights, betas)
        # Beta is summed over covered names only: 0.5*1 - 1*1 = -0.5;
        # covered gross = 0.5 + 1.0 of a total 2.0 -> coverage 0.75.
        assert np.allclose(out["portfolio_beta"].values, -0.5, atol=1e-12)
        assert np.allclose(out["beta_coverage"].values, 0.75, atol=1e-12)

    def test_unheld_symbol_with_nan_beta_does_not_hurt_coverage(self):
        weights = _const_panel({"AAA": 0.5, "BBB": 0.5, "CCC": -1.0, "DDD": 0.0})
        betas = _const_panel({"AAA": 1.0, "BBB": 2.0, "CCC": 1.0, "DDD": np.nan})
        out = portfolio_beta_series(weights, betas)
        assert np.allclose(out["beta_coverage"].values, 1.0, atol=1e-12)

    def test_empty_book_row_gives_nan_coverage_not_inf(self):
        # A bar with no positions (gross 0) must not divide-by-zero to inf.
        weights = _const_panel({"AAA": 0.0, "BBB": 0.0})
        betas = _const_panel({"AAA": 1.0, "BBB": 2.0})
        out = portfolio_beta_series(weights, betas)
        assert out["beta_coverage"].isna().all()
        assert np.isfinite(out["beta_coverage"].to_numpy()).sum() == 0


# ---------------------------------------------------------------------------
# realized_beta_diagnostics
# ---------------------------------------------------------------------------

def _regime_market(n_each=12):
    """Alternating up/down market with varying magnitudes (>= 2 distinct
    values per regime so subsample variances are positive)."""
    idx = _utc_index(2 * n_each)
    vals = []
    for i in range(n_each):
        vals.append(0.01 * (i % 3 + 1))   # up bars: 1%, 2%, 3%
        vals.append(-0.01 * (i % 3 + 1))  # down bars: -1%, -2%, -3%
    return pd.Series(vals, index=idx, name="m")


class TestRealizedBetaDiagnostics:
    def test_scaled_market_recovers_beta_everywhere(self):
        m = _regime_market()
        out = realized_beta_diagnostics(2.0 * m, m)
        assert np.isclose(out["beta_full"], 2.0, atol=1e-12)
        assert np.isclose(out["corr_full"], 1.0, atol=1e-12)
        assert np.isclose(out["beta_up"], 2.0, atol=1e-12)
        assert np.isclose(out["beta_down"], 2.0, atol=1e-12)
        assert out["n_obs"] == len(m)

    def test_asymmetric_exposure_split_by_regime(self):
        m = _regime_market()
        s = m.where(m > 0, 0.0)  # long the market in rallies, flat in selloffs
        out = realized_beta_diagnostics(s, m)
        assert np.isclose(out["beta_up"], 1.0, atol=1e-12)
        assert np.isclose(out["beta_down"], 0.0, atol=1e-12)
        # ... while the full-sample beta blends the two regimes.
        assert 0.0 < out["beta_full"] < 1.0

    def test_disjoint_indices_return_empty_without_crashing(self):
        s = pd.Series([0.01, 0.02, 0.03], index=_utc_index(3, start="2023-01-01"))
        m = pd.Series([0.01, 0.02, 0.03], index=_utc_index(3, start="2024-01-01"))
        out = realized_beta_diagnostics(s, m)
        assert out["n_obs"] == 0
        assert np.isnan(out["beta_full"]) and np.isnan(out["corr_full"])
        assert np.isnan(out["beta_up"]) and np.isnan(out["beta_down"])
        assert out["rolling_beta"].isna().all()

    def test_small_regime_returns_nan_with_counts(self):
        idx = _utc_index(6)
        m = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03, -0.03], index=idx)
        out = realized_beta_diagnostics(2.0 * m, m, min_regime_obs=10)
        assert np.isnan(out["beta_up"]) and np.isnan(out["beta_down"])
        assert out["n_up"] == 3 and out["n_down"] == 3
        assert np.isclose(out["beta_full"], 2.0, atol=1e-12)

    def test_rolling_beta_warmup_and_value(self):
        m = _regime_market()
        out = realized_beta_diagnostics(2.0 * m, m, rolling_window=8)
        rolling = out["rolling_beta"]
        assert rolling.iloc[:7].isna().all()
        assert np.allclose(rolling.iloc[7:].values, 2.0, atol=1e-12)


# ---------------------------------------------------------------------------
# build_beta_hedged_weights
# ---------------------------------------------------------------------------

class TestBuildBetaHedgedWeights:
    def _base(self):
        weights = _const_panel({"AAA": 0.5, "BBB": 0.5, "CCC": -1.0})
        betas = _const_panel(
            {"AAA": 1.0, "BBB": 2.0, "CCC": 1.0, "XBT/USD": 1.0}
        )
        return weights, betas

    def test_hedge_added_in_new_benchmark_column(self):
        weights, betas = self._base()
        hedged = build_beta_hedged_weights(weights, betas, normalize_gross=False)
        assert "XBT/USD" in hedged.columns
        # beta_p = 0.5, benchmark beta 1 -> hedge -0.5
        assert np.allclose(hedged["XBT/USD"].values, -0.5, atol=1e-12)
        # Original book untouched without normalization.
        pd.testing.assert_frame_equal(hedged[weights.columns], weights)

    def test_gross_normalized_to_pre_hedge_gross(self):
        weights, betas = self._base()
        hedged = build_beta_hedged_weights(weights, betas)
        # Pre-hedge gross 2.0; raw hedged gross 2.5 -> every weight scaled 0.8.
        assert np.allclose(hedged.abs().sum(axis=1).values, 2.0, atol=1e-12)
        assert np.allclose(hedged["XBT/USD"].values, -0.4, atol=1e-12)
        assert np.allclose(hedged["AAA"].values, 0.4, atol=1e-12)

    def test_normalization_preserves_zero_beta(self):
        weights, betas = self._base()
        hedged = build_beta_hedged_weights(weights, betas)
        out = portfolio_beta_series(hedged, betas)
        assert np.allclose(out["portfolio_beta"].values, 0.0, atol=1e-12)

    def test_normalize_gross_false_keeps_raw_overlay(self):
        weights, betas = self._base()
        hedged = build_beta_hedged_weights(weights, betas, normalize_gross=False)
        # Raw overlay gross = pre-hedge gross + |hedge| = 2.0 + 0.5.
        assert np.allclose(hedged.abs().sum(axis=1).values, 2.5, atol=1e-12)

    def test_hedged_book_has_zero_ex_ante_beta(self):
        weights, betas = self._base()
        hedged = build_beta_hedged_weights(weights, betas)
        out = portfolio_beta_series(hedged, betas)
        assert np.allclose(out["portfolio_beta"].values, 0.0, atol=1e-12)

    def test_existing_benchmark_position_nets_with_hedge(self):
        weights = _const_panel(
            {"AAA": 0.5, "BBB": 0.5, "CCC": -1.0, "XBT/USD": -0.2}
        )
        betas = _const_panel(
            {"AAA": 1.0, "BBB": 2.0, "CCC": 1.0, "XBT/USD": 1.0}
        )
        hedged = build_beta_hedged_weights(weights, betas, normalize_gross=False)
        # beta_p = 0.5 - 0.2 = 0.3 -> hedge -0.3, netting to -0.5 total.
        assert np.allclose(hedged["XBT/USD"].values, -0.5, atol=1e-12)
        out = portfolio_beta_series(hedged, betas)
        assert np.allclose(out["portfolio_beta"].values, 0.0, atol=1e-12)

    def test_max_hedge_weight_clips(self):
        weights, betas = self._base()
        hedged = build_beta_hedged_weights(
            weights, betas, max_hedge_weight=0.3, normalize_gross=False
        )
        assert np.allclose(hedged["XBT/USD"].values, -0.3, atol=1e-12)

    def test_missing_beta_fill_applied_to_held_positions(self):
        weights = _const_panel({"AAA": 1.0, "BBB": -1.0})
        betas = _const_panel({"AAA": 2.0, "BBB": np.nan, "XBT/USD": 1.0})
        hedged = build_beta_hedged_weights(
            weights, betas, missing_beta_fill=1.0, normalize_gross=False
        )
        # BBB beta filled with 1.0 -> beta_p = 2 - 1 = 1 -> hedge -1.
        assert np.allclose(hedged["XBT/USD"].values, -1.0, atol=1e-12)

    def test_raises_on_non_positive_cap(self):
        weights, betas = self._base()
        with pytest.raises(ValueError, match="max_hedge_weight"):
            build_beta_hedged_weights(weights, betas, max_hedge_weight=0.0)

    def test_time_varying_betas_hedged_per_bar(self):
        # Betas change bar to bar; the hedge must neutralise each bar on its own.
        idx = _utc_index(4)
        weights = pd.DataFrame(
            {"AAA": [1.0] * 4, "BBB": [-1.0] * 4}, index=idx
        )
        betas = pd.DataFrame(
            {"AAA": [1.0, 1.5, 2.0, 0.5],
             "BBB": [1.0, 1.0, 1.0, 1.0],
             "XBT/USD": [1.0, 1.0, 1.0, 1.0]},
            index=idx,
        )
        hedged = build_beta_hedged_weights(weights, betas)
        out = portfolio_beta_series(hedged, betas)
        assert np.allclose(out["portfolio_beta"].values, 0.0, atol=1e-12)
        # Raw (un-normalised) hedge equals -(beta_p) each bar: -(1-1), -(1.5-1)...
        raw = build_beta_hedged_weights(weights, betas, normalize_gross=False)
        assert np.allclose(raw["XBT/USD"].values, [0.0, -0.5, -1.0, 0.5], atol=1e-12)

    def test_normalize_gross_empty_book_row_is_safe(self):
        # A flat bar (no positions, no hedge) must not divide-by-zero.
        weights = _const_panel({"AAA": 0.0, "BBB": 0.0})
        betas = _const_panel({"AAA": 1.0, "BBB": 2.0, "XBT/USD": 1.0})
        hedged = build_beta_hedged_weights(weights, betas)
        assert np.isfinite(hedged.to_numpy()).all()
        assert np.allclose(hedged.to_numpy(), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# run_light_backtest with externally built weights (weights_wide)
# ---------------------------------------------------------------------------

class TestRunLightBacktestWeightsWide:
    def test_equivalence_with_internal_quantile_weights(self):
        idx = _utc_index(30)
        cols = [f"A{i}" for i in range(6)]
        rng = np.random.default_rng(99)
        fwd = pd.DataFrame(rng.normal(0.01, 0.02, (30, 6)), index=idx, columns=cols)
        signal = fwd.rank(axis=1, pct=True)
        via_signal = run_light_backtest(signal, fwd, fee_bps=10, half_spread_bps=2)
        via_weights = run_light_backtest(
            None, fwd, fee_bps=10, half_spread_bps=2,
            weights_wide=build_quantile_weights(signal),
        )
        pd.testing.assert_series_equal(
            via_signal["net_returns"], via_weights["net_returns"]
        )
        assert via_signal["metrics"] == via_weights["metrics"]

    def test_raises_without_signal_or_weights(self):
        idx = _utc_index(4)
        fwd = pd.DataFrame({"AAA": [0.01] * 4}, index=idx)
        with pytest.raises(ValueError, match="signal_wide or weights_wide"):
            run_light_backtest(None, fwd)

    def _hedged_setup(self):
        """4-bar book: constant weights for 2 bars, then the hedge unwinds."""
        idx = _utc_index(4)
        w = pd.DataFrame(
            {
                "AAA": [0.5, 0.5, 1.0, 1.0],
                "BBB": [0.5, 0.5, 0.0, 0.0],
                "CCC": [-1.0, -1.0, -1.0, -1.0],
                "XBT/USD": [-0.5, -0.5, 0.0, 0.0],
            },
            index=idx,
        )
        fwd = pd.DataFrame(
            {
                "AAA": [0.02] * 4,
                "BBB": [0.00] * 4,
                "CCC": [-0.01] * 4,
                "XBT/USD": [0.01] * 4,
            },
            index=idx,
        )
        return w, fwd

    def test_hand_computed_gross_pnl_and_exposure(self):
        w, fwd = self._hedged_setup()
        bt = run_light_backtest(None, fwd, weights_wide=w, fee_bps=0, half_spread_bps=0)
        # Bar 1: 0.5*0.02 + 0.5*0 + (-1)*(-0.01) + (-0.5)*0.01 = 0.015
        assert np.isclose(bt["gross_returns"].iloc[0], 0.015, atol=1e-12)
        # Gross exposure: 2.5 on hedged bars, 2.0 after the hedge unwinds.
        assert np.isclose(bt["metrics"]["mean_gross_exposure"], 2.25, atol=1e-12)

    def test_hedge_turnover_charged_flat(self):
        w, fwd = self._hedged_setup()
        bt = run_light_backtest(None, fwd, weights_wide=w, fee_bps=10, half_spread_bps=0)
        # Bar 3 rebalance: traded notional |1-0.5| + |0-0.5| + 0 + |0-(-0.5)|
        # = 1.5 (0.5 of it the hedge unwind); one-sided turnover = 0.75.
        assert np.isclose(bt["turnover"].iloc[2], 0.75, atol=1e-12)
        assert np.isclose(bt["costs"].iloc[2], 1.5 * 10e-4, atol=1e-15)

    def test_hedge_turnover_charged_at_benchmark_spread(self):
        w, fwd = self._hedged_setup()
        spreads = pd.Series(
            {"AAA": 10.0, "BBB": 10.0, "CCC": 10.0, "XBT/USD": 0.0}
        )
        bt = run_light_backtest(
            None, fwd, weights_wide=w, fee_bps=0, half_spread_bps=spreads
        )
        # Bar 3: AAA 0.5 and BBB 0.5 traded at 10 bps; the 0.5 XBT hedge trade
        # at its own 0 bps -> 1.0 * 10e-4, not 1.5 * 10e-4.
        assert np.isclose(bt["costs"].iloc[2], 1.0 * 10e-4, atol=1e-15)


class TestOneWayCostConvention:
    """Hand-priced rebalances: fee_bps and half_spread_bps are one-way rates
    paid on every fill, so a full book rotation pays them on 4 units of
    notional per unit of gross-2 capital."""

    def _rotation(self):
        idx = _utc_index(3, freq="1h")
        cols = ["A", "B", "C", "D"]
        w = pd.DataFrame(
            [[0.0, 0.0, 0.0, 0.0],
             [1.0, -1.0, 0.0, 0.0],   # enter: buy A, sell B
             [0.0, 0.0, 1.0, -1.0]],  # rotate: sell A, buy C, buy B, sell D
            index=idx, columns=cols,
        )
        fwd = pd.DataFrame(0.0, index=idx, columns=cols)
        return w, fwd

    def test_full_rotation_pays_one_way_rate_on_every_fill(self):
        w, fwd = self._rotation()
        bt = run_light_backtest(
            None, fwd, weights_wide=w, fee_bps=40, half_spread_bps=10,
            periods_per_year=8760,
        )
        # Entry: 2 fills x 50 bps; rotation: 4 fills x 50 bps.
        assert np.allclose(bt["costs"].to_numpy(), [0.0100, 0.0200], atol=1e-15)
        # Turnover is reported one-sided: 1 on entry, 2 on the rotation.
        assert np.allclose(bt["turnover"].to_numpy(), [1.0, 2.0], atol=1e-15)
        assert np.allclose(bt["net_returns"].to_numpy(), [-0.0100, -0.0200], atol=1e-15)

    def test_per_asset_spreads_charged_per_fill(self):
        w, fwd = self._rotation()
        spreads = pd.Series({"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0})
        bt = run_light_backtest(
            None, fwd, weights_wide=w, fee_bps=0, half_spread_bps=spreads,
            periods_per_year=8760,
        )
        # Entry: A 10 + B 20 = 30 bps; rotation: 10 + 20 + 30 + 40 = 100 bps.
        assert np.allclose(bt["costs"].to_numpy(), [0.0030, 0.0100], atol=1e-15)


class TestFeatureICRow:
    def _data(self):
        idx = _utc_index(30, freq="1h")
        rng = np.random.default_rng(11)
        cols = [f"S{i}" for i in range(8)]
        feat = pd.DataFrame(rng.normal(size=(30, 8)), index=idx, columns=cols)
        fwd = pd.DataFrame(rng.normal(size=(30, 8)), index=idx, columns=cols)
        return feat, fwd

    def test_reference_only_equals_ic_summary(self):
        feat, fwd = self._data()
        row = feature_ic_row(feat, fwd, nw_lag=2, min_assets=4)
        assert row == ic_summary(compute_ic_series(feat, fwd, min_assets=4), nw_lag=2)
        assert "mean_ic_trad" not in row

    def test_identical_tradable_returns_zero_haircut(self):
        feat, fwd = self._data()
        row = feature_ic_row(feat, fwd, nw_lag=1, min_assets=4, tradable_forward_return_wide=fwd)
        assert row["mean_ic_trad"] == row["mean_ic"]
        assert row["t_nw_trad"] == row["t_stat_ic_nw"]
        assert row["ic_haircut"] == 0.0


class TestGrossSharpe:
    def test_hand_computed(self):
        idx = _utc_index(4, freq="1h")
        bt = {
            "gross_returns": pd.Series([0.01, 0.03, 0.01, 0.03], index=idx),
            "metrics": {"periods_per_year": 100.0},
        }
        # mean 0.02, sample sd 0.011547 -> 10 * 0.02 / 0.011547
        assert np.isclose(gross_sharpe(bt), 10 * 0.02 / np.std([0.01, 0.03, 0.01, 0.03], ddof=1))

    def test_zero_variance_is_nan(self):
        idx = _utc_index(3, freq="1h")
        bt = {"gross_returns": pd.Series([0.01] * 3, index=idx), "metrics": {"periods_per_year": 1.0}}
        assert np.isnan(gross_sharpe(bt))


class TestBuildBandedBook:
    def _signal(self, n=12, m=10, seed=5):
        idx = _utc_index(n, freq="1h")
        rng = np.random.default_rng(seed)
        return pd.DataFrame(rng.normal(size=(n, m)), index=idx, columns=[f"S{i}" for i in range(m)])

    def test_weights_change_only_on_decision_rows(self):
        sig = self._signal()
        book = build_banded_book(sig, holding_period_bars=4, band=0.1)
        assert book.index.equals(sig.index)
        for start in range(0, 12, 4):
            block = book.iloc[start:start + 4]
            assert (block.eq(block.iloc[0], axis=1)).all().all()

    def test_band_zero_matches_quantile_weights_on_decision_rows(self):
        sig = self._signal()
        book = build_banded_book(sig, holding_period_bars=3, band=0.0)
        expected = build_quantile_weights(sig.iloc[::3]).fillna(0.0)
        pd.testing.assert_frame_equal(book.iloc[::3], expected, check_freq=False)


class TestWalkForwardSharpeTable:
    def test_fold_sharpes_and_embargo(self):
        idx = _utc_index(20, freq="1D")
        returns = pd.Series(np.tile([0.01, 0.03], 10), index=idx)
        table = walk_forward_sharpe_table(returns, periods_per_year=365, n_folds=2, embargo_obs=2)
        assert table["n_obs"].tolist() == [10, 8]
        expected = np.sqrt(365) * 0.02 / np.std([0.01, 0.03] * 5, ddof=1)
        assert np.isclose(table.loc[1, "sharpe_net"], expected)

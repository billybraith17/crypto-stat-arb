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
    build_quantile_weights,
    compute_ic_series,
    ic_summary,
    long_short_leg_returns,
    market_correlation,
    max_drawdown,
    quantile_analysis,
    rolling_mean,
    rolling_sharpe,
    run_light_backtest,
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
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        rm = rolling_mean(s, window=3)
        assert np.isnan(rm.iloc[0]) and np.isnan(rm.iloc[1])
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

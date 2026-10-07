"""Unit tests for src/signals/cs_momentum.py.

Each test exercises a single function with a small, deterministic DataFrame
whose expected output can be derived analytically without running the full
pipeline. No database or filesystem access is required.
"""

import numpy as np
import pandas as pd
import pytest

from src.signals.cs_momentum import (
    _validate_long_panel,
    apply_rebalance_decimation,
    build_feature_panels,
    build_market_index_returns,
    build_momentum_signal,
    build_monthly_universe_mask,
    build_relative_momentum_features,
    build_residual_momentum_features,
    build_residual_return_panel,
    build_vol_adjusted_features,
    compute_bar_returns,
    compute_forward_returns,
    compute_return_horizons,
    cross_sectional_rank_or_zscore,
    estimate_rolling_betas,
    resample_to_signal_timeframe,
    rolling_mean_std,
    rolling_momentum_score,
    select_momentum_feature_panel,
    to_simple_returns,
)

# ---------------------------------------------------------------------------
# _validate_long_panel
# ---------------------------------------------------------------------------

class TestValidateLongPanel:
    def test_passes_on_valid_panel(self, long_panel_df):
        _validate_long_panel(long_panel_df)  # should not raise

    def test_raises_on_missing_column(self, long_panel_df):
        with pytest.raises(ValueError, match="Missing required columns"):
            _validate_long_panel(long_panel_df.drop(columns=["close"]))

    def test_raises_on_duplicate_ts_symbol(self, long_panel_df):
        duped = pd.concat([long_panel_df, long_panel_df.iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            _validate_long_panel(duped)

    def test_empty_dataframe_passes(self):
        empty = pd.DataFrame(columns=["ts", "symbol", "close"])
        _validate_long_panel(empty)  # should not raise

    def test_raises_on_multiple_missing_columns(self):
        df = pd.DataFrame({"ts": [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            _validate_long_panel(df)


# ---------------------------------------------------------------------------
# resample_to_signal_timeframe
# ---------------------------------------------------------------------------

class TestResampleToSignalTimeframe:
    def test_identity_when_already_at_signal_freq(self, long_panel_df):
        """Resampling hourly data to the same hourly frequency is a no-op."""
        result = resample_to_signal_timeframe(long_panel_df, "h")
        assert result.shape[1] == 3
        assert set(result.columns) == {"AAA", "BBB", "CCC"}

    def test_output_is_wide_format(self, long_panel_df):
        result = resample_to_signal_timeframe(long_panel_df, "4h")
        assert isinstance(result, pd.DataFrame)
        assert result.index.tz is not None  # UTC-aware

    def test_raises_on_duplicate_rows(self, long_panel_df):
        duped = pd.concat([long_panel_df, long_panel_df.iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError):
            resample_to_signal_timeframe(duped, "4h")


# ---------------------------------------------------------------------------
# compute_bar_returns
# ---------------------------------------------------------------------------

class TestComputeBarReturns:
    def test_log_returns_double_prices(self, simple_close_wide):
        """Doubling prices each bar → log return = ln(2) everywhere."""
        result = compute_bar_returns(simple_close_wide, log_returns=True)
        ln2 = np.log(2.0)
        non_nan = result.iloc[1:]
        assert np.allclose(non_nan.values, ln2, atol=1e-12)

    def test_first_row_is_nan(self, simple_close_wide):
        result = compute_bar_returns(simple_close_wide, log_returns=True)
        assert result.iloc[0].isna().all()

    def test_simple_returns_double_prices(self, simple_close_wide):
        result = compute_bar_returns(simple_close_wide, log_returns=False)
        non_nan = result.iloc[1:]
        assert np.allclose(non_nan.values, 1.0, atol=1e-12)  # 100% return per bar

    def test_shape_preserved(self, simple_close_wide):
        result = compute_bar_returns(simple_close_wide)
        assert result.shape == simple_close_wide.shape


class TestToSimpleReturns:
    def test_log_input_converts_exactly(self, simple_close_wide):
        log_ret = compute_bar_returns(simple_close_wide, log_returns=True)
        simple_ret = compute_bar_returns(simple_close_wide, log_returns=False)
        converted = to_simple_returns(log_ret, log_returns=True)
        pd.testing.assert_frame_equal(converted, simple_ret)

    def test_simple_input_is_noop(self, simple_close_wide):
        simple_ret = compute_bar_returns(simple_close_wide, log_returns=False)
        result = to_simple_returns(simple_ret, log_returns=False)
        assert result is simple_ret

    def test_log_and_simple_consistent(self, multi_symbol_close_wide):
        """exp(log_return) - 1 should equal simple_return."""
        log_ret = compute_bar_returns(multi_symbol_close_wide, log_returns=True)
        simple_ret = compute_bar_returns(multi_symbol_close_wide, log_returns=False)
        mask = (log_ret.notna() & simple_ret.notna()).values
        log_vals = log_ret.values[mask]
        simple_vals = simple_ret.values[mask]
        assert np.allclose(np.exp(log_vals) - 1.0, simple_vals, atol=1e-12)


# ---------------------------------------------------------------------------
# compute_return_horizons
# ---------------------------------------------------------------------------

class TestComputeReturnHorizons:
    def test_horizon_1_matches_bar_returns(self, simple_close_wide):
        bar_ret = compute_bar_returns(simple_close_wide, log_returns=True)
        horizons = compute_return_horizons(simple_close_wide, horizons=[1], log_returns=True)
        assert np.allclose(
            bar_ret.fillna(0).values, horizons[1].fillna(0).values, atol=1e-12
        )

    def test_multiple_horizons_returned(self, simple_close_wide):
        horizons = compute_return_horizons(simple_close_wide, horizons=[1, 2, 3])
        assert set(horizons.keys()) == {1, 2, 3}

    def test_horizon_2_log_return(self, simple_close_wide):
        """H=2 with doubling prices → 2*ln(2) at each non-NaN bar."""
        horizons = compute_return_horizons(simple_close_wide, horizons=[2], log_returns=True)
        valid = horizons[2].iloc[2:]  # first 2 rows are NaN
        assert np.allclose(valid.values, 2 * np.log(2.0), atol=1e-12)

    def test_raises_on_non_positive_horizon(self, simple_close_wide):
        with pytest.raises(ValueError):
            compute_return_horizons(simple_close_wide, horizons=[0])

    def test_first_h_rows_are_nan(self, simple_close_wide):
        horizons = compute_return_horizons(simple_close_wide, horizons=[2])
        assert horizons[2].iloc[:2].isna().all().all()

    def test_skip_bars_shifts_horizon_window(self, simple_close_wide):
        horizons = compute_return_horizons(
            simple_close_wide, horizons=[2], log_returns=True, skip_bars=1
        )
        expected = np.log(simple_close_wide.shift(1) / simple_close_wide.shift(3))
        assert np.allclose(
            horizons[2].fillna(0).values, expected.fillna(0).values, atol=1e-12
        )


# ---------------------------------------------------------------------------
# rolling_momentum_score
# ---------------------------------------------------------------------------

class TestRollingMomentumScore:
    def test_lookback_1_equals_input(self, simple_close_wide):
        """With lookback=1, momentum is just the single-period return."""
        ret = compute_bar_returns(simple_close_wide, log_returns=True)
        score = rolling_momentum_score(ret, lookback_bars=1, log_returns=True)
        mask = ret.notna().values
        assert np.allclose(score.values[mask], ret.values[mask], atol=1e-12)

    def test_nan_before_min_periods(self, simple_close_wide):
        """With lookback=3, first 3 rows must be NaN (min_periods=lookback_bars)."""
        ret = compute_bar_returns(simple_close_wide, log_returns=True)
        score = rolling_momentum_score(ret, lookback_bars=3, log_returns=True)
        # ret has NaN at row 0; lookback=3 means rows 0..2 are NaN, row 3 is not
        assert score.iloc[:3].isna().all().all()
        assert score.iloc[3].notna().all()

    def test_log_score_is_sum_of_log_returns(self, multi_symbol_close_wide):
        """Log momentum score equals sum of constituent log returns."""
        ret = compute_bar_returns(multi_symbol_close_wide, log_returns=True)
        score = rolling_momentum_score(ret, lookback_bars=3, log_returns=True)
        # At the last bar, score should equal ret[-1] + ret[-2] + ret[-3]
        manual = ret.iloc[-3:].sum(axis=0)
        assert np.allclose(score.iloc[-1].values, manual.values, atol=1e-12)

    def test_raises_on_non_positive_lookback(self, simple_close_wide):
        ret = compute_bar_returns(simple_close_wide)
        with pytest.raises(ValueError):
            rolling_momentum_score(ret, lookback_bars=0)

    def test_skip_bars_excludes_most_recent_returns(self, simple_close_wide):
        ret = compute_bar_returns(simple_close_wide, log_returns=True)
        score = rolling_momentum_score(
            ret, lookback_bars=1, log_returns=True, skip_bars=1
        )
        expected = ret.shift(1)
        assert np.allclose(score.fillna(0).values, expected.fillna(0).values, atol=1e-12)


# ---------------------------------------------------------------------------
# build_relative_momentum_features
# ---------------------------------------------------------------------------

class TestBuildRelativeMomentumFeatures:
    def _returns(self, close_wide):
        return compute_bar_returns(close_wide, log_returns=True).dropna()

    def test_benchmark_minus_itself_is_zero(self, multi_symbol_close_wide):
        ret = self._returns(multi_symbol_close_wide)
        features = build_relative_momentum_features(
            ret, benchmark_symbol="AAA", residual_space="log", log_returns=True
        )
        benchmark_col = features["minus_benchmark"]["AAA"]
        assert np.allclose(benchmark_col.dropna().values, 0.0, atol=1e-12)

    def test_minus_xsec_mean_has_zero_cross_sectional_mean(self, multi_symbol_close_wide):
        ret = self._returns(multi_symbol_close_wide)
        features = build_relative_momentum_features(
            ret, benchmark_symbol="AAA", residual_space="log", log_returns=True
        )
        row_means = features["minus_xsec_mean"].mean(axis=1)
        assert np.allclose(row_means.dropna().values, 0.0, atol=1e-10)

    def test_raises_on_missing_benchmark(self, multi_symbol_close_wide):
        ret = self._returns(multi_symbol_close_wide)
        with pytest.raises(KeyError, match="benchmark_symbol"):
            build_relative_momentum_features(ret, benchmark_symbol="MISSING")

    def test_raises_on_invalid_residual_space(self, multi_symbol_close_wide):
        ret = self._returns(multi_symbol_close_wide)
        with pytest.raises(ValueError, match="residual_space"):
            build_relative_momentum_features(ret, benchmark_symbol="AAA", residual_space="bad")

    def test_universe_mask_restricts_mean_computation(self, multi_symbol_close_wide):
        """When only one symbol is in mask, minus_xsec_mean at that row equals 0."""
        ret = self._returns(multi_symbol_close_wide)
        mask = pd.DataFrame(False, index=ret.index, columns=ret.columns)
        mask["AAA"] = True
        features = build_relative_momentum_features(
            ret, benchmark_symbol="AAA", residual_space="log",
            log_returns=True, universe_mask=mask
        )
        # With universe_mask={AAA}, xsec_mean = AAA return, so AAA residual = 0
        assert np.allclose(features["minus_xsec_mean"]["AAA"].dropna().values, 0.0, atol=1e-10)

    def test_simple_space_converts_log_inputs(self, multi_symbol_close_wide):
        """In simple residual_space the output must differ from log-space."""
        ret = self._returns(multi_symbol_close_wide)
        log_feat = build_relative_momentum_features(
            ret, benchmark_symbol="AAA", residual_space="log"
        )
        simple_feat = build_relative_momentum_features(
            ret, benchmark_symbol="AAA", residual_space="simple", log_returns=True
        )
        # They should not be equal (conversion changes the values)
        assert not np.allclose(
            log_feat["minus_xsec_mean"].fillna(0).values,
            simple_feat["minus_xsec_mean"].fillna(0).values,
        )


# ---------------------------------------------------------------------------
# build_vol_adjusted_features
# ---------------------------------------------------------------------------

class TestBuildVolAdjustedFeatures:
    def test_output_keys(self, multi_symbol_close_wide):
        ret = compute_bar_returns(multi_symbol_close_wide).dropna()
        out = build_vol_adjusted_features(ret, vol_window_bars=4)
        assert set(out.keys()) == {"rolling_vol", "return_over_vol", "zscored_return"}

    def test_rolling_vol_non_negative(self, multi_symbol_close_wide):
        ret = compute_bar_returns(multi_symbol_close_wide).dropna()
        vol = build_vol_adjusted_features(ret, vol_window_bars=4)["rolling_vol"]
        assert (vol.dropna() >= 0).all().all()

    def test_zscored_differs_from_return_over_vol(self, multi_symbol_close_wide):
        """z-scored (demean then scale) and return-over-vol (scale only) must differ."""
        ret = compute_bar_returns(multi_symbol_close_wide).dropna()
        out = build_vol_adjusted_features(ret, vol_window_bars=4)
        z = out["zscored_return"].dropna(how="all")
        rov = out["return_over_vol"].dropna(how="all")
        # They are equal only if the rolling mean is zero; for arbitrary returns they differ
        assert not np.allclose(z.fillna(0).values, rov.fillna(0).values)

    def test_raises_on_window_too_small(self, multi_symbol_close_wide):
        ret = compute_bar_returns(multi_symbol_close_wide).dropna()
        with pytest.raises(ValueError):
            build_vol_adjusted_features(ret, vol_window_bars=1)

    def test_return_over_vol_nan_where_vol_zero(self):
        """A constant-price series has zero vol; return_over_vol must be NaN."""
        idx = pd.date_range("2023-01-01", periods=10, freq="h", tz="UTC")
        df = pd.DataFrame({"A": np.zeros(10)}, index=idx)
        out = build_vol_adjusted_features(df, vol_window_bars=4)
        assert out["return_over_vol"].dropna().shape[0] == 0


# ---------------------------------------------------------------------------
# build_monthly_universe_mask
# ---------------------------------------------------------------------------

class TestBuildMonthlyUniverseMask:
    def test_none_universe_returns_none(self, multi_symbol_close_wide):
        result = build_monthly_universe_mask(
            multi_symbol_close_wide.index, multi_symbol_close_wide.columns, universe_df=None
        )
        assert result is None

    def test_empty_universe_returns_all_false(self, multi_symbol_close_wide):
        empty = pd.DataFrame(columns=["rebalance_date", "symbol"])
        result = build_monthly_universe_mask(
            multi_symbol_close_wide.index, multi_symbol_close_wide.columns, universe_df=empty
        )
        assert isinstance(result, pd.DataFrame)
        assert not result.any().any()

    def test_forward_fill_behaviour(self, universe_df):
        """After the first rebalance_date+1day, AAA and BBB should be True."""
        idx = pd.date_range("2023-01-02", periods=5, freq="D", tz="UTC")
        cols = ["AAA", "BBB", "CCC"]
        mask = build_monthly_universe_mask(idx, cols, universe_df)
        # Jan rebalance: AAA+BBB in universe; CCC not until Feb
        assert mask.loc[idx[0], "AAA"]
        assert mask.loc[idx[0], "BBB"]
        assert not mask.loc[idx[0], "CCC"]

    def test_second_rebalance_adds_new_symbol(self, universe_df):
        """After Feb rebalance + 1 day, AAA and CCC should be True."""
        idx = pd.date_range("2023-02-02", periods=3, freq="D", tz="UTC")
        cols = ["AAA", "BBB", "CCC"]
        mask = build_monthly_universe_mask(idx, cols, universe_df)
        assert mask.loc[idx[0], "AAA"]
        assert mask.loc[idx[0], "CCC"]

    def test_output_is_bool_dtype(self, universe_df):
        idx = pd.date_range("2023-01-02", periods=4, freq="D", tz="UTC")
        mask = build_monthly_universe_mask(idx, ["AAA", "BBB", "CCC"], universe_df)
        assert mask.dtypes.apply(pd.api.types.is_bool_dtype).all()

    def test_raises_on_missing_required_columns(self):
        bad_df = pd.DataFrame({"symbol": ["AAA"]})
        idx = pd.date_range("2023-01-01", periods=2, freq="D", tz="UTC")
        with pytest.raises(ValueError, match="missing required columns"):
            build_monthly_universe_mask(idx, ["AAA"], bad_df)


# ---------------------------------------------------------------------------
# cross_sectional_rank_or_zscore
# ---------------------------------------------------------------------------

class TestCrossSectionalRankOrZscore:
    def test_zscore_cross_sectional_mean_is_zero(self, multi_symbol_close_wide):
        ret = compute_bar_returns(multi_symbol_close_wide).dropna()
        z = cross_sectional_rank_or_zscore(ret, method="zscore", min_assets_per_timestamp=2)
        row_means = z.mean(axis=1).dropna()
        assert np.allclose(row_means.values, 0.0, atol=1e-10)

    def test_zscore_cross_sectional_std_is_one(self, multi_symbol_close_wide):
        """ddof=0 std used, so population std should equal 1."""
        ret = compute_bar_returns(multi_symbol_close_wide).dropna()
        z = cross_sectional_rank_or_zscore(ret, method="zscore", min_assets_per_timestamp=2)
        row_stds = z.std(axis=1, ddof=0).dropna()
        assert np.allclose(row_stds.values, 1.0, atol=1e-10)

    def test_rank_output_in_zero_one(self, multi_symbol_close_wide):
        ret = compute_bar_returns(multi_symbol_close_wide).dropna()
        ranked = cross_sectional_rank_or_zscore(ret, method="rank", min_assets_per_timestamp=2)
        vals = ranked.stack().dropna().values
        assert (vals >= 0.0).all() and (vals <= 1.0).all()

    def test_rows_below_min_assets_are_nan(self):
        """A row with fewer valid entries than min_assets must be entirely NaN."""
        idx = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")
        # Row 0 has 5 valid values (sufficient); row 1 has only 2 (below threshold of 6)
        data = {
            "A": [1.0, 1.0, 1.0],
            "B": [2.0, 2.0, 2.0],
            "C": [3.0, np.nan, 3.0],
            "D": [4.0, np.nan, 4.0],
            "E": [5.0, np.nan, 5.0],
            "F": [6.0, np.nan, 6.0],
        }
        df = pd.DataFrame(data, index=idx)
        z = cross_sectional_rank_or_zscore(df, method="zscore", min_assets_per_timestamp=6)
        assert z.iloc[1].isna().all()
        assert z.iloc[0].notna().all()

    def test_method_none_returns_raw_signal(self, multi_symbol_close_wide):
        ret = compute_bar_returns(multi_symbol_close_wide).dropna()
        result = cross_sectional_rank_or_zscore(ret, method="none", min_assets_per_timestamp=2)
        assert np.allclose(result.fillna(0).values, ret.fillna(0).values, atol=1e-12)

    def test_raises_on_invalid_method(self, multi_symbol_close_wide):
        ret = compute_bar_returns(multi_symbol_close_wide).dropna()
        with pytest.raises(ValueError, match="method must be one of"):
            cross_sectional_rank_or_zscore(ret, method="invalid")


# ---------------------------------------------------------------------------
# apply_rebalance_decimation
# ---------------------------------------------------------------------------

class TestApplyRebalanceDecimation:
    def test_n1_is_noop(self, multi_symbol_close_wide):
        ret = compute_bar_returns(multi_symbol_close_wide).dropna()
        result = apply_rebalance_decimation(ret, every_n_bars=1)
        pd.testing.assert_frame_equal(result, ret)

    def test_n3_samples_every_third_row(self, multi_symbol_close_wide):
        ret = compute_bar_returns(multi_symbol_close_wide).dropna()
        result = apply_rebalance_decimation(ret, every_n_bars=3)
        # Rows 0 and 3 and 6 should be unchanged (sampled rows)
        pd.testing.assert_series_equal(result.iloc[0], ret.iloc[0])
        pd.testing.assert_series_equal(result.iloc[3], ret.iloc[3])

    def test_n3_holds_between_samples(self, multi_symbol_close_wide):
        """Intermediate rows (1, 2) should carry the values from row 0 forward."""
        ret = compute_bar_returns(multi_symbol_close_wide).dropna()
        result = apply_rebalance_decimation(ret, every_n_bars=3)
        # Values must match; index labels differ (row 1 keeps its own timestamp)
        pd.testing.assert_series_equal(result.iloc[1], ret.iloc[0], check_names=False)
        pd.testing.assert_series_equal(result.iloc[2], ret.iloc[0], check_names=False)

    def test_empty_dataframe_returns_empty(self):
        empty = pd.DataFrame(dtype=float)
        result = apply_rebalance_decimation(empty, every_n_bars=3)
        assert result.empty


# ---------------------------------------------------------------------------
# compute_forward_returns
# ---------------------------------------------------------------------------

class TestComputeForwardReturns:
    def test_h1_log_forward_return(self, simple_close_wide):
        """H=1 forward log return at bar t = log(close[t+1] / close[t])."""
        fwd = compute_forward_returns(simple_close_wide, holding_period_bars=1, log_returns=True)
        bar_ret = compute_bar_returns(simple_close_wide, log_returns=True)
        # fwd[t] should equal bar_ret[t+1], so fwd shifted forward by 1 = bar_ret
        expected = bar_ret.shift(-1)
        assert np.allclose(fwd.fillna(0).values, expected.fillna(0).values, atol=1e-12)

    def test_last_h_rows_are_nan(self, simple_close_wide):
        """The last H rows cannot be priced forward and must be NaN."""
        fwd = compute_forward_returns(simple_close_wide, holding_period_bars=2, log_returns=True)
        assert fwd.iloc[-2:].isna().all().all()

    def test_raises_on_non_positive_horizon(self, simple_close_wide):
        with pytest.raises(ValueError):
            compute_forward_returns(simple_close_wide, holding_period_bars=0)

    def test_log_and_simple_consistent(self, multi_symbol_close_wide):
        """exp(log_fwd) - 1 should equal simple_fwd."""
        log_fwd = compute_forward_returns(multi_symbol_close_wide, 2, log_returns=True)
        simple_fwd = compute_forward_returns(multi_symbol_close_wide, 2, log_returns=False)
        mask = (log_fwd.notna() & simple_fwd.notna()).values
        assert np.allclose(
            np.exp(log_fwd.values[mask]) - 1.0, simple_fwd.values[mask], atol=1e-12
        )


# ---------------------------------------------------------------------------
# build_momentum_signal (integration smoke test)
# ---------------------------------------------------------------------------

class TestBuildMomentumSignal:
    def test_returns_expected_keys(self, long_panel_df):
        result = build_momentum_signal(
            long_panel_df,
            signal_timeframe="h",
            momentum_lookback_bars=2,
        )
        assert set(result.keys()) == {
            "close_wide", "return_wide", "raw_signal", "signal", "signal_fresh",
            "universe_mask",
        }

    def test_signal_shape_matches_close_wide(self, long_panel_df):
        result = build_momentum_signal(
            long_panel_df, signal_timeframe="h", momentum_lookback_bars=2
        )
        assert result["signal"].shape == result["close_wide"].shape

    def test_universe_mask_is_none_when_no_universe_passed(self, long_panel_df):
        result = build_momentum_signal(
            long_panel_df, signal_timeframe="h", momentum_lookback_bars=2
        )
        assert result["universe_mask"] is None

    def test_universe_mask_filters_signal(self, long_panel_df, universe_df):
        result = build_momentum_signal(
            long_panel_df,
            signal_timeframe="h",
            momentum_lookback_bars=2,
            universe_df=universe_df,
        )
        assert result["universe_mask"] is not None

    def test_zscore_transform_gives_zero_mean_rows(self, long_panel_df):
        result = build_momentum_signal(
            long_panel_df,
            signal_timeframe="h",
            momentum_lookback_bars=1,
            cross_sectional_transform="zscore",
            min_assets_per_timestamp=2,
        )
        row_means = result["signal"].mean(axis=1).dropna()
        assert np.allclose(row_means.values, 0.0, atol=1e-10)

    def test_momentum_skip_bars_wires_into_raw_signal(self, long_panel_df):
        result_skip = build_momentum_signal(
            long_panel_df,
            signal_timeframe="h",
            momentum_lookback_bars=1,
            momentum_skip_bars=1,
            cross_sectional_transform="none",
            min_assets_per_timestamp=1,
        )
        expected = result_skip["return_wide"].shift(1)
        assert np.allclose(
            result_skip["raw_signal"].fillna(0).values,
            expected.fillna(0).values,
            atol=1e-12,
        )


class TestBuildFeaturePanels:
    def test_skip_bars_propagates_to_core_horizons(self, multi_symbol_close_wide):
        out = build_feature_panels(
            close_wide=multi_symbol_close_wide,
            horizons=[2],
            benchmark_symbol="AAA",
            log_returns=True,
            skip_bars=1,
        )
        expected = np.log(multi_symbol_close_wide.shift(1) / multi_symbol_close_wide.shift(3))
        assert np.allclose(
            out["core_returns"][2].fillna(0).values,
            expected.fillna(0).values,
            atol=1e-12,
        )

    def test_no_residual_family_without_beta_args(self, multi_symbol_close_wide):
        out = build_feature_panels(
            close_wide=multi_symbol_close_wide,
            horizons=[2],
            benchmark_symbol="AAA",
        )
        assert "residual_features" not in out
        assert set(out.keys()) == {
            "core_returns", "relative_features", "vol_adjusted_features",
        }

    def test_residual_family_added_with_beta_args(self, multi_symbol_close_wide):
        ret_wide = compute_bar_returns(multi_symbol_close_wide, log_returns=True)
        market = build_market_index_returns(ret_wide)
        betas = pd.DataFrame(
            1.0, index=ret_wide.index, columns=ret_wide.columns
        )
        out = build_feature_panels(
            close_wide=multi_symbol_close_wide,
            horizons=[2],
            benchmark_symbol="AAA",
            beta_panel=betas,
            market_returns=market,
        )
        assert set(out["residual_features"][2].keys()) == {
            "residual_momentum", "residual_momentum_scaled",
        }
        # Legacy families are unchanged by the residual addition.
        legacy = build_feature_panels(
            close_wide=multi_symbol_close_wide, horizons=[2], benchmark_symbol="AAA"
        )
        pd.testing.assert_frame_equal(
            out["core_returns"][2], legacy["core_returns"][2]
        )

    def test_raises_when_only_one_beta_arg_passed(self, multi_symbol_close_wide):
        ret_wide = compute_bar_returns(multi_symbol_close_wide, log_returns=True)
        with pytest.raises(ValueError, match="provided together"):
            build_feature_panels(
                close_wide=multi_symbol_close_wide,
                horizons=[2],
                benchmark_symbol="AAA",
                market_returns=build_market_index_returns(ret_wide),
            )


# ---------------------------------------------------------------------------
# build_market_index_returns
# ---------------------------------------------------------------------------

class TestBuildMarketIndexReturns:
    def test_equal_weight_on_identical_returns(self, simple_close_wide):
        ret_wide = compute_bar_returns(simple_close_wide, log_returns=True)
        market = build_market_index_returns(ret_wide)
        # All symbols double every bar, so the index return is exactly ln(2).
        assert np.allclose(market.iloc[1:].values, np.log(2.0), atol=1e-12)
        assert np.isnan(market.iloc[0])

    def test_benchmark_mode_returns_that_column(self, multi_symbol_close_wide):
        ret_wide = compute_bar_returns(multi_symbol_close_wide, log_returns=True)
        market = build_market_index_returns(
            ret_wide, mode="benchmark", benchmark_symbol="BBB"
        )
        assert np.allclose(
            market.fillna(0).values, ret_wide["BBB"].fillna(0).values, atol=1e-15
        )

    def test_universe_mask_restricts_mean(self, multi_symbol_close_wide):
        ret_wide = compute_bar_returns(multi_symbol_close_wide, log_returns=True)
        mask = pd.DataFrame(False, index=ret_wide.index, columns=ret_wide.columns)
        mask[["AAA", "BBB"]] = True
        market = build_market_index_returns(ret_wide, universe_mask=mask)
        expected = ret_wide[["AAA", "BBB"]].mean(axis=1)
        assert np.allclose(
            market.fillna(0).values, expected.fillna(0).values, atol=1e-15
        )

    def test_min_assets_nans_sparse_rows(self, multi_symbol_close_wide):
        ret_wide = compute_bar_returns(multi_symbol_close_wide, log_returns=True)
        sparse = ret_wide.copy()
        sparse.iloc[3, 1:] = np.nan  # leave one valid symbol at bar 3
        market = build_market_index_returns(sparse, min_assets=2)
        assert np.isnan(market.iloc[3])
        assert not np.isnan(market.iloc[4])

    def test_raises_on_missing_benchmark(self, multi_symbol_close_wide):
        ret_wide = compute_bar_returns(multi_symbol_close_wide, log_returns=True)
        with pytest.raises(KeyError, match="XBT/USD"):
            build_market_index_returns(ret_wide, mode="benchmark")

    def test_raises_on_invalid_min_assets(self, multi_symbol_close_wide):
        ret_wide = compute_bar_returns(multi_symbol_close_wide, log_returns=True)
        with pytest.raises(ValueError, match="min_assets"):
            build_market_index_returns(ret_wide, min_assets=0)

    def test_all_nan_row_is_nan(self, multi_symbol_close_wide):
        ret_wide = compute_bar_returns(multi_symbol_close_wide, log_returns=True)
        ret_wide.iloc[4, :] = np.nan
        market = build_market_index_returns(ret_wide, min_assets=1)
        assert np.isnan(market.iloc[4])

    def test_raises_on_invalid_mode(self, multi_symbol_close_wide):
        ret_wide = compute_bar_returns(multi_symbol_close_wide, log_returns=True)
        with pytest.raises(ValueError, match="mode must be"):
            build_market_index_returns(ret_wide, mode="cap_weight")


# ---------------------------------------------------------------------------
# estimate_rolling_betas
# ---------------------------------------------------------------------------

def _beta_test_panel():
    """8-bar market series with three assets of known beta.

    AAA equals the market (beta 1), BBB is 2x the market (beta 2), CCC is the
    market plus a constant (beta 1 — adding a constant changes neither the
    covariance nor the variance).
    """
    idx = pd.date_range("2023-01-01", periods=8, freq="1D", tz="UTC")
    m = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.02], index=idx)
    ret_wide = pd.DataFrame(
        {"AAA": m, "BBB": 2.0 * m, "CCC": m + 0.005}, index=idx
    )
    return ret_wide, m


class TestEstimateRollingBetas:
    def test_known_betas_recovered_exactly(self):
        ret_wide, m = _beta_test_panel()
        betas = estimate_rolling_betas(ret_wide, m, window_bars=4)
        valid = betas.iloc[3:]
        assert np.allclose(valid["AAA"].values, 1.0, atol=1e-12)
        assert np.allclose(valid["BBB"].values, 2.0, atol=1e-12)
        assert np.allclose(valid["CCC"].values, 1.0, atol=1e-12)

    def test_first_window_minus_one_rows_nan(self):
        ret_wide, m = _beta_test_panel()
        betas = estimate_rolling_betas(ret_wide, m, window_bars=4)
        assert betas.iloc[:3].isna().all().all()

    def test_shrinkage_pulls_toward_target(self):
        ret_wide, m = _beta_test_panel()
        betas = estimate_rolling_betas(
            ret_wide, m, window_bars=4, shrinkage=0.5, shrink_target=1.0
        )
        # BBB raw beta 2.0 -> 0.5*2.0 + 0.5*1.0 = 1.5
        assert np.allclose(betas["BBB"].iloc[3:].values, 1.5, atol=1e-12)
        assert np.allclose(betas["AAA"].iloc[3:].values, 1.0, atol=1e-12)

    def test_zero_variance_market_gives_nan_not_inf(self):
        idx = pd.date_range("2023-01-01", periods=6, freq="1D", tz="UTC")
        m = pd.Series(0.01, index=idx)  # constant market
        ret_wide = pd.DataFrame({"AAA": np.linspace(0.0, 0.05, 6)}, index=idx)
        betas = estimate_rolling_betas(ret_wide, m, window_bars=3)
        assert betas["AAA"].isna().all()
        assert not np.isinf(betas["AAA"].fillna(0.0)).any()

    def test_nan_return_blocks_strict_window(self):
        ret_wide, m = _beta_test_panel()
        holed = ret_wide.copy()
        holed.loc[holed.index[4], "AAA"] = np.nan  # traded-mask hole
        strict = estimate_rolling_betas(holed, m, window_bars=4)
        # Windows covering the hole (bars 4..7) lack a full set of pairs.
        assert strict["AAA"].iloc[4:8].isna().all()
        relaxed = estimate_rolling_betas(holed, m, window_bars=4, min_periods=3)
        assert np.allclose(relaxed["AAA"].iloc[4:8].dropna().values, 1.0, atol=1e-12)

    def test_raises_on_bad_params(self):
        ret_wide, m = _beta_test_panel()
        with pytest.raises(ValueError, match="window_bars"):
            estimate_rolling_betas(ret_wide, m, window_bars=1)
        with pytest.raises(ValueError, match="shrinkage"):
            estimate_rolling_betas(ret_wide, m, window_bars=4, shrinkage=1.5)
        with pytest.raises(ValueError, match="min_periods"):
            estimate_rolling_betas(ret_wide, m, window_bars=4, min_periods=1)

    def test_min_periods_relaxation_estimates_earlier(self):
        # Relaxing min_periods below the window yields betas before the window
        # is full — the behaviour the research configs rely on (e.g. 540/720).
        ret_wide, m = _beta_test_panel()
        strict = estimate_rolling_betas(ret_wide, m, window_bars=4)
        relaxed = estimate_rolling_betas(ret_wide, m, window_bars=4, min_periods=2)
        # Strict: first valid at row 3 (0-indexed). Relaxed: first valid at row 1.
        assert strict["AAA"].iloc[:3].isna().all()
        assert relaxed["AAA"].iloc[0:1].isna().all()
        assert not np.isnan(relaxed["AAA"].iloc[1])
        # Where both are defined the estimates still recover the known betas.
        assert np.allclose(relaxed["BBB"].iloc[3:].values, 2.0, atol=1e-12)

    def test_full_shrinkage_collapses_to_target(self):
        ret_wide, m = _beta_test_panel()
        betas = estimate_rolling_betas(
            ret_wide, m, window_bars=4, shrinkage=1.0, shrink_target=1.3
        )
        valid = betas.iloc[3:]
        assert np.allclose(valid.values, 1.3, atol=1e-12)


# ---------------------------------------------------------------------------
# build_residual_return_panel
# ---------------------------------------------------------------------------

class TestBuildResidualReturnPanel:
    def test_residual_is_alpha_for_known_beta(self):
        ret_wide, m = _beta_test_panel()
        alpha = pd.DataFrame(
            {"AAA": 0.0, "BBB": 0.0, "CCC": 0.005},
            index=ret_wide.index,
        )
        betas = pd.DataFrame(
            {"AAA": 1.0, "BBB": 2.0, "CCC": 1.0}, index=ret_wide.index
        )
        resid = build_residual_return_panel(ret_wide, betas, m)
        assert np.allclose(resid.values, alpha.values, atol=1e-15)

    def test_nan_beta_propagates(self):
        ret_wide, m = _beta_test_panel()
        betas = pd.DataFrame(1.0, index=ret_wide.index, columns=ret_wide.columns)
        betas.loc[betas.index[2], "AAA"] = np.nan
        resid = build_residual_return_panel(ret_wide, betas, m)
        assert np.isnan(resid.loc[resid.index[2], "AAA"])
        assert not np.isnan(resid.loc[resid.index[2], "BBB"])


# ---------------------------------------------------------------------------
# build_residual_momentum_features
# ---------------------------------------------------------------------------

class TestBuildResidualMomentumFeatures:
    def test_constant_residual_sums_and_scaled_nan(self):
        idx = pd.date_range("2023-01-01", periods=6, freq="1D", tz="UTC")
        resid = pd.DataFrame({"AAA": 0.01}, index=idx)
        feats = build_residual_momentum_features(resid, horizons=[3])
        # h-bar sum of a constant c is exactly h*c ...
        assert np.allclose(
            feats[3]["residual_momentum"].iloc[2:].values, 0.03, atol=1e-15
        )
        # ... and the rolling std is 0, so the scaled variant is NaN, not inf.
        assert feats[3]["residual_momentum_scaled"].isna().all().all()

    def test_scaled_matches_hand_computation(self):
        idx = pd.date_range("2023-01-01", periods=5, freq="1D", tz="UTC")
        vals = np.array([0.01, -0.02, 0.03, 0.00, 0.02])
        resid = pd.DataFrame({"AAA": vals}, index=idx)
        feats = build_residual_momentum_features(resid, horizons=[3])
        window = vals[2:5]
        expected = window.sum() / window.std()  # numpy std is ddof=0
        assert np.isclose(
            feats[3]["residual_momentum_scaled"].iloc[4], expected, atol=1e-12
        )

    def test_skip_bars_excludes_most_recent_residual(self):
        idx = pd.date_range("2023-01-01", periods=6, freq="1D", tz="UTC")
        vals = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 100.0])
        resid = pd.DataFrame({"AAA": vals}, index=idx)
        feats = build_residual_momentum_features(resid, horizons=[3], skip_bars=1)
        # At the last bar, the skip excludes the huge 100.0 print.
        assert np.isclose(
            feats[3]["residual_momentum"].iloc[5], 0.03 + 0.04 + 0.05, atol=1e-12
        )

    def test_raises_on_bad_params(self):
        idx = pd.date_range("2023-01-01", periods=4, freq="1D", tz="UTC")
        resid = pd.DataFrame({"AAA": 0.01}, index=idx)
        with pytest.raises(ValueError, match="horizons"):
            build_residual_momentum_features(resid, horizons=[0])
        with pytest.raises(ValueError, match="skip_bars"):
            build_residual_momentum_features(resid, horizons=[2], skip_bars=-1)


# ---------------------------------------------------------------------------
# select_momentum_feature_panel — residual family
# ---------------------------------------------------------------------------

class TestSelectMomentumFeaturePanelResidual:
    def _pack_with_residuals(self, close_wide):
        ret_wide = compute_bar_returns(close_wide, log_returns=True)
        market = build_market_index_returns(ret_wide)
        betas = pd.DataFrame(1.0, index=ret_wide.index, columns=ret_wide.columns)
        return build_feature_panels(
            close_wide=close_wide,
            horizons=[2],
            benchmark_symbol="AAA",
            beta_panel=betas,
            market_returns=market,
        )

    def test_returns_residual_panel_and_label(self, multi_symbol_close_wide):
        pack = self._pack_with_residuals(multi_symbol_close_wide)
        panel, label = select_momentum_feature_panel(
            feature_pack=pack,
            baseline_signal=None,
            available_horizons=[2],
            feature_family="residual",
            feature_horizon=2,
        )
        assert label == "residual_momentum_scaled_2bar"
        pd.testing.assert_frame_equal(
            panel, pack["residual_features"][2]["residual_momentum_scaled"]
        )

    def test_raises_when_pack_lacks_residuals(self, multi_symbol_close_wide):
        pack = build_feature_panels(
            close_wide=multi_symbol_close_wide, horizons=[2], benchmark_symbol="AAA"
        )
        with pytest.raises(ValueError, match="residual_features"):
            select_momentum_feature_panel(
                feature_pack=pack,
                baseline_signal=None,
                available_horizons=[2],
                feature_family="residual",
                feature_horizon=2,
            )

    def test_raises_on_unknown_residual_feature(self, multi_symbol_close_wide):
        pack = self._pack_with_residuals(multi_symbol_close_wide)
        with pytest.raises(ValueError, match="residual_feature must be"):
            select_momentum_feature_panel(
                feature_pack=pack,
                baseline_signal=None,
                available_horizons=[2],
                feature_family="residual",
                feature_horizon=2,
                residual_feature="not_a_feature",
            )


# ---------------------------------------------------------------------------
# rolling_mean_std
# ---------------------------------------------------------------------------

class TestRollingMeanStd:
    def _panel(self, values):
        idx = pd.date_range("2022-05-01", periods=len(values), freq="h", tz="UTC")
        return pd.DataFrame({"A": values}, index=idx)

    def test_matches_pandas_on_well_conditioned_data(self):
        rng = np.random.default_rng(0)
        panel = self._panel(100.0 + rng.normal(size=300).cumsum())
        mean, std = rolling_mean_std(panel, 24)
        rolling = panel.rolling(24, min_periods=24)
        pd.testing.assert_frame_equal(mean, rolling.mean(), rtol=1e-10)
        pd.testing.assert_frame_equal(std, rolling.std(ddof=0), rtol=1e-8)

    def test_exact_after_price_level_collapse(self):
        """Moments stay accurate when the level falls by six orders of magnitude."""
        rng = np.random.default_rng(1)
        high = 80.0 * np.exp(rng.normal(0, 0.01, 200).cumsum())
        low = 1e-4 * np.exp(rng.normal(0, 0.01, 200).cumsum())
        panel = self._panel(np.concatenate([high, low]))
        mean, std = rolling_mean_std(panel, 24)

        tail = panel["A"].to_numpy()[-24:]
        assert mean["A"].iloc[-1] == pytest.approx(tail.mean(), rel=1e-12)
        assert std["A"].iloc[-1] == pytest.approx(tail.std(), rel=1e-12)
        assert (std["A"].iloc[230:] > 0).all()

    def test_constant_window_has_exactly_zero_std(self):
        panel = self._panel([0.1, 0.2, 0.3, 0.3, 0.3, 0.3])
        mean, std = rolling_mean_std(panel, 3)
        assert std["A"].iloc[-1] == 0.0
        assert mean["A"].iloc[-1] == 0.3
        assert std["A"].iloc[2] > 0

    def test_nan_in_window_and_warmup_yield_nan(self):
        panel = self._panel([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0])
        mean, std = rolling_mean_std(panel, 3)
        assert mean["A"].isna().tolist() == [True, True, True, True, True, False, False]
        assert std["A"].isna().tolist() == [True, True, True, True, True, False, False]

    def test_window_longer_than_panel_is_all_nan(self):
        mean, std = rolling_mean_std(self._panel([1.0, 2.0]), 5)
        assert mean.isna().all().all() and std.isna().all().all()

    def test_ddof(self):
        panel = self._panel([1.0, 2.0, 4.0])
        _, std = rolling_mean_std(panel, 3, ddof=1)
        assert std["A"].iloc[-1] == pytest.approx(np.std([1.0, 2.0, 4.0], ddof=1))

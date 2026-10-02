"""Unit tests for feature lookup / grid-building helpers in the signal modules."""

import numpy as np
import pandas as pd
import pytest

from src.signals.cs_mean_reversion import (
    build_return_reversal_features,
    build_reversal_family_panel,
    build_vol_adjusted_move,
    get_mean_reversion_feature,
    mean_reversion_feature_lookback,
)
from src.signals.cs_momentum import (
    build_residual_momentum_features,
    build_residual_momentum_signal,
    build_residual_return_panel,
    build_selected_momentum_signal,
)

SETTINGS = {
    "price_zscore_window_bars": 24,
    "bollinger_window_bars": 20,
    "rsi_window_bars": 14,
    "range_position_window_bars": 12,
    "reversal_lookback_bars": 2,
}


def _close(n=40, m=6, seed=0):
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, size=(n, m))
    return pd.DataFrame(100 * np.exp(rets.cumsum(axis=0)), index=idx,
                        columns=[f"S{i}" for i in range(m)])


class TestMeanReversionFeatureLookback:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("reversal_h4", 4),
            ("vol_blowoff_h6", 6),
            ("price_zscore", 24),
            ("bollinger_touch", 20),
            ("rsi_proxy", 14),
            ("range_position", 12),
            ("xsec_rank_price_z", 24),
            ("distance_from_vwap", 2),
        ],
    )
    def test_lookbacks(self, name, expected):
        assert mean_reversion_feature_lookback(name, SETTINGS) == expected


class TestGetMeanReversionFeature:
    def _packs(self):
        return {
            "reversal": {1: {"reversal": "r1", "xsec_reversal": "x1"}},
            "vol_adj": {1: {"vol_adj_reversal": "va1", "vol_adj_xsec_reversal": "vax1"}},
            "vol_exh": {1: {"vol_exhaustion": "ve1", "vol_blowoff": "vb1"}},
            "price_z": {"price_zscore": "pz", "bollinger_touch": "bt"},
            "extreme": {"signed_z": "sz", "extreme_flag": "ef"},
            "vwap": {"distance_from_vwap": "dv", "distance_from_ma": "dm"},
            "rsi": {"rsi_proxy": "rsi"},
            "range": {"range_position": "rp"},
            "xsec_z": {"xsec_rank_price_z": "xz"},
        }

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("reversal_h1", "r1"), ("xsec_reversal_h1", "x1"),
            ("vol_adj_reversal_h1", "va1"), ("vol_adj_xsec_reversal_h1", "vax1"),
            ("vol_exhaustion_h1", "ve1"), ("vol_blowoff_h1", "vb1"),
            ("price_zscore", "pz"), ("bollinger_touch", "bt"), ("extreme_signed_z", "sz"),
            ("extreme_flag", "ef"), ("distance_from_vwap", "dv"), ("distance_from_ma", "dm"),
            ("rsi_proxy", "rsi"), ("range_position", "rp"), ("xsec_rank_price_z", "xz"),
        ],
    )
    def test_every_name_resolves(self, name, expected):
        assert get_mean_reversion_feature(name, self._packs()) == expected

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_mean_reversion_feature("momentum_core_h1", self._packs())


class TestBuildReversalFamilyPanel:
    def test_reversal_matches_builder(self):
        close = _close()
        expected = build_return_reversal_features(close, [3])[3]["reversal"]
        pd.testing.assert_frame_equal(build_reversal_family_panel(close, "reversal", 3), expected)

    def test_vol_adjusted_xsec_matches_builder(self):
        close = _close()
        expected = build_vol_adjusted_move(close, [2], vol_window_bars=5)[2]["vol_adj_xsec_reversal"]
        out = build_reversal_family_panel(close, "vol_adj_xsec_reversal", 2, vol_window_bars=5)
        pd.testing.assert_frame_equal(out, expected)

    def test_universe_mask_blanks_outside_cells(self):
        close = _close()
        mask = pd.DataFrame(True, index=close.index, columns=close.columns)
        mask["S0"] = False
        out = build_reversal_family_panel(close, "reversal", 1, universe_mask=mask)
        assert out["S0"].isna().all() and out["S1"].notna().any()


class TestBuildResidualMomentumSignal:
    def test_matches_manual_pipeline(self):
        rets = np.log(_close(seed=2)).diff()
        market = rets.mean(axis=1)
        betas = pd.DataFrame(1.0, index=rets.index, columns=rets.columns)
        resid = build_residual_return_panel(rets, betas, market)
        feats = build_residual_momentum_features(resid, horizons=[5], skip_bars=1)
        expected = build_selected_momentum_signal(
            raw_panel=feats[5]["residual_momentum_scaled"],
            cross_sectional_transform="rank",
            min_assets_per_timestamp=4,
            apply_rebalance_decimation_flag=False,
        )
        out = build_residual_momentum_signal(
            rets, betas, market, horizon=5, skip_bars=1,
            cross_sectional_transform="rank", min_assets_per_timestamp=4,
        )
        pd.testing.assert_frame_equal(out, expected)

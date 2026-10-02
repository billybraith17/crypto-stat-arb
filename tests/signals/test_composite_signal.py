"""Unit tests for build_composite_signal (equal-weight z-scored composite)."""

import numpy as np
import pandas as pd
import pytest

from src.signals.cs_mean_reversion import build_composite_signal
from src.signals.cs_momentum import cross_sectional_rank_or_zscore


def _utc_index(n, freq="1h", start="2023-01-01"):
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC")


@pytest.fixture
def base_panel():
    idx = _utc_index(6)
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        rng.normal(0, 1, size=(6, 6)),
        index=idx,
        columns=list("ABCDEF"),
    )


class TestBuildCompositeSignal:
    def test_identical_panels_equal_single_zscore(self, base_panel):
        z = cross_sectional_rank_or_zscore(base_panel, method="zscore", min_assets_per_timestamp=6)
        composite = build_composite_signal(
            [base_panel, base_panel.copy()], min_assets_per_timestamp=6
        )
        pd.testing.assert_frame_equal(composite, z)

    def test_opposite_panels_cancel(self, base_panel):
        composite = build_composite_signal(
            [base_panel, -base_panel], min_assets_per_timestamp=6
        )
        np.testing.assert_allclose(composite.to_numpy(), 0.0, atol=1e-12)

    def test_dict_and_list_inputs_equivalent(self, base_panel):
        from_list = build_composite_signal([base_panel, 2 * base_panel], min_assets_per_timestamp=6)
        from_dict = build_composite_signal(
            {"a": base_panel, "b": 2 * base_panel}, min_assets_per_timestamp=6
        )
        pd.testing.assert_frame_equal(from_list, from_dict)

    def test_min_features_gate(self, base_panel):
        # Second panel is NaN for symbol A -> A's composite has only 1 valid
        # feature -> NaN under min_features=2, present under min_features=1.
        other = base_panel * 3.0
        other["A"] = np.nan
        strict = build_composite_signal(
            [base_panel, other], min_assets_per_timestamp=5, min_features=2
        )
        loose = build_composite_signal(
            [base_panel, other], min_assets_per_timestamp=5, min_features=1
        )
        assert strict["A"].isna().all()
        assert loose["A"].notna().all()
        # Fully covered symbols are unaffected by the gate.
        pd.testing.assert_series_equal(strict["B"], loose["B"])

    def test_equal_weighting_of_features(self, base_panel):
        # Composite of z(p) and z(3p): z-scoring makes them identical, so the
        # average equals either one — scale invariance per feature.
        composite = build_composite_signal([base_panel, 3.0 * base_panel], min_assets_per_timestamp=6)
        z = cross_sectional_rank_or_zscore(base_panel, method="zscore", min_assets_per_timestamp=6)
        pd.testing.assert_frame_equal(composite, z)

    def test_validation(self, base_panel):
        with pytest.raises(ValueError, match="empty"):
            build_composite_signal([])
        with pytest.raises(ValueError, match="min_features"):
            build_composite_signal([base_panel], min_features=0)

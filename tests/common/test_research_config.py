"""Unit tests for src/common/research_config.py.

Focuses on config loading, type coercion, and validation guards. Tests load
the real momentum_signal.yaml so they also serve as a smoke test that the
checked-in config is internally consistent.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.common.research_config import load_research_settings


REAL_CONFIG = "configs/research/momentum_signal.yaml"


# ---------------------------------------------------------------------------
# Smoke test on the real config file
# ---------------------------------------------------------------------------

class TestLoadRealConfig:
    def test_loads_without_error(self):
        settings = load_research_settings(REAL_CONFIG)
        assert isinstance(settings, dict)

    def test_required_keys_present(self):
        settings = load_research_settings(REAL_CONFIG)
        required = {
            "signal_timeframe",
            "momentum_lookback_bars",
            "holding_period_bars",
            "rebalance_every_n_bars",
            "execution_delay_bars",
            "log_returns",
            "vol_window_bars",
            "benchmark_symbol",
            "residual_space",
            "cross_sectional_transform",
            "min_assets_per_timestamp",
            "top_quantile",
            "bottom_quantile",
            "fee_bps",
            "half_spread_bps",
            "use_monthly_universe",
            "feature_horizons_bars",
            "momentum_lookback_grid_bars",
            "holding_period_grid_bars",
            "cost_stress_multipliers",
            "data_start_date",
            "data_end_date",
            "train_split_date",
        }
        missing = required - set(settings.keys())
        assert not missing, f"Missing keys in loaded settings: {missing}"

    def test_types_are_correct(self):
        s = load_research_settings(REAL_CONFIG)
        assert isinstance(s["signal_timeframe"], str)
        assert isinstance(s["momentum_lookback_bars"], int)
        assert isinstance(s["holding_period_bars"], int)
        assert isinstance(s["log_returns"], bool)
        assert isinstance(s["top_quantile"], float)
        assert isinstance(s["fee_bps"], float)
        assert isinstance(s["feature_horizons_bars"], list)
        assert all(isinstance(h, int) for h in s["feature_horizons_bars"])
        assert isinstance(s["cost_stress_multipliers"], list)
        assert all(isinstance(x, float) for x in s["cost_stress_multipliers"])

    def test_residual_space_is_valid(self):
        s = load_research_settings(REAL_CONFIG)
        assert s["residual_space"] in ("log", "simple")

    def test_quantiles_in_valid_range(self):
        s = load_research_settings(REAL_CONFIG)
        assert 0.0 < s["top_quantile"] < 1.0
        assert 0.0 < s["bottom_quantile"] < 1.0

    def test_date_ordering(self):
        """data_start_date must precede data_end_date."""
        s = load_research_settings(REAL_CONFIG)
        assert s["data_start_date"] < s["data_end_date"]

    def test_train_split_within_data_range(self):
        s = load_research_settings(REAL_CONFIG)
        assert s["data_start_date"] <= s["train_split_date"] <= s["data_end_date"]

    def test_grids_are_non_empty_lists(self):
        s = load_research_settings(REAL_CONFIG)
        assert len(s["momentum_lookback_grid_bars"]) > 0
        assert len(s["holding_period_grid_bars"]) > 0
        assert len(s["feature_horizons_bars"]) > 0

    def test_all_grid_bars_positive(self):
        s = load_research_settings(REAL_CONFIG)
        for key in ("momentum_lookback_grid_bars", "holding_period_grid_bars",
                    "feature_horizons_bars"):
            assert all(v > 0 for v in s[key]), f"Non-positive value in {key}"

    def test_min_assets_positive(self):
        s = load_research_settings(REAL_CONFIG)
        assert s["min_assets_per_timestamp"] >= 2


# ---------------------------------------------------------------------------
# Validation guards via synthetic config files
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, data: dict) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(data))
    return str(p)


class TestValidationGuards:
    def test_raises_on_invalid_residual_space(self, tmp_path):
        data = {
            "reference_base_config": "configs/base.yaml",
            "residual_space": "geometric",
        }
        with pytest.raises(ValueError, match="residual_space"):
            load_research_settings(_write_yaml(tmp_path, data))

    def test_default_signal_timeframe_when_missing(self, tmp_path):
        data = {"reference_base_config": "configs/base.yaml"}
        settings = load_research_settings(_write_yaml(tmp_path, data))
        assert settings["signal_timeframe"] == "4h"

    def test_default_log_returns_is_true(self, tmp_path):
        data = {"reference_base_config": "configs/base.yaml"}
        settings = load_research_settings(_write_yaml(tmp_path, data))
        assert settings["log_returns"] is True

    def test_max_assets_none_when_unset(self, tmp_path):
        data = {"reference_base_config": "configs/base.yaml"}
        settings = load_research_settings(_write_yaml(tmp_path, data))
        assert settings["max_assets"] is None

    def test_max_assets_cast_to_int_when_set(self, tmp_path):
        data = {"reference_base_config": "configs/base.yaml", "max_assets": 15}
        settings = load_research_settings(_write_yaml(tmp_path, data))
        assert settings["max_assets"] == 15
        assert isinstance(settings["max_assets"], int)

    def test_empty_yaml_uses_all_defaults(self, tmp_path):
        """An empty YAML file with a known base config must not raise."""
        p = tmp_path / "empty.yaml"
        p.write_text("reference_base_config: configs/base.yaml\n")
        settings = load_research_settings(str(p))
        assert "signal_timeframe" in settings

    def test_overrides_base_config_dates(self, tmp_path):
        data = {
            "reference_base_config": "configs/base.yaml",
            "data_start_date": "2021-01-01 00:00:00+00:00",
            "data_end_date": "2023-01-01 00:00:00+00:00",
        }
        settings = load_research_settings(_write_yaml(tmp_path, data))
        import pandas as pd
        assert settings["data_start_date"] == pd.Timestamp("2021-01-01", tz="UTC")
        assert settings["data_end_date"] == pd.Timestamp("2023-01-01", tz="UTC")

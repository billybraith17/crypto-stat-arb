"""Unit tests for selection-bias corrections and walk-forward evaluation.

Covers max_over_trials_pvalue, expected_max_sharpe, probabilistic /
deflated Sharpe, walk_forward_splits / _ic_table / _selection in
src/research/momentum_eval.py.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.research.momentum_eval import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    max_over_trials_pvalue,
    probabilistic_sharpe_ratio,
    walk_forward_ic_table,
    walk_forward_selection,
    walk_forward_splits,
)


def _utc_index(n, freq="1h", start="2023-01-01"):
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC")


class TestMaxOverTrialsPvalue:
    def test_single_trial_matches_normal_tail(self):
        t = 2.0
        assert max_over_trials_pvalue(t, 1) == pytest.approx(1 - norm.cdf(t))

    def test_increases_with_trials(self):
        t = 3.0
        p = [max_over_trials_pvalue(t, n) for n in [1, 5, 25, 100]]
        assert all(a < b for a, b in zip(p, p[1:]))

    def test_moderate_t_is_unremarkable_over_many_trials(self):
        # t=3.15 looks strong alone (p ~ 0.0008) but over 25 trials the
        # probability of seeing at least one such max is ~2%. Over a 240-row
        # grid it is ~18%.
        assert max_over_trials_pvalue(3.15, 1) < 0.001
        assert max_over_trials_pvalue(3.15, 240) > 0.15

    def test_invalid_inputs(self):
        assert np.isnan(max_over_trials_pvalue(np.nan, 5))
        with pytest.raises(ValueError):
            max_over_trials_pvalue(2.0, 0)


class TestExpectedMaxSharpe:
    def test_single_trial_returns_mean(self):
        assert expected_max_sharpe(1, var_sharpe=1.0, mean_sharpe=0.3) == 0.3

    def test_increases_with_trials(self):
        vals = [expected_max_sharpe(n, var_sharpe=0.01) for n in [2, 10, 100, 1000]]
        assert all(a < b for a, b in zip(vals, vals[1:]))

    def test_matches_formula(self):
        n, var = 10, 1.0
        gamma = 0.5772156649015329
        expected = (1 - gamma) * norm.ppf(1 - 1 / n) + gamma * norm.ppf(1 - 1 / (n * np.e))
        assert expected_max_sharpe(n, var) == pytest.approx(expected)

    def test_scales_with_sd(self):
        assert expected_max_sharpe(10, 4.0) == pytest.approx(2 * expected_max_sharpe(10, 1.0))


class TestProbabilisticSharpe:
    def test_equal_to_benchmark_is_half(self):
        assert probabilistic_sharpe_ratio(0.1, 0.1, n_obs=500) == pytest.approx(0.5)

    def test_above_benchmark_converges_to_one_with_obs(self):
        lo = probabilistic_sharpe_ratio(0.1, 0.05, n_obs=50)
        hi = probabilistic_sharpe_ratio(0.1, 0.05, n_obs=5000)
        assert 0.5 < lo < hi < 1.0

    def test_fat_tails_reduce_confidence(self):
        normal = probabilistic_sharpe_ratio(0.1, 0.0, n_obs=500, kurt=3.0)
        fat = probabilistic_sharpe_ratio(0.1, 0.0, n_obs=500, kurt=10.0)
        assert fat < normal

    def test_too_few_obs_is_nan(self):
        assert np.isnan(probabilistic_sharpe_ratio(0.1, 0.0, n_obs=1))


class TestDeflatedSharpe:
    def test_winner_equal_to_expected_max_is_coin_flip(self):
        n_trials, var = 25, 0.004
        sr_star = expected_max_sharpe(n_trials, var)
        out = deflated_sharpe_ratio(sr_star, n_obs=1000, n_trials=n_trials, var_sharpe=var)
        assert out["deflated_sharpe_prob"] == pytest.approx(0.5)
        assert out["expected_max_sharpe"] == pytest.approx(sr_star)

    def test_more_trials_lower_probability(self):
        kwargs = dict(sharpe=0.15, n_obs=1000, var_sharpe=0.004)
        few = deflated_sharpe_ratio(n_trials=5, **kwargs)
        many = deflated_sharpe_ratio(n_trials=500, **kwargs)
        assert many["deflated_sharpe_prob"] < few["deflated_sharpe_prob"]


class TestWalkForwardSplits:
    def test_partitions_without_embargo(self):
        idx = _utc_index(100)
        splits = walk_forward_splits(idx, n_folds=5, embargo_obs=0)
        assert len(splits) == 5
        recombined = splits[0]["test_index"]
        for s in splits[1:]:
            recombined = recombined.append(s["test_index"])
        assert recombined.equals(idx)

    def test_embargo_drops_head_of_later_folds(self):
        idx = _utc_index(100)
        splits = walk_forward_splits(idx, n_folds=5, embargo_obs=3)
        assert len(splits[0]["test_index"]) == 20
        for s in splits[1:]:
            assert len(s["test_index"]) == 17
        # First retained observation of fold 2 is 3 past the raw boundary.
        assert splits[1]["test_index"][0] == idx[23]

    def test_validation(self):
        idx = _utc_index(10)
        with pytest.raises(ValueError, match="n_folds"):
            walk_forward_splits(idx, n_folds=1)
        with pytest.raises(ValueError, match="embargo"):
            walk_forward_splits(idx, n_folds=2, embargo_obs=-1)
        with pytest.raises(ValueError, match="fewer"):
            walk_forward_splits(idx[:3], n_folds=5)


class TestWalkForwardIcTable:
    def test_constant_ic_is_stable_across_folds(self):
        ic = pd.Series(0.05, index=_utc_index(200))
        table = walk_forward_ic_table(ic, n_folds=4, embargo_obs=2)
        assert len(table) == 4
        np.testing.assert_allclose(table["mean_ic"], 0.05)
        assert table["n_obs"].iloc[0] == 50
        assert (table["n_obs"].iloc[1:] == 48).all()

    def test_regime_break_is_visible(self):
        idx = _utc_index(200)
        ic = pd.Series(np.r_[np.full(100, 0.10), np.full(100, -0.02)], index=idx)
        table = walk_forward_ic_table(ic, n_folds=4)
        assert table["mean_ic"].iloc[0] == pytest.approx(0.10)
        assert table["mean_ic"].iloc[-1] == pytest.approx(-0.02)


class TestWalkForwardSelection:
    def test_selects_dominant_config_and_pools_oos(self):
        idx = _utc_index(200)
        rng = np.random.default_rng(3)
        configs = {
            "good": pd.Series(0.08 + rng.normal(0, 0.001, 200), index=idx),
            "bad": pd.Series(0.00 + rng.normal(0, 0.001, 200), index=idx),
        }
        table, pooled = walk_forward_selection(configs, n_folds=4, embargo_obs=2)
        assert (table["selected_config"] == "good").all()
        # Pooled OOS covers folds 2..4 (embargoed): 3 * 48 observations.
        assert pooled["n_obs"] == 3 * 48
        assert pooled["mean_ic"] == pytest.approx(0.08, abs=0.001)

    def test_lucky_early_config_gets_punished_out_of_sample(self):
        # "lucky" wins the first selection window then decays to zero; the
        # honest OOS record shows the low realized IC instead of the lucky one.
        idx = _utc_index(200)
        lucky = pd.Series(np.r_[np.full(50, 0.20), np.full(150, 0.0)], index=idx)
        steady = pd.Series(0.08, index=idx)
        table, pooled = walk_forward_selection(
            {"lucky": lucky, "steady": steady}, n_folds=4, embargo_obs=0
        )
        assert table.loc[2, "selected_config"] == "lucky"
        assert table.loc[2, "oos_mean_ic"] == pytest.approx(0.0)
        # By fold 4 the expanding window mean of "lucky" (50*0.2/150 = 0.067)
        # has fallen below steady's 0.08 -> selection switches.
        assert table.loc[4, "selected_config"] == "steady"

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            walk_forward_selection({})

    def test_nw_lag_by_config_used_for_pooled_stats(self):
        idx = _utc_index(100)
        configs = {"only": pd.Series(0.05, index=idx)}
        _, pooled = walk_forward_selection(
            configs, n_folds=2, embargo_obs=0, nw_lag_by_config={"only": 7}
        )
        assert pooled["nw_lag"] == 7

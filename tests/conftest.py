"""Shared pytest fixtures for the momentum signal test suite.

All fixtures produce small, deterministic DataFrames with known prices so that
expected outputs can be computed by hand and verified exactly (or to machine
epsilon for floating-point operations).
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Date/time helpers
# ---------------------------------------------------------------------------

def _utc_index(n, freq="4h", start="2023-01-01"):
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC")


# ---------------------------------------------------------------------------
# Price panel fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_close_wide():
    """4-bar × 3-symbol panel with clean, round prices.

    Prices double every bar so log returns are all exactly ln(2) ≈ 0.6931.
    """
    idx = _utc_index(4)
    data = {
        "AAA": [100.0, 200.0, 400.0, 800.0],
        "BBB": [50.0, 100.0, 200.0, 400.0],
        "CCC": [10.0, 20.0, 40.0, 80.0],
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def close_wide_with_nan():
    """Panel with NaN in one cell — used to test min_periods guards."""
    idx = _utc_index(5)
    data = {
        "AAA": [100.0, 110.0, 121.0, 133.1, 146.41],
        "BBB": [200.0, 210.0, np.nan, 231.0, 244.0],
        "CCC": [50.0, 55.0, 60.5, 66.55, 73.2],
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def long_panel_df():
    """Minimal long-format OHLCV panel with ts, symbol, close columns."""
    rows = []
    idx = _utc_index(3)
    for ts in idx:
        for sym, base in [("AAA", 100.0), ("BBB", 200.0), ("CCC", 50.0)]:
            rows.append({"ts": ts, "symbol": sym, "close": base})
    return pd.DataFrame(rows)


@pytest.fixture
def multi_symbol_close_wide():
    """10-bar × 6-symbol panel for cross-sectional transform tests.

    Prices are set so that the cross-sectional ordering at each bar is
    deterministic: AAA is always the highest-momentum symbol, FFF the lowest.
    """
    idx = _utc_index(10)
    rng = np.random.default_rng(42)
    # Each column has a distinct deterministic drift; randomness is small.
    drifts = np.array([0.10, 0.06, 0.02, -0.02, -0.06, -0.10])
    log_rets = drifts[None, :] + rng.normal(0, 0.005, size=(10, 6))
    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    syms = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    return pd.DataFrame(prices, index=idx, columns=syms)


@pytest.fixture
def universe_df():
    """Monthly universe with two rebalance dates covering two symbols."""
    return pd.DataFrame(
        {
            "rebalance_date": pd.to_datetime(
                ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"], utc=True
            ),
            "symbol": ["AAA", "BBB", "AAA", "CCC"],
        }
    )

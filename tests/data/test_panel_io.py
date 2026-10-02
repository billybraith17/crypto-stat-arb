"""DB-free validation tests for src/data/panel_io.py.

The accessors are DB-backed, but their input guards (table whitelist, delay
sign) fire before any query is built, so they are unit-testable with a dummy
engine. These guard the SQL string interpolation against injection and
nonsense input.
"""

import pytest

from src.data.panel_io import (
    ALLOWED_OHLCV_TABLES,
    fetch_minute_exec_closes,
    fetch_ohlcv_long,
)


class TestFetchOhlcvLongTableGuard:
    def test_default_table_allowed(self):
        assert "ohlcv" in ALLOWED_OHLCV_TABLES
        assert "ohlcv_1m" in ALLOWED_OHLCV_TABLES

    def test_rejects_unknown_table(self):
        with pytest.raises(ValueError, match="table must be one of"):
            fetch_ohlcv_long(None, "2023-01-01", "2023-01-02", table="ohlcv; DROP TABLE ohlcv")

    def test_rejects_empty_table(self):
        with pytest.raises(ValueError):
            fetch_ohlcv_long(None, "2023-01-01", "2023-01-02", table="")


class TestFetchMinuteExecClosesDelayGuard:
    def test_rejects_negative_delay(self):
        with pytest.raises(ValueError, match="delay_minutes must be >= 0"):
            fetch_minute_exec_closes(None, "2023-01-01", "2023-01-02", delay_minutes=-1)

    def test_zero_delay_is_allowed_past_guard(self):
        # delay=0 is valid; the call proceeds past the guard and only then
        # fails trying to use the dummy engine — i.e. NOT a ValueError.
        with pytest.raises(Exception) as exc:
            fetch_minute_exec_closes(None, "2023-01-01", "2023-01-02", delay_minutes=0)
        assert not isinstance(exc.value, ValueError)

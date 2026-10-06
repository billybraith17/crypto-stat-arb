"""Unit tests for the sparse 1-minute loader (src/data/build_minute_database.py).

Pure-pandas paths only — COPY loading and universe listing need a live
database and are covered by the pipeline's own checks.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.build_database_from_csv import _clean_raw_frame, prepare_dataframe
from src.data.build_minute_database import (
    OHLCV_COLUMNS,
    format_ts_for_copy,
    minute_csv_path,
    prepare_minute_dataframe,
)

START = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")
END = pd.Timestamp("2023-01-02 23:59:59", tz="UTC")


# ---------------------------------------------------------------------------
# minute_csv_path
# ---------------------------------------------------------------------------

class TestMinuteCsvPath:
    def test_maps_symbol_to_filename(self):
        path = minute_csv_path(Path("/raw"), "XBT/USD", 1)
        assert path == Path("/raw/XBTUSD_1.csv")

    def test_timeframe_parameterised(self):
        path = minute_csv_path(Path("/raw"), "ETH/USD", 5)
        assert path.name == "ETHUSD_5.csv"

    def test_numeric_base_symbol(self):
        path = minute_csv_path(Path("/raw"), "0G/USD", 1)
        assert path.name == "0GUSD_1.csv"


# ---------------------------------------------------------------------------
# prepare_minute_dataframe (sparse — no synthetic rows)
# ---------------------------------------------------------------------------

class TestPrepareMinuteDataframe:
    def _prepare(self, raw):
        return prepare_minute_dataframe(
            raw.copy(), "AAA", "USD", "AAA/USD", START, END
        )

    def test_output_is_sparse(self, sparse_minute_raw_df):
        """Row count equals distinct in-range input timestamps — nothing filled."""
        out = self._prepare(sparse_minute_raw_df)
        in_range = sparse_minute_raw_df[
            (pd.to_datetime(sparse_minute_raw_df["timestamp"], unit="s", utc=True) > START)
            & (pd.to_datetime(sparse_minute_raw_df["timestamp"], unit="s", utc=True) <= END)
        ]
        assert len(out) == in_range["timestamp"].nunique()
        # gap between minute 6 and minute 60 must NOT be filled
        deltas = out["ts"].diff().dropna()
        assert (deltas > pd.Timedelta(minutes=1)).any()

    def test_boundary_semantics_exclusive_start_inclusive_end(self, sparse_minute_raw_df):
        out = self._prepare(sparse_minute_raw_df)
        assert (out["ts"] > START).all()
        assert (out["ts"] <= END).all()

    def test_duplicates_removed(self, sparse_minute_raw_df):
        out = self._prepare(sparse_minute_raw_df)
        assert not out["ts"].duplicated().any()

    def test_dollar_volume(self, sparse_minute_raw_df):
        out = self._prepare(sparse_minute_raw_df)
        assert np.allclose(out["dollar_volume"], out["close"] * out["volume"])

    def test_column_order_matches_hourly_schema(self, sparse_minute_raw_df):
        out = self._prepare(sparse_minute_raw_df)
        assert list(out.columns) == OHLCV_COLUMNS

    def test_metadata_columns(self, sparse_minute_raw_df):
        out = self._prepare(sparse_minute_raw_df)
        assert (out["symbol"] == "AAA/USD").all()
        assert (out["base_asset"] == "AAA").all()
        assert (out["quote_asset"] == "USD").all()
        assert out["trades"].dtype.kind in "iu"


# ---------------------------------------------------------------------------
# format_ts_for_copy — COPY literals must be timezone-explicit
# ---------------------------------------------------------------------------

class TestFormatTsForCopy:
    def test_utc_iso_with_explicit_offset(self):
        """Every literal must carry +00: without it Postgres parses in the
        session timezone and silently shifts DST-period rows."""
        ts = pd.Series(pd.to_datetime(
            ["2021-07-01 12:34:00", "2021-01-01 00:00:00"], utc=True
        ))
        out = format_ts_for_copy(ts)
        assert list(out) == ["2021-07-01T12:34:00+00", "2021-01-01T00:00:00+00"]

    def test_non_utc_input_converted(self):
        ts = pd.Series(
            pd.to_datetime(["2021-07-01 13:34:00"]).tz_localize("Europe/London")
        )
        out = format_ts_for_copy(ts)
        assert list(out) == ["2021-07-01T12:34:00+00"]


# ---------------------------------------------------------------------------
# _clean_raw_frame refactor: hourly prepare_dataframe output unchanged
# ---------------------------------------------------------------------------

class TestHourlyRefactorUnchanged:
    def test_prepare_dataframe_still_densifies_and_ffills(self):
        base_ts = int(pd.Timestamp("2023-01-01 00:00:00", tz="UTC").timestamp())
        # Hours 0, 1 and 3 trade; hour 2 is missing and must be filled.
        raw = pd.DataFrame(
            {
                "timestamp": [base_ts, base_ts + 3600, base_ts + 3 * 3600],
                "open": [100.0, 101.0, 103.0],
                "high": [100.5, 101.5, 103.5],
                "low": [99.5, 100.5, 102.5],
                "close": [100.0, 101.0, 103.0],
                "volume": [1.0, 2.0, 3.0],
                "trades": [10, 20, 30],
            }
        )
        out = prepare_dataframe(
            raw, "AAA", "USD", "AAA/USD",
            pd.Timestamp("2022-12-31 23:59:59", tz="UTC"),
            pd.Timestamp("2023-01-02 00:00:00", tz="UTC"),
        )
        assert len(out) == 4  # dense hourly index including the gap hour
        gap_row = out[out["ts"] == pd.Timestamp("2023-01-01 02:00:00", tz="UTC")]
        assert len(gap_row) == 1
        # filled from previous close, zero volume/trades
        assert float(gap_row["open"].iloc[0]) == 101.0
        assert float(gap_row["close"].iloc[0]) == 101.0
        assert float(gap_row["volume"].iloc[0]) == 0.0
        assert int(gap_row["trades"].iloc[0]) == 0

    def test_clean_raw_frame_sorts_dedupes_and_filters(self, sparse_minute_raw_df):
        out = _clean_raw_frame(sparse_minute_raw_df.copy(), START, END)
        assert out["ts"].is_monotonic_increasing
        assert not out["ts"].duplicated().any()
        assert (out["ts"] > START).all() and (out["ts"] <= END).all()


class TestCopyLoadSymbol:
    def test_empty_frame_returns_zero_without_engine(self):
        """The empty-df short-circuit fires before any DB access, so a dummy
        engine is never touched — guards against a no-op reload crashing."""
        from src.data.build_minute_database import copy_load_symbol
        assert copy_load_symbol(pd.DataFrame(columns=OHLCV_COLUMNS), engine=None) == 0

"""Database loaders for research panels."""

import pandas as pd
from sqlalchemy import text

ALLOWED_OHLCV_TABLES = {"ohlcv", "ohlcv_1m"}


def _to_utc_timestamp(value):
    """Return a UTC-aware pandas Timestamp from naive or tz-aware input."""
    ts = pd.Timestamp(value)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def list_universe_symbols(engine, as_of_date=None, top_n=None):
    """Return symbols from monthly_universe for a date."""
    if as_of_date is None:
        query = """
            SELECT rebalance_date, symbol, rank
            FROM monthly_universe
            ORDER BY rebalance_date, rank
        """
        params = {}
    else:
        query = """
            SELECT rebalance_date, symbol, rank
            FROM monthly_universe
            WHERE rebalance_date = (
                SELECT MAX(rebalance_date)
                FROM monthly_universe
                WHERE rebalance_date < :as_of_date
            )
            ORDER BY rank
        """
        params = {"as_of_date": pd.Timestamp(as_of_date).date()}

    df = pd.read_sql(text(query), engine, params=params)
    if top_n is not None:
        df = df[df["rank"] <= int(top_n)]
    return df.reset_index(drop=True)


def fetch_ohlcv_long(
    engine,
    start_ts,
    end_ts,
    symbols=None,
    table="ohlcv",
):
    """Load OHLCV in long format for a time window.

    `table="ohlcv"` (default) is the dense hourly table; `table="ohlcv_1m"`
    is the sparse 1-minute table (real trade bars only, no filled gaps).
    """
    if table not in ALLOWED_OHLCV_TABLES:
        raise ValueError(
            f"table must be one of {sorted(ALLOWED_OHLCV_TABLES)}, got {table!r}"
        )
    query = f"""
        SELECT
            ts,
            symbol,
            open,
            high,
            low,
            close,
            volume,
            trades,
            dollar_volume
        FROM {table}
        WHERE ts >= :start_ts
          AND ts <= :end_ts
    """
    params = {
        "start_ts": _to_utc_timestamp(start_ts),
        "end_ts": _to_utc_timestamp(end_ts),
    }

    if symbols:
        query += " AND symbol = ANY(:symbols)"
        params["symbols"] = list(symbols)

    query += " ORDER BY ts, symbol"
    df = pd.read_sql(text(query), engine, params=params)
    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    numeric_cols = ["open", "high", "low", "close", "volume", "trades", "dollar_volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_minute_exec_closes(
    engine,
    start_ts,
    end_ts,
    delay_minutes,
    symbols=None,
):
    """Fetch hourly-bucketed execution closes from the sparse 1m table.

    All bars are labelled at bar open, so the close of the signal bar
    labelled ``T`` is realized at wall-clock ``T+1h`` and execution
    ``delay_minutes`` after it happens at ``T+1h+delay``. The freshest fully
    formed 1m close at that instant is the last 1m bar labelled
    ``tau <= T+1h+delay-1min`` — exactly the bars grouped into bucket ``T``
    by ``date_trunc('hour', ts - delay)``. At ``delay_minutes=0`` the bucket
    is ``[T, T+1h)`` and the result equals the hourly ``ohlcv`` close.

    Returns a long frame ``(bucket_ts, symbol, exec_close)`` with one row per
    symbol-hour that had at least one trade in its bucket; empty buckets are
    simply absent (handled by LOCF in
    ``src.research.execution.build_execution_close_panel``).
    """
    delay = int(delay_minutes)
    if delay < 0:
        raise ValueError("delay_minutes must be >= 0")

    # Extend past end_ts so the final signal bars' buckets are complete.
    end_extended = _to_utc_timestamp(end_ts) + pd.Timedelta(hours=1, minutes=delay)

    query = f"""
        SELECT
            symbol,
            date_trunc('hour', ts - INTERVAL '{delay} minutes') AS bucket_ts,
            (array_agg(close ORDER BY ts DESC))[1] AS exec_close
        FROM ohlcv_1m
        WHERE ts >= :start_ts
          AND ts <= :end_ts
    """
    params = {
        "start_ts": _to_utc_timestamp(start_ts),
        "end_ts": end_extended,
    }
    if symbols:
        query += " AND symbol = ANY(:symbols)"
        params["symbols"] = list(symbols)
    query += " GROUP BY 1, 2 ORDER BY 2, 1"

    df = pd.read_sql(text(query), engine, params=params)
    if df.empty:
        return df

    df["bucket_ts"] = pd.to_datetime(df["bucket_ts"], utc=True)
    df["exec_close"] = pd.to_numeric(df["exec_close"], errors="coerce")
    return df

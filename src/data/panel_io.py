"""Database loaders for research panels."""

import pandas as pd
from sqlalchemy import text


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
):
    """Load hourly OHLCV in long format for a time window."""
    query = """
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
        FROM ohlcv
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

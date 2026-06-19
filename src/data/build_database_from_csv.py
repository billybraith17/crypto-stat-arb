"""Build Kraken market database from raw CSV files.

Steps:
1. Read raw Kraken CSV files
2. Load hourly USD pairs only (*_60.csv)
3. Fill missing no-trade hours
4. Store clean OHLCVT data in PostgreSQL
5. Build monthly top-n liquidity universe
"""

import os
import re

import pandas as pd
from pandas.errors import EmptyDataError
from sqlalchemy import text

from src.common.config import load_settings
from src.common.db import make_engine

# =====================================================
# TABLES
# =====================================================
def create_tables(engine):
    sql = """
    CREATE TABLE IF NOT EXISTS ohlcv (
        ts TIMESTAMPTZ NOT NULL,
        symbol TEXT NOT NULL,
        base_asset TEXT NOT NULL,
        quote_asset TEXT NOT NULL,
        open NUMERIC NOT NULL,
        high NUMERIC NOT NULL,
        low NUMERIC NOT NULL,
        close NUMERIC NOT NULL,
        volume NUMERIC NOT NULL,
        trades INTEGER NOT NULL,
        dollar_volume NUMERIC NOT NULL,
        PRIMARY KEY (ts, symbol)
    );

    CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_ts
    ON ohlcv(symbol, ts);

    CREATE TABLE IF NOT EXISTS monthly_universe (
        rebalance_date DATE NOT NULL,
        symbol TEXT NOT NULL,
        rank INTEGER NOT NULL,
        liquidity_monthly NUMERIC NOT NULL,
        PRIMARY KEY (rebalance_date, symbol)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(sql))

# =====================================================
# FILENAME PARSER
# Example: XBTUSD_60.csv
# =====================================================
def parse_filename(filename, excluded_bases):
    m = re.match(r"^([A-Z0-9]+)(USD)_(60)\.CSV$", filename.upper())
    if not m:
        return None

    base = m.group(1)
    quote = m.group(2)
    freq = int(m.group(3))
    symbol = f"{base}/{quote}"
    
    if base in excluded_bases:
        return None

    return base, quote, freq, symbol

# =====================================================
# READ CSV
# Handles header/no header files
# =====================================================
def read_raw_csv(path):
    cols = ["timestamp", "open", "high", "low", "close", "volume", "trades"]

    # try no header first and also handle EmptyDataError
    try:
        df = pd.read_csv(path, header=None)
    except EmptyDataError:
        return pd.DataFrame()

    # if file has header row, detect non-numeric timestamp
    first_val = str(df.iloc[0, 0])
    if not first_val.replace(".", "", 1).isdigit():
        df = pd.read_csv(path)

    df.columns = cols
    return df

# =====================================================
# CLEAN + FILL MISSING HOURS
# =====================================================
def prepare_dataframe(df, base, quote, symbol, data_start_date, data_end_date):
    # types
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    for c in ["open", "high", "low", "close", "volume", "trades"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # timestamp
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    # cutoff dates that shouldn't be included in the database
    start_cutoff = pd.Timestamp(data_start_date)
    end_cutoff = pd.Timestamp(data_end_date)
    df = df[(df["ts"] > start_cutoff) & (df["ts"] <= end_cutoff)]

    df = df.sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)

    # full hourly index
    full_index = pd.date_range(
        start=df["ts"].min(),
        end=df["ts"].max(),
        freq="1h",
        tz="UTC"
    )

    df = df.set_index("ts").reindex(full_index)
    df.index.name = "ts"

    # fill no-trade hours using previous close
    prev_close = df["close"].ffill()

    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].fillna(prev_close)

    df["volume"] = df["volume"].fillna(0.0)
    df["trades"] = df["trades"].fillna(0).astype(int)

    # drop leading rows if still NaN
    df = df.dropna(subset=["close"]).reset_index()

    # metadata
    df["symbol"] = symbol
    df["base_asset"] = base
    df["quote_asset"] = quote
    df["dollar_volume"] = df["close"] * df["volume"]

    return df[
        [
            "ts", "symbol", "base_asset", "quote_asset",
            "open", "high", "low", "close",
            "volume", "trades", "dollar_volume"
        ]
    ]

# =====================================================
# LOAD TO POSTGRES
# =====================================================
def upsert_table(df, table_name, key_cols, engine):
    if df.empty:
        return

    temp = "tmp_load"

    with engine.begin() as conn:
        df.to_sql(temp, conn, if_exists="replace", index=False)

        cols = list(df.columns)
        insert_cols = ", ".join(cols)
        select_cols = ", ".join(cols)

        conflict_cols = ", ".join(key_cols)

        update_cols = [
            c for c in cols if c not in key_cols
        ]

        update_sql = ", ".join(
            [f"{c}=EXCLUDED.{c}" for c in update_cols]
        )

        sql = f"""
        INSERT INTO {table_name} ({insert_cols})
        SELECT {select_cols}
        FROM {temp}
        ON CONFLICT ({conflict_cols})
        DO UPDATE SET {update_sql};
        """

        conn.execute(text(sql))
        conn.execute(text(f"DROP TABLE IF EXISTS {temp};"))

# =====================================================
# PROCESS ALL FILES
# =====================================================
def load_all_csvs(engine, settings):
    files = sorted(settings["raw_dir"].glob("*USD_60.csv"))
    # print(files)

    for path in files:
        parsed = parse_filename(path.name, settings["excluded_bases"])

        if parsed is None:
            continue

        base, quote, freq, symbol = parsed
        print(f"Loading {path.name} -> {symbol}")

        raw = read_raw_csv(path)

        if raw.empty:
            print(f"Empty data for {path.name}")
            continue

        clean = prepare_dataframe(
            raw,
            base,
            quote,
            symbol,
            settings["data_start_date"],
            settings["data_end_date"],
        )

        upsert_table(clean, "ohlcv", ["ts", "symbol"], engine)

# =====================================================
# BUILD MONTHLY UNIVERSE
# Top N by trailing liquidity with minimum thresholds
# =====================================================
def build_monthly_universe(engine, settings):
    sql = f"""
    DELETE FROM monthly_universe;

    WITH month_ends AS (
        SELECT DISTINCT
            (date_trunc('month', ts AT TIME ZONE 'UTC')
                + interval '1 month - 1 day')::date AS rebalance_date
        FROM ohlcv
    ),

    first_seen AS (
        SELECT
            symbol,
            MIN(ts) AS first_ts
        FROM ohlcv
        GROUP BY 1
    ),

    liquidity_monthly AS (
        SELECT
            m.rebalance_date,
            o.symbol,
            AVG(o.dollar_volume) AS liquidity_monthly,
            AVG(o.dollar_volume) FILTER (
                WHERE o.ts >= (m.rebalance_date - interval '7 day')
            ) AS liquidity_7d
        FROM month_ends m
        JOIN ohlcv o
          ON o.ts >= (m.rebalance_date - interval '{settings["lookback_days"]} day')
         AND o.ts <  m.rebalance_date
        JOIN first_seen fs
          ON fs.symbol = o.symbol
         AND fs.first_ts < (m.rebalance_date - interval '{settings["min_listing_days"]} day')
        GROUP BY 1,2
    ),

    ranked AS (
        SELECT
            rebalance_date,
            symbol,
            liquidity_monthly,
            liquidity_7d,
            ROW_NUMBER() OVER (
                PARTITION BY rebalance_date
                ORDER BY liquidity_monthly DESC
            ) AS rank
        FROM liquidity_monthly
        WHERE liquidity_monthly >= {settings["min_liquidity_monthly_threshold"]}
          AND liquidity_7d >= {settings["min_liquidity_7d_threshold"]}
    )

    INSERT INTO monthly_universe (rebalance_date, symbol, rank, liquidity_monthly)
    SELECT rebalance_date, symbol, rank, liquidity_monthly
    FROM ranked
    WHERE rank <= {settings["top_n"]};
    """

    with engine.begin() as conn:
        conn.execute(text(sql))

# =====================================================
# MAIN
# =====================================================
def main(engine, settings):
    create_tables(engine)
    load_all_csvs(engine, settings)
    build_monthly_universe(engine, settings)

if __name__ == "__main__":
    _settings = load_settings()
    _engine = make_engine(_settings)
    main(_engine, _settings)
    # create_tables(_engine)

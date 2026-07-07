"""Build the sparse 1-minute OHLCV table from raw Kraken CSV files.

Loads `{BASE}USD_{minutes}.csv` files only for symbols that ever appear in
`monthly_universe`, so the hourly `ohlcv` table (and the universe build that
depends on it) must exist before this runs.

Unlike the hourly loader, rows are stored *sparse*: no-trade minutes are not
reindexed or forward-filled — only real trade bars are kept. Regularisation
happens at read time (see `src.data.panel_io.fetch_minute_exec_closes` and
`src.research.execution`).
"""

import io

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.common.config import load_settings
from src.common.db import make_engine
from src.data.build_database_from_csv import _clean_raw_frame, read_raw_csv

OHLCV_COLUMNS = [
    "ts", "symbol", "base_asset", "quote_asset",
    "open", "high", "low", "close",
    "volume", "trades", "dollar_volume",
]

COPY_CHUNK_ROWS = 1_000_000


# =====================================================
# TABLES
# =====================================================
def create_minute_tables(engine):
    """Create the bare table. The (ts, symbol) primary key and secondary index
    are added by `finalize_minute_table` after the bulk load — COPYing ~100M+
    rows through a live btree is what dominates load time otherwise."""
    sql = """
    CREATE TABLE IF NOT EXISTS ohlcv_1m (
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
        dollar_volume NUMERIC NOT NULL
    );
    """
    with engine.begin() as conn:
        conn.execute(text(sql))


def _has_primary_key(engine, table="ohlcv_1m"):
    query = text("""
        SELECT 1 FROM pg_constraint
        WHERE conrelid = :table ::regclass AND contype = 'p'
    """)
    with engine.connect() as conn:
        return conn.execute(query, {"table": table}).first() is not None


def _table_is_empty(engine, table="ohlcv_1m"):
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT 1 FROM {table} LIMIT 1")).first() is None


def create_secondary_index(engine):
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_1m_symbol_ts "
            "ON ohlcv_1m(symbol, ts);"
        ))


def finalize_minute_table(engine):
    """Add the primary key (if missing) and secondary index, then ANALYZE."""
    if not _has_primary_key(engine):
        with engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE ohlcv_1m ADD PRIMARY KEY (ts, symbol);"
            ))
    create_secondary_index(engine)
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("ANALYZE ohlcv_1m;"))


# =====================================================
# SYMBOL / FILE RESOLUTION
# =====================================================
def list_universe_symbols_for_minutes(engine):
    """Symbols that ever appear in monthly_universe — the load scope."""
    df = pd.read_sql(
        text("SELECT DISTINCT symbol FROM monthly_universe ORDER BY symbol"),
        engine,
    )
    return df["symbol"].tolist()


def minute_csv_path(raw_dir, symbol, timeframe_minutes):
    """Map a universe symbol like 'XBT/USD' to raw_dir/'XBTUSD_1.csv'."""
    base, quote = symbol.split("/")
    return raw_dir / f"{base}{quote}_{int(timeframe_minutes)}.csv"


# =====================================================
# CLEAN (SPARSE — no reindex/ffill, real trade bars only)
# =====================================================
def prepare_minute_dataframe(df, base, quote, symbol, data_start_date, data_end_date):
    df = _clean_raw_frame(df, data_start_date, data_end_date)
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    df["symbol"] = symbol
    df["base_asset"] = base
    df["quote_asset"] = quote
    df["trades"] = df["trades"].fillna(0).astype(int)
    df["dollar_volume"] = df["close"] * df["volume"]

    return df[OHLCV_COLUMNS]


# =====================================================
# LOAD TO POSTGRES (COPY-based; delete+COPY per symbol is idempotent)
# =====================================================
def format_ts_for_copy(ts_series):
    """Vectorised UTC ISO strings for COPY: per-row strftime via to_csv's
    date_format roughly doubles serialization time at this scale. The explicit
    +00 offset is load-bearing — without it Postgres parses the literal in the
    *session* timezone, silently shifting DST-period rows."""
    return np.char.add(
        np.datetime_as_string(
            ts_series.dt.tz_convert("UTC").values.astype("datetime64[s]"), unit="s"
        ),
        "+00",
    )


def copy_load_symbol(df, engine, table="ohlcv_1m", delete_existing=True):
    if df.empty:
        return 0

    df = df.copy()
    df["ts"] = format_ts_for_copy(df["ts"])

    cols = ", ".join(OHLCV_COLUMNS)
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            if delete_existing:
                cur.execute(
                    f"DELETE FROM {table} WHERE symbol = %s", (df["symbol"].iloc[0],)
                )
            for start in range(0, len(df), COPY_CHUNK_ROWS):
                chunk = df.iloc[start:start + COPY_CHUNK_ROWS]
                buf = io.StringIO()
                chunk.to_csv(buf, index=False, header=False)
                buf.seek(0)
                cur.copy_expert(
                    f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv)", buf
                )
        raw_conn.commit()
    except Exception:
        raw_conn.rollback()
        raise
    finally:
        raw_conn.close()
    return len(df)


def _symbol_already_loaded(engine, symbol, table="ohlcv_1m"):
    query = text(f"SELECT 1 FROM {table} WHERE symbol = :symbol LIMIT 1")
    with engine.connect() as conn:
        return conn.execute(query, {"symbol": symbol}).first() is not None


# =====================================================
# PROCESS ALL UNIVERSE SYMBOLS
# =====================================================
def load_minute_csvs(engine, settings, fresh_table=False):
    """Load 1m CSVs for universe symbols.

    ``fresh_table=True`` (empty, index-free table) skips the per-symbol
    presence checks and DELETEs — without indexes both would seqscan an
    ever-growing table once per symbol.
    """
    symbols = list_universe_symbols_for_minutes(engine)
    if not symbols:
        raise ValueError(
            "monthly_universe is empty — run the hourly build "
            "(make build-db) before loading minute data."
        )

    timeframe_minutes = settings["minute_timeframe_minutes"]
    reload_mode = settings["minute_reload_mode"]
    total_rows = 0
    loaded, skipped, missing = 0, 0, 0

    for symbol in symbols:
        if (
            not fresh_table
            and reload_mode == "if_empty"
            and _symbol_already_loaded(engine, symbol)
        ):
            print(f"Skipping {symbol} (already loaded)")
            skipped += 1
            continue

        path = minute_csv_path(settings["raw_dir"], symbol, timeframe_minutes)
        if not path.exists():
            print(f"WARNING: no minute CSV for {symbol} ({path.name}), skipping")
            missing += 1
            continue

        raw = read_raw_csv(path)
        if raw.empty:
            print(f"Empty data for {path.name}")
            missing += 1
            continue

        base, quote = symbol.split("/")
        clean = prepare_minute_dataframe(
            raw, base, quote, symbol,
            settings["data_start_date"], settings["data_end_date"],
        )
        n = copy_load_symbol(clean, engine, delete_existing=not fresh_table)
        total_rows += n
        loaded += 1
        print(f"Loaded {path.name} -> {symbol}: {n:,} rows ({total_rows:,} total)")

    print(
        f"Minute load complete: {loaded} loaded, {skipped} skipped, "
        f"{missing} missing, {total_rows:,} rows inserted"
    )
    return total_rows


# =====================================================
# MAIN
# =====================================================
def main(engine, settings):
    create_minute_tables(engine)
    fresh = _table_is_empty(engine)
    if not fresh:
        # Resuming/reloading: make the per-symbol presence checks and DELETEs
        # index-fast before touching data.
        create_secondary_index(engine)
    load_minute_csvs(engine, settings, fresh_table=fresh)
    finalize_minute_table(engine)


if __name__ == "__main__":
    _settings = load_settings()
    _engine = make_engine(_settings)
    main(_engine, _settings)

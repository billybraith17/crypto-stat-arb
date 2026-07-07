"""Hard data-quality checks for market data."""

import pandas as pd
from sqlalchemy import text

ALLOWED_TABLES = {"ohlcv", "ohlcv_1m"}


def _count(engine, query):
    return int(pd.read_sql(query, engine).iloc[0, 0])


def run_checks(engine, table="ohlcv", include_gap_check=True):
    """Run hard checks. Raises ValueError if any check fails.

    The gap check asserts consecutive rows are at most one hour apart, which
    only makes sense for the dense hourly table — `ohlcv_1m` stores sparse
    real-trade minutes, so callers must disable it there.
    """
    if table not in ALLOWED_TABLES:
        raise ValueError(f"table must be one of {sorted(ALLOWED_TABLES)}, got {table!r}")

    check_queries = {
        "duplicate_keys": f"""
            SELECT COUNT(*) AS n
            FROM (
              SELECT ts, symbol, COUNT(*) c
              FROM {table}
              GROUP BY 1,2
              HAVING COUNT(*) > 1
            ) x
        """,
        "null_rows": f"""
            SELECT COUNT(*)
            FROM {table}
            WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
        """,
        "bad_ohlc": f"""
            SELECT COUNT(*)
            FROM {table}
            WHERE high < GREATEST(open, close)
               OR low  > LEAST(open, close)
               OR high < low
        """,
        "negative_vals": f"""
            SELECT COUNT(*)
            FROM {table}
            WHERE volume < 0 OR trades < 0
        """,
    }
    if include_gap_check:
        check_queries["gaps_sample"] = f"""
            WITH x AS (
              SELECT symbol, ts,
                     ts - LAG(ts) OVER (PARTITION BY symbol ORDER BY ts) AS diff
              FROM {table}
            )
            SELECT COUNT(*) FROM x WHERE diff > INTERVAL '1 hour'
        """

    failed = []
    results = {}
    for name, query in check_queries.items():
        value = _count(engine, query)
        results[name] = value
        print(f"{name}: {value}")
        if value > 0:
            failed.append(f"{name}={value}")

    if failed:
        raise ValueError("Failed data-quality checks: " + ", ".join(failed))

    return results


def _check_cross_timeframe_consistency(engine, settings):
    """Compare 1m closes aggregated to hourly against the hourly table.

    Both tables label bars at bar open, so the last 1m close in
    [hour, hour+1h) must equal the hourly close for the same label — both
    derive from the same trades. Only *real* hourly bars (volume > 0) are
    compared: synthetic forward-filled bars either have no 1m rows (genuine
    no-trade hour, drops out of the join) or, rarely, the raw hourly dump is
    missing a bar that minute data has (e.g. XBT/USD 2024-03-31 23:00) — the
    1m side is the more complete source there, not an inconsistency. Sampled
    on the most liquid universe symbols to keep the scan cheap.
    """
    n_sample = int(settings["minute_consistency_sample_symbols"])
    rel_tol = float(settings["minute_consistency_rel_tol"])

    symbols_query = text("""
        SELECT symbol
        FROM monthly_universe
        GROUP BY symbol
        ORDER BY AVG(rank) ASC
        LIMIT :n
    """)
    symbols = pd.read_sql(symbols_query, engine, params={"n": n_sample})["symbol"].tolist()
    if not symbols:
        raise ValueError("monthly_universe is empty — cannot run consistency check")

    query = text("""
        WITH m AS (
            SELECT symbol,
                   date_trunc('hour', ts) AS hr,
                   (array_agg(close ORDER BY ts DESC))[1] AS m_close
            FROM ohlcv_1m
            WHERE symbol = ANY(:symbols)
            GROUP BY 1, 2
        )
        SELECT COUNT(*) AS n
        FROM m
        JOIN ohlcv h ON h.symbol = m.symbol AND h.ts = m.hr
        WHERE h.volume > 0
          AND ABS(h.close - m.m_close) / NULLIF(ABS(h.close), 0) > :rel_tol
    """)
    with engine.connect() as conn:
        mismatches = int(conn.execute(
            query, {"symbols": symbols, "rel_tol": rel_tol}
        ).scalar())

    print(f"cross_timeframe_consistency ({len(symbols)} symbols): {mismatches}")
    if mismatches > 0:
        raise ValueError(
            f"1m→hourly close mismatch on {mismatches} bars "
            f"(symbols sampled: {symbols}, rel_tol={rel_tol})"
        )
    return mismatches


def run_minute_checks(engine, settings):
    """Hard checks for the sparse ohlcv_1m table.

    Runs the frequency-agnostic base checks (gap check excluded — sparse
    minutes are expected) plus a sampled 1m→hourly consistency check.
    """
    results = run_checks(engine, table="ohlcv_1m", include_gap_check=False)
    results["cross_timeframe_consistency"] = _check_cross_timeframe_consistency(
        engine, settings
    )
    return results

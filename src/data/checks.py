"""Hard data-quality checks for market data."""

import pandas as pd


def _count(engine, query):
    return int(pd.read_sql(query, engine).iloc[0, 0])


def run_checks(engine):
    """Run hard checks. Raises ValueError if any check fails."""
    check_queries = {
        "duplicate_keys": """
            SELECT COUNT(*) AS n
            FROM (
              SELECT ts, symbol, COUNT(*) c
              FROM ohlcv
              GROUP BY 1,2
              HAVING COUNT(*) > 1
            ) x
        """,
        "null_rows": """
            SELECT COUNT(*)
            FROM ohlcv
            WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
        """,
        "bad_ohlc": """
            SELECT COUNT(*)
            FROM ohlcv
            WHERE high < GREATEST(open, close)
               OR low  > LEAST(open, close)
               OR high < low
        """,
        "negative_vals": """
            SELECT COUNT(*)
            FROM ohlcv
            WHERE volume < 0 OR trades < 0
        """,
        "gaps_sample": """
            WITH x AS (
              SELECT symbol, ts,
                     ts - LAG(ts) OVER (PARTITION BY symbol ORDER BY ts) AS diff
              FROM ohlcv
            )
            SELECT COUNT(*) FROM x WHERE diff > INTERVAL '1 hour'
        """,
    }

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

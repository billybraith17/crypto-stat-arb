"""Soft warning diagnostics and event extraction for market data."""

import pandas as pd
from sqlalchemy import text


def _get_extreme_return_details(engine, threshold):
    return pd.read_sql(
        text(
            """
        WITH universe_symbols AS (
            SELECT DISTINCT symbol
            FROM monthly_universe
        ),
        r AS (
            SELECT
                symbol,
                ts,
                close / LAG(close) OVER (
                    PARTITION BY symbol ORDER BY ts
                ) - 1 AS ret
            FROM ohlcv
            WHERE symbol IN (SELECT symbol FROM universe_symbols)
        )
        SELECT symbol, ts, ret
        FROM r
        WHERE ABS(ret) >= :threshold
        ORDER BY ts, symbol
        """
        ),
        engine,
        params={"threshold": threshold},
    )


def _get_large_symbol_gap_details(engine, threshold):
    return pd.read_sql(
        text(
            """
        WITH universe_symbols AS (
            SELECT DISTINCT symbol
            FROM monthly_universe
        ),
        x AS (
            SELECT
                symbol,
                ts,
                EXTRACT(EPOCH FROM (
                    ts - LAG(ts) OVER (
                        PARTITION BY symbol ORDER BY ts
                    )
                )) / 3600.0 AS gap_hours
            FROM ohlcv
            WHERE symbol IN (SELECT symbol FROM universe_symbols)
        )
        SELECT symbol, ts, gap_hours
        FROM x
        WHERE gap_hours > :threshold
        ORDER BY ts, symbol
        """
        ),
        engine,
        params={"threshold": threshold},
    )


def _get_liquidity_collapse_details(engine, threshold):
    return pd.read_sql(
        text(
            """
        WITH universe_symbols AS (
            SELECT DISTINCT symbol
            FROM monthly_universe
        ),
        d AS (
            SELECT
                symbol,
                (date_trunc('day', ts AT TIME ZONE 'UTC') AT TIME ZONE 'UTC')
                    AS d,
                AVG(dollar_volume) AS dv
            FROM ohlcv
            WHERE symbol IN (SELECT symbol FROM universe_symbols)
            GROUP BY 1,2
        ),
        x AS (
            SELECT
                symbol,
                d,
                AVG(dv) OVER (
                    PARTITION BY symbol
                    ORDER BY d
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) AS avg_7d,
                AVG(dv) OVER (
                    PARTITION BY symbol
                    ORDER BY d
                    ROWS BETWEEN 36 PRECEDING AND 7 PRECEDING
                ) AS prev_30d
            FROM d
        ),
        flagged AS (
            SELECT
                symbol,
                d,
                avg_7d,
                prev_30d,
                (prev_30d > 0 AND avg_7d < :threshold * prev_30d) AS is_collapse
            FROM x
            WHERE prev_30d IS NOT NULL
        ),
        onset AS (
            SELECT
                symbol,
                d,
                avg_7d,
                prev_30d
            FROM (
                SELECT
                    symbol,
                    d,
                    avg_7d,
                    prev_30d,
                    is_collapse,
                    LAG(is_collapse, 1, FALSE) OVER (
                        PARTITION BY symbol
                        ORDER BY d
                    ) AS was_collapse
                FROM flagged
            ) z
            WHERE is_collapse AND NOT was_collapse
        )
        SELECT symbol, d, avg_7d, prev_30d
        FROM onset
        ORDER BY d, symbol
        """
        ),
        engine,
        params={"threshold": threshold},
    )


def _run_symbol_lifecycle_warnings(engine, thresholds):
    """Flag new listings, delistings and short-lived symbols."""
    q = """
    WITH universe_symbols AS (
        SELECT DISTINCT symbol
        FROM monthly_universe
    ),
    bounds AS (
        SELECT
            symbol,
            MIN(ts) AS first_ts,
            MAX(ts) AS last_ts,
            COUNT(*) AS rows
        FROM ohlcv
        WHERE symbol IN (SELECT symbol FROM universe_symbols)
        GROUP BY symbol
    ),
    global_bounds AS (
        SELECT
            MIN(ts) AS global_start,
            MAX(ts) AS global_end
        FROM ohlcv
        WHERE symbol IN (SELECT symbol FROM universe_symbols)
    )
    SELECT
        b.symbol,
        b.first_ts,
        b.last_ts,
        b.rows,
        (b.first_ts - g.global_start) AS starts_late_by,
        (g.global_end - b.last_ts) AS ends_early_by
    FROM bounds b
    CROSS JOIN global_bounds g
    ORDER BY b.symbol
    """

    df = pd.read_sql(q, engine)

    late_threshold_days = thresholds["lifecycle_start_late_days"]
    early_threshold_days = thresholds["lifecycle_end_early_days"]
    short_rows_threshold = thresholds["short_life_min_rows"]

    starts_late = df[df["starts_late_by"].dt.days > late_threshold_days].sort_values(
        by="first_ts"
    )
    ends_early = df[df["ends_early_by"].dt.days > early_threshold_days].sort_values(
        by="last_ts"
    )
    short_life = df[df["rows"] < short_rows_threshold].sort_values(by="symbol")

    return {
        "starts_late": starts_late,
        "ends_early": ends_early,
        "short_life": short_life,
    }


def _get_universe_turnover_details(engine, threshold):
    universe_df = pd.read_sql(
        """
        SELECT rebalance_date, symbol
        FROM monthly_universe
        ORDER BY rebalance_date
        """,
        engine,
    )

    turnover_rows = []
    dates = sorted(universe_df["rebalance_date"].unique())
    for d_prev, d_cur in zip(dates[:-1], dates[1:]):
        prev_symbols = set(universe_df[universe_df["rebalance_date"] == d_prev]["symbol"])
        cur_symbols = set(universe_df[universe_df["rebalance_date"] == d_cur]["symbol"])
        if prev_symbols:
            turnover = 1 - len(prev_symbols & cur_symbols) / len(prev_symbols)
            if turnover > threshold:
                turnover_rows.append(
                    {
                        "d_prev": d_prev,
                        "d_cur": d_cur,
                        "turnover": float(turnover),
                    }
                )

    return pd.DataFrame(turnover_rows)


def _build_event_details(
    extreme_return_details,
    large_symbol_gap_details,
    liquidity_collapse_details,
    universe_turnover_details,
    lifecycle_warnings,
    thresholds,
):
    event_details = []
    large_gap_hours = thresholds["large_symbol_gap_hours"]
    universe_turnover_threshold = thresholds["high_universe_turnover"]
    start_late_days = thresholds["lifecycle_start_late_days"]
    end_early_days = thresholds["lifecycle_end_early_days"]

    for _, row in extreme_return_details.iterrows():
        event_details.append(
            {
                "event_type": "extreme return detected",
                "symbol": row["symbol"],
                "ts": row["ts"],
                "metric_value": float(abs(row["ret"])),
                "threshold": float(thresholds["extreme_return_abs"]),
                "notes": f"hourly_return={float(row['ret']):.6f}",
            }
        )

    for _, row in large_symbol_gap_details.iterrows():
        event_details.append(
            {
                "event_type": "suspicious gap detected",
                "symbol": row["symbol"],
                "ts": row["ts"],
                "metric_value": float(row["gap_hours"]),
                "threshold": float(large_gap_hours),
                "notes": (
                    "gap between consecutive observations exceeded "
                    f"{float(large_gap_hours):g}h"
                ),
            }
        )

    for _, row in liquidity_collapse_details.iterrows():
        ratio = float(row["avg_7d"] / row["prev_30d"]) if row["prev_30d"] else None
        event_details.append(
            {
                "event_type": "liquidity collapse detected",
                "symbol": row["symbol"],
                "ts": row["d"],
                "metric_value": ratio,
                "threshold": float(thresholds["liquidity_collapse_ratio"]),
                "notes": (
                    f"avg_7d={float(row['avg_7d']):.6f}, "
                    f"prev_30d={float(row['prev_30d']):.6f}"
                ),
            }
        )

    for _, row in universe_turnover_details.iterrows():
        event_details.append(
            {
                "event_type": "high universe turnover detected",
                "symbol": None,
                "ts": row["d_cur"],
                "metric_value": float(row["turnover"]),
                "threshold": float(universe_turnover_threshold),
                "notes": (
                    f"turnover={float(row['turnover'])}"
                ),
            }
        )

    for _, row in lifecycle_warnings["starts_late"].iterrows():
        event_details.append(
            {
                "event_type": "symbol started late",
                "symbol": row["symbol"],
                "ts": row["first_ts"],
                "metric_value": float(row["starts_late_by"].total_seconds() / 86400.0),
                "threshold": float(start_late_days),
                "notes": (
                    f"symbol started >{float(start_late_days):g} days after global start"
                ),
            }
        )

    for _, row in lifecycle_warnings["ends_early"].iterrows():
        event_details.append(
            {
                "event_type": "symbol ended early",
                "symbol": row["symbol"],
                "ts": row["last_ts"],
                "metric_value": float(row["ends_early_by"].total_seconds() / 86400.0),
                "threshold": float(end_early_days),
                "notes": (
                    f"symbol ended >{float(end_early_days):g} days before global end"
                ),
            }
        )

    for _, row in lifecycle_warnings["short_life"].iterrows():
        life_days = float((row["last_ts"] - row["first_ts"]).total_seconds() / 86400.0)
        event_details.append(
            {
                "event_type": "symbol has short life",
                "symbol": row["symbol"],
                "ts": row["first_ts"],
                "metric_value": life_days,
                "threshold": float(thresholds["short_life_min_rows"]),
                "notes": f"symbol exists for {life_days:.6f} days",
            }
        )

    return event_details


def run_warnings(engine, settings):
    """Run soft checks and return summary counts, lifecycle details and events."""
    thresholds = settings["warning_thresholds"]

    extreme_return_details = _get_extreme_return_details(
        engine, thresholds["extreme_return_abs"]
    )
    large_symbol_gap_details = _get_large_symbol_gap_details(
        engine, thresholds["large_symbol_gap_hours"]
    )
    liquidity_collapse_details = _get_liquidity_collapse_details(
        engine, thresholds["liquidity_collapse_ratio"]
    )
    universe_turnover_details = _get_universe_turnover_details(
        engine, thresholds["high_universe_turnover"]
    )
    lifecycle_warnings = _run_symbol_lifecycle_warnings(engine, thresholds)

    warnings = {
        "extreme_returns": len(extreme_return_details),
        "large_symbol_gaps": len(large_symbol_gap_details),
        "liquidity_collapse": len(liquidity_collapse_details),
        "high_universe_turnover": len(universe_turnover_details),
        "starts_late": len(lifecycle_warnings["starts_late"]),
        "ends_early": len(lifecycle_warnings["ends_early"]),
        "short_life": len(lifecycle_warnings["short_life"]),
    }

    event_details = _build_event_details(
        extreme_return_details,
        large_symbol_gap_details,
        liquidity_collapse_details,
        universe_turnover_details,
        lifecycle_warnings,
        thresholds,
    )

    for name, value in warnings.items():
        print(f"WARNING - {name}: {value}")

    return warnings, lifecycle_warnings, event_details

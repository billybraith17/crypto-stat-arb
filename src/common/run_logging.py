"""Pipeline run-level logging utilities for summary and diagnostics events."""

from sqlalchemy import text


def create_run_logging_tables(engine):
    """Create run summary and event tables if they do not exist."""
    sql = """
    CREATE TABLE IF NOT EXISTS pipeline_run_summary (
        run_id TEXT PRIMARY KEY,
        run_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        pipeline_name TEXT NOT NULL,
        version_tag TEXT,
        top_n INTEGER,
        rows_loaded BIGINT NOT NULL,
        symbols_loaded INTEGER NOT NULL,
        checks_passed BOOLEAN NOT NULL,
        warning_count INTEGER NOT NULL,
        status TEXT NOT NULL,
        error_message TEXT
    );

    CREATE TABLE IF NOT EXISTS pipeline_run_events (
        event_id BIGSERIAL PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES pipeline_run_summary(run_id) ON DELETE CASCADE,
        event_type TEXT NOT NULL,
        symbol TEXT,
        ts TIMESTAMPTZ,
        metric_value DOUBLE PRECISION,
        threshold DOUBLE PRECISION,
        notes TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_pipeline_run_events_run_id
    ON pipeline_run_events(run_id);

    CREATE INDEX IF NOT EXISTS idx_pipeline_run_events_event_type
    ON pipeline_run_events(event_type);
    """
    with engine.begin() as conn:
        conn.execute(text(sql))


def collect_run_metrics(engine):
    """Collect aggregate metrics for summary run logging."""
    metrics_sql = """
    SELECT
        COUNT(*) AS rows_loaded,
        COUNT(DISTINCT symbol) AS symbols_loaded
    FROM ohlcv
    """

    with engine.begin() as conn:
        row = conn.execute(text(metrics_sql)).mappings().one()
        return {
            "rows_loaded": int(row["rows_loaded"] or 0),
            "symbols_loaded": int(row["symbols_loaded"] or 0),
        }


def log_pipeline_run(engine, summary, event_details):
    """Persist a single run summary and its associated event details."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO pipeline_run_summary (
                    run_id,
                    run_ts,
                    pipeline_name,
                    version_tag,
                    top_n,
                    rows_loaded,
                    symbols_loaded,
                    checks_passed,
                    warning_count,
                    status,
                    error_message
                )
                VALUES (
                    :run_id,
                    :run_ts,
                    :pipeline_name,
                    :version_tag,
                    :top_n,
                    :rows_loaded,
                    :symbols_loaded,
                    :checks_passed,
                    :warning_count,
                    :status,
                    :error_message
                )
                """
            ),
            summary,
        )

        if event_details:
            event_rows = []
            for event in event_details:
                event_rows.append(
                    {
                        "run_id": summary["run_id"],
                        "event_type": event.get("event_type"),
                        "symbol": event.get("symbol"),
                        "ts": event.get("ts"),
                        "metric_value": event.get("metric_value"),
                        "threshold": event.get("threshold"),
                        "notes": event.get("notes"),
                    }
                )

            conn.execute(
                text(
                    """
                    INSERT INTO pipeline_run_events (
                        run_id,
                        event_type,
                        symbol,
                        ts,
                        metric_value,
                        threshold,
                        notes
                    )
                    VALUES (
                        :run_id,
                        :event_type,
                        :symbol,
                        :ts,
                        :metric_value,
                        :threshold,
                        :notes
                    )
                    """
                ),
                event_rows,
            )

"""Pipeline orchestration helpers."""

from datetime import datetime, timezone
from uuid import uuid4

from src.common.run_logging import (
    collect_run_metrics,
    create_run_logging_tables,
    log_pipeline_run,
)
from src.data.build_database_from_csv import main as build_database
from src.data.checks import run_checks
from src.data.warnings import run_warnings


def execute_pipeline(engine, settings):
    """Run build, checks, warnings and persistent run logging."""
    create_run_logging_tables(engine)

    run_id = str(uuid4())
    run_ts = datetime.now(timezone.utc)
    pipeline_name = settings.get("pipeline_name", "core_pipeline")
    version_tag = settings.get("version_tag")

    checks_passed = False
    warnings = {}
    event_details = []
    status = "SUCCESS"
    error_message = None

    try:
        print("=== STEP 1: BUILD DATABASE ===")
        build_database(engine, settings)

        print("\n=== STEP 2: DATA QUALITY CHECKS ===")
        run_checks(engine)
        checks_passed = True
        warnings, _, event_details = run_warnings(engine, settings)

        print("\nPipeline completed successfully.")
    except Exception as exc:
        status = "FAILED"
        error_message = str(exc)
        print(f"\nPipeline failed: {exc}")
        raise
    finally:
        try:
            metrics = collect_run_metrics(engine)
            warning_count = int(sum(warnings.values())) if warnings else 0

            summary = {
                "run_id": run_id,
                "run_ts": run_ts,
                "pipeline_name": pipeline_name,
                "version_tag": version_tag,
                "top_n": int(settings["top_n"]) if "top_n" in settings else None,
                "rows_loaded": metrics["rows_loaded"],
                "symbols_loaded": metrics["symbols_loaded"],
                "checks_passed": checks_passed,
                "warning_count": warning_count,
                "status": status,
                "error_message": error_message,
            }
            log_pipeline_run(engine, summary, event_details)
            print(f"Run logging persisted. run_id={run_id}")
        except Exception as log_exc:
            print(f"Run logging failed (non-blocking): {log_exc}")

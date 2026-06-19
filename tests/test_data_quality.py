"""Compatibility wrapper for quality checks and warnings.

The implementation now lives in src/quality/.
"""

from src.common.config import load_settings
from src.common.db import make_engine
from src.data.checks import run_checks
from src.data.warnings import run_warnings


if __name__ == "__main__":
    _settings = load_settings()
    _engine = make_engine(_settings)
    run_checks(_engine)
    summary, lifecycle, events = run_warnings(_engine, _settings)
    print(lifecycle)

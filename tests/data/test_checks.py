"""DB-free validation tests for src/data/checks.py.

run_checks executes SQL, but the table whitelist fires before any query is
built (guarding the f-string interpolation), so the reject path is testable
with a dummy engine.
"""

import pytest

from src.data.checks import ALLOWED_TABLES, run_checks


class TestRunChecksTableGuard:
    def test_allowed_tables(self):
        assert ALLOWED_TABLES == {"ohlcv", "ohlcv_1m"}

    def test_rejects_unknown_table(self):
        with pytest.raises(ValueError, match="table must be one of"):
            run_checks(None, table="ohlcv_1m; DROP TABLE ohlcv")

    def test_rejects_arbitrary_identifier(self):
        with pytest.raises(ValueError):
            run_checks(None, table="pg_catalog.pg_user")

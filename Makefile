.PHONY: help build-db build-db-1m pipeline quality-checks quality-warnings quality-all

PYTHON ?= python3

help:
	@echo "Available targets:"
	@echo "  make build-db          Build database from raw CSV files"
	@echo "  make build-db-1m       Build sparse 1-minute table (needs monthly_universe)"
	@echo "  make pipeline          Run full project pipeline"
	@echo "  make quality           Run data-quality checks and warnings"

build-db:
	$(PYTHON) -m src.data.build_database_from_csv

build-db-1m:
	$(PYTHON) -m src.data.build_minute_database

pipeline:
	$(PYTHON) -m run_pipeline

quality:
	$(PYTHON) -m tests.test_data_quality

.PHONY: help build-db pipeline quality-checks quality-warnings quality-all

PYTHON ?= python3

help:
	@echo "Available targets:"
	@echo "  make build-db          Build database from raw CSV files"
	@echo "  make pipeline          Run full project pipeline"
	@echo "  make quality           Run data-quality checks and warnings"

build-db:
	$(PYTHON) -m src.data.build_database_from_csv

pipeline:
	$(PYTHON) -m run_pipeline

quality:
	$(PYTHON) -m tests.test_data_quality

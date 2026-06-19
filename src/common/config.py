"""Load project settings from YAML and environment variables."""

import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv


def load_settings(config_path="configs/base.yaml"):
    load_dotenv()

    with open(Path(config_path), "r") as f:
        cfg = yaml.safe_load(f)

    settings = {
        "top_n": cfg["top_n"],
        "lookback_days": cfg["lookback_days"],
        "min_listing_days": cfg.get("min_listing_days", 90),
        "min_liquidity_monthly_threshold": cfg.get("min_liquidity_monthly_threshold", 0.0),
        "min_liquidity_7d_threshold": cfg.get("min_liquidity_7d_threshold", 0.0),
        "warning_thresholds": cfg["warning_thresholds"],
        "raw_dir": Path(cfg["raw_dir"]),
        "excluded_bases": set(cfg["excluded_bases"]),
        "data_start_date": pd.Timestamp(cfg["data_start_date"]),
        "data_end_date": pd.Timestamp(cfg["data_end_date"]),

        "pg_host": os.getenv("PG_HOST", "localhost"),
        "pg_port": os.getenv("PG_PORT", "5432"),
        "pg_db": os.getenv("PG_DB", "crypto"),
        "pg_user": os.getenv("PG_USER", "postgres"),
        "pg_password": os.getenv("PG_PASSWORD", ""),
    }

    return settings
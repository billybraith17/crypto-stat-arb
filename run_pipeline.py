"""Run the core project pipeline in sequence."""

# import argparse

from src.common.config import load_settings
from src.common.db import make_engine
from src.common.pipeline_runner import execute_pipeline


def main():
    # # Not necessary right now but wanted for later
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--skip-build",
    #     action="store_true",
    #     help="Skip database build and run quality checks only.",
    # )
    # args = parser.parse_args()

    # if args.skip_build:
    #     print("=== STEP 1: BUILD DATABASE (SKIPPED) ===")
    # else:

    settings = load_settings()
    engine = make_engine(settings)
    execute_pipeline(engine, settings)


if __name__ == "__main__":
    main()

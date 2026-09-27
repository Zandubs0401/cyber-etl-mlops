"""Run the whole pipeline end to end from the command line (no Airflow needed).

    python -m cyber_pipeline.pipeline               # generate data if missing, ETL, train
    python -m cyber_pipeline.pipeline --no-api      # skip the live NVD API (use offline sample)
"""
from __future__ import annotations

import argparse

from . import config
from .extract import run_extract
from .generate_data import main as generate
from .load import load_features
from .logging_config import get_logger, log_duration
from .train import main as train
from .transform import run_transform

log = get_logger("pipeline")


def run(use_api: bool = True, rows: int = 300_000, max_train_rows: int | None = None) -> dict:
    with log_duration(log, "FULL PIPELINE"):
        if not config.AUTH_EVENTS_CSV.exists():
            generate(rows)
        staging = run_extract(use_api=use_api)
        features = run_transform(staging)
        load_features(features)
        return train(max_train_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-api", action="store_true", help="use the offline NVD sample")
    ap.add_argument("--rows", type=int, default=300_000)
    ap.add_argument("--max-train-rows", type=int, default=None)
    a = ap.parse_args()
    print(run(use_api=not a.no_api, rows=a.rows, max_train_rows=a.max_train_rows))

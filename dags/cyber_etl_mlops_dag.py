"""Airflow DAG: daily ETL + model retraining for suspicious-login detection.

    extract_auth_events ─┐
    extract_assets ──────┼─> transform_and_load ─> train_and_register ─> reload_api
    extract_vulns (API) ─┘

The three extract tasks run in parallel. Failed tasks retry automatically, and every task
writes to the shared pipeline log as well as the Airflow task log.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make the project package importable from the DAG folder
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:  # Airflow 3
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task

API_URL = os.getenv("MODEL_API_URL", "http://localhost:8000")
log = logging.getLogger(__name__)


def notify_failure(context):
    ti = context.get("task_instance")
    log.error("Task %s failed in DAG run %s", getattr(ti, "task_id", "?"), context.get("run_id"))


default_args = {
    "owner": "lisa",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": notify_failure,
}


@dag(
    dag_id="cyber_etl_mlops_pipeline",
    description="Extract login events, assets and CVEs; build features; retrain and deploy the model",
    schedule="@daily",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["etl", "mlops", "cybersecurity"],
)
def cyber_etl_mlops_pipeline():

    @task
    def extract_auth_events() -> str:
        from cyber_pipeline.config import AUTH_EVENTS_CSV
        from cyber_pipeline.extract import extract_auth_events as run
        from cyber_pipeline.generate_data import main as generate
        if not AUTH_EVENTS_CSV.exists():
            generate()
        return str(run())

    @task
    def extract_assets() -> str:
        from cyber_pipeline.extract import extract_assets as run
        return str(run())

    @task(retries=3, retry_delay=timedelta(minutes=2))
    def extract_vulnerabilities() -> str:
        from cyber_pipeline.extract import extract_vulnerabilities as run
        return str(run(use_api=True))

    @task
    def transform_and_load(events: str, assets: str, vulns: str) -> int:
        # Transform and load share one task so the large DataFrame never passes through XCom.
        from cyber_pipeline.load import load_features
        from cyber_pipeline.transform import run_transform
        df = run_transform({"auth_events": events, "assets": assets, "vulnerabilities": vulns})
        return load_features(df)

    @task
    def train_and_register(rows_loaded: int) -> dict:
        from cyber_pipeline.train import main as train
        log.info("Training on %d freshly loaded rows", rows_loaded)
        meta = train()
        return {"version": meta["model_version"], "roc_auc": meta["metrics"]["roc_auc"]}

    @task
    def reload_api(model: dict) -> str:
        import requests
        try:
            r = requests.post(f"{API_URL}/reload", timeout=30)
            r.raise_for_status()
            return f"API now serving model v{r.json()['model_version']} (ROC-AUC {model['roc_auc']})"
        except requests.RequestException as exc:
            log.warning("Could not reach API at %s (%s); new model will load on next restart", API_URL, exc)
            return "api-not-reachable"

    rows = transform_and_load(extract_auth_events(), extract_assets(), extract_vulnerabilities())
    reload_api(train_and_register(rows))


cyber_etl_mlops_pipeline()

"""Checks the Airflow DAG imports cleanly and has the expected task graph.
Skipped automatically when Airflow is not installed (e.g. the lightweight CI job)."""
import importlib.util
from pathlib import Path

import pytest

DAG_FILE = Path(__file__).resolve().parents[1] / "dags" / "cyber_etl_mlops_dag.py"


def test_dag_structure():
    pytest.importorskip("airflow")
    spec = importlib.util.spec_from_file_location("cyber_dag", DAG_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dags = [v for v in vars(module).values() if type(v).__name__ == "DAG"]
    if not dags:  # TaskFlow @dag registers the DAG object when the decorated function is called
        dags = [module.cyber_etl_mlops_pipeline()]
    dag = dags[0]
    assert dag.dag_id == "cyber_etl_mlops_pipeline"
    tasks = {t.task_id: t for t in dag.tasks}
    assert set(tasks) == {"extract_auth_events", "extract_assets", "extract_vulnerabilities",
                          "transform_and_load", "train_and_register", "reload_api"}
    for extract in ("extract_auth_events", "extract_assets", "extract_vulnerabilities"):
        assert tasks[extract].downstream_task_ids == {"transform_and_load"}
    assert tasks["train_and_register"].downstream_task_ids == {"reload_api"}

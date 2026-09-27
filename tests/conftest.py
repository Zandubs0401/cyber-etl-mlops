"""Run the tests against a small, isolated copy of the pipeline in a temp directory."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

_TMP = tempfile.mkdtemp(prefix="cyber_pipeline_test_")
os.environ["PROJECT_ROOT"] = _TMP                      # must be set before config is imported
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_TMP}/mlflow.db"
os.environ["CSV_CHUNK_SIZE"] = "2000"                  # force several batches
os.environ["DASK_PARTITIONS"] = "4"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def pipeline_result():
    from cyber_pipeline.pipeline import run
    return run(use_api=False, rows=12_000)

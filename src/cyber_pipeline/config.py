"""Central configuration. Every setting can be overridden with an environment variable."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# Source 1: CSV of authentication events (large; read in batches)
AUTH_EVENTS_CSV = Path(os.getenv("AUTH_EVENTS_CSV", RAW_DIR / "auth_events.csv"))
CSV_CHUNK_SIZE = int(os.getenv("CSV_CHUNK_SIZE", "50000"))

# Source 2: operational database holding the asset inventory (any SQLAlchemy URL)
SOURCE_DB_URL = os.getenv("SOURCE_DB_URL", f"sqlite:///{RAW_DIR / 'asset_inventory.db'}")

# Source 3: NIST National Vulnerability Database REST API (CVE 2.0)
NVD_API_URL = os.getenv("NVD_API_URL", "https://services.nvd.nist.gov/rest/json/cves/2.0")
NVD_API_KEY = os.getenv("NVD_API_KEY")  # optional; raises the rate limit
NVD_KEYWORDS = os.getenv("NVD_KEYWORDS", "OpenSSH,Apache HTTP Server,nginx,Microsoft Windows,PostgreSQL").split(",")
NVD_RESULTS_PER_KEYWORD = int(os.getenv("NVD_RESULTS_PER_KEYWORD", "200"))
NVD_CACHE_FILE = SAMPLE_DIR / "nvd_cves_sample.json"

# Destination: analytics warehouse (SQLite by default, PostgreSQL via env var)
WAREHOUSE_DB_URL = os.getenv("WAREHOUSE_DB_URL", f"sqlite:///{DATA_DIR / 'warehouse.db'}")
FEATURES_TABLE = "login_features"

# Dask
DASK_PARTITIONS = int(os.getenv("DASK_PARTITIONS", "8"))

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "suspicious-login-detection")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "suspicious-login-classifier")
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODEL_DIR / "model.joblib"))

for _d in (RAW_DIR, SAMPLE_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

"""EXTRACT phase: pull data from three different kinds of sources into a staging area.

  1. CSV file (large)        -> read in fixed-size batches with pandas, written as Parquet parts
  2. Relational database     -> queried with SQLAlchemy
  3. Web API (NIST NVD)      -> paged HTTP requests with retries, rate limiting and an offline fallback

Staging everything as Parquet means the Transform phase (Dask) can read the parts in parallel.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from sqlalchemy import create_engine, text
from urllib3.util.retry import Retry

from . import config
from .logging_config import get_logger, log_duration

log = get_logger("extract")
STAGING = config.PROCESSED_DIR / "staging"

REQUIRED_EVENT_COLUMNS = {"event_id", "timestamp", "username", "asset_id", "status"}


# ----------------------------------------------------------------------------- CSV
def extract_auth_events(csv_path: Path = config.AUTH_EVENTS_CSV,
                        chunk_size: int = config.CSV_CHUNK_SIZE) -> Path:
    """Stream the CSV in batches so memory use stays flat no matter how big the file is."""
    out_dir = STAGING / "auth_events"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    total = 0
    with log_duration(log, f"extract CSV {csv_path.name} (batch={chunk_size})"):
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, dtype={"asset_id": "string"})):
            missing = REQUIRED_EVENT_COLUMNS - set(chunk.columns)
            if missing:
                raise ValueError(f"CSV batch {i} is missing required columns: {missing}")
            chunk.to_parquet(out_dir / f"part-{i:04d}.parquet", index=False)
            total += len(chunk)
            log.info("  batch %d: %d rows (running total %d)", i, len(chunk), total)
    log.info("Extracted %d auth events into %d Parquet parts", total, i + 1)
    return out_dir


# ------------------------------------------------------------------------ Database
def extract_assets(db_url: str = config.SOURCE_DB_URL) -> Path:
    out = STAGING / "assets.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    with log_duration(log, "extract assets from database"):
        engine = create_engine(db_url)
        query = text("""SELECT asset_id, hostname, software, criticality, internet_facing, owner_dept
                        FROM assets""")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        df.to_parquet(out, index=False)
    log.info("Extracted %d assets", len(df))
    return out


# ----------------------------------------------------------------------------- API
def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    if config.NVD_API_KEY:
        s.headers["apiKey"] = config.NVD_API_KEY
    return s


def _base_score(metrics: dict) -> float | None:
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if metrics.get(key):
            return metrics[key][0]["cvssData"]["baseScore"]
    return None


def _flatten(vulns: list[dict], keyword: str | None = None) -> list[dict]:
    rows = []
    for v in vulns:
        cve = v["cve"]
        rows.append({
            "cve_id": cve["id"],
            "keyword": keyword or cve.get("_keyword"),
            "published": cve.get("published"),
            "base_score": _base_score(cve.get("metrics", {})),
            "description": next((d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"), ""),
        })
    return rows


def fetch_nvd_cves(keywords: list[str] = config.NVD_KEYWORDS,
                   per_keyword: int = config.NVD_RESULTS_PER_KEYWORD) -> pd.DataFrame:
    """Page through the NVD CVE 2.0 API for each software keyword."""
    session, rows = _session(), []
    pause = 0.7 if config.NVD_API_KEY else 6.5   # NVD public rate limit: 5 requests / 30 s without a key
    for kw in keywords:
        start = 0
        while start < per_keyword:
            page = min(100, per_keyword - start)
            r = session.get(config.NVD_API_URL, timeout=30,
                            params={"keywordSearch": kw, "resultsPerPage": page, "startIndex": start})
            r.raise_for_status()
            payload = r.json()
            rows.extend(_flatten(payload.get("vulnerabilities", []), kw))
            log.info("  NVD '%s': fetched %d (total available %d)", kw, start + page, payload.get("totalResults", 0))
            start += page
            if start >= payload.get("totalResults", 0):
                break
            time.sleep(pause)
        time.sleep(pause)
    return pd.DataFrame(rows)


def extract_vulnerabilities(use_api: bool = True) -> Path:
    out = STAGING / "vulnerabilities.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    with log_duration(log, "extract vulnerabilities (NVD API)"):
        df = None
        if use_api:
            try:
                df = fetch_nvd_cves()
                log.info("Pulled %d CVEs from the live NVD API", len(df))
            except requests.RequestException as exc:
                log.warning("NVD API unavailable (%s) - falling back to cached sample", exc)
        if df is None or df.empty:
            if not config.NVD_CACHE_FILE.exists():
                import numpy as np

                from .generate_data import make_nvd_fallback
                config.NVD_CACHE_FILE.write_text(json.dumps(make_nvd_fallback(np.random.default_rng(42))))
            payload = json.loads(config.NVD_CACHE_FILE.read_text())
            df = pd.DataFrame(_flatten(payload["vulnerabilities"]))
            log.info("Loaded %d CVEs from offline sample %s", len(df), config.NVD_CACHE_FILE.name)
        df.to_parquet(out, index=False)
    return out


def run_extract(use_api: bool = True) -> dict[str, str]:
    return {
        "auth_events": str(extract_auth_events()),
        "assets": str(extract_assets()),
        "vulnerabilities": str(extract_vulnerabilities(use_api=use_api)),
    }


if __name__ == "__main__":
    print(run_extract())

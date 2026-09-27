"""Generate realistic synthetic source data for the lab.

Creates:
  * data/raw/auth_events.csv          - authentication events (structured + free-text log message)
  * data/raw/asset_inventory.db       - SQLite "operational" database with an assets table
  * data/sample/nvd_cves_sample.json  - OFFLINE FALLBACK for the NVD API (clearly synthetic IDs)

The auth events are synthetic so no real user data is involved. Data-quality problems
(missing values, duplicates, inconsistent casing) are injected on purpose so the
Transform phase has real cleaning work to do.

Usage:  python -m cyber_pipeline.generate_data --rows 300000
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from .config import AUTH_EVENTS_CSV, NVD_CACHE_FILE, SOURCE_DB_URL
from .logging_config import get_logger

log = get_logger("generate_data")

SOFTWARE = ["OpenSSH", "Apache HTTP Server", "nginx", "Microsoft Windows", "PostgreSQL"]
COUNTRIES = ["US", "US", "US", "PT", "GB", "DE", "CA", "RU", "CN", "BR", "NG", "KP"]
HIGH_RISK_COUNTRIES = {"RU", "CN", "NG", "KP"}
AUTH_METHODS = ["password", "mfa", "ssh_key", "sso"]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) Safari/17.5",
    "OpenSSH_9.6",
    "python-requests/2.32",
    "curl/8.5.0",
    "Hydra v9.5",
]


def make_assets(rng: np.random.Generator, n_assets: int = 120) -> pd.DataFrame:
    return pd.DataFrame({
        "asset_id": [f"A{i:04d}" for i in range(n_assets)],
        "hostname": [f"srv-{i:04d}.corp.local" for i in range(n_assets)],
        "software": rng.choice(SOFTWARE, n_assets),
        "criticality": rng.integers(1, 6, n_assets),          # 1 = low, 5 = crown jewel
        "internet_facing": rng.random(n_assets) < 0.3,
        "owner_dept": rng.choice(["IT", "Finance", "HR", "Engineering", "Sales"], n_assets),
    })


def make_events(rng: np.random.Generator, assets: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    n_users = 800
    users = np.array([f"user{i:04d}" for i in range(n_users)] + ["root", "admin", "test", "oracle"])
    user_p = np.r_[np.full(n_users, 0.97 / n_users), np.full(4, 0.03 / 4)]

    start = pd.Timestamp("2026-06-01")
    ts = start + pd.to_timedelta(rng.integers(0, 90 * 24 * 3600, n_rows), unit="s")
    user_idx = rng.choice(len(users), n_rows, p=user_p)
    username = users[user_idx]
    # Each account has a home country; ~12% of logins come from somewhere else
    home = rng.choice(["US", "US", "US", "PT", "GB", "DE", "CA", "BR"], len(users))
    home[-4:] = rng.choice(COUNTRIES, 4)  # shared/privileged accounts are hit from everywhere
    roaming = (rng.random(n_rows) < 0.12) | np.isin(username, ["root", "admin", "test", "oracle"])
    country = np.where(roaming, rng.choice(COUNTRIES, n_rows), home[user_idx])
    ua = rng.choice(USER_AGENTS, n_rows, p=[0.38, 0.30, 0.20, 0.05, 0.05, 0.02])
    asset_idx = rng.integers(0, len(assets), n_rows)
    asset_id = assets["asset_id"].to_numpy()[asset_idx]
    criticality = assets["criticality"].to_numpy()[asset_idx]
    internet = assets["internet_facing"].to_numpy()[asset_idx]
    method = rng.choice(AUTH_METHODS, n_rows, p=[0.45, 0.30, 0.15, 0.10])

    hour = ts.hour.to_numpy()
    off_hours = (hour < 6) | (hour >= 22)
    risky_country = np.isin(country, list(HIGH_RISK_COUNTRIES))
    scripted = np.isin(ua, ["python-requests/2.32", "curl/8.5.0", "Hydra v9.5"])
    privileged = np.isin(username, ["root", "admin", "test", "oracle"])

    # Latent risk score -> ground-truth label (with noise so the task is not trivial)
    logit = (-4.2 + 1.3 * off_hours + 1.6 * risky_country + 1.9 * scripted + 1.7 * privileged
             + 0.25 * criticality + 0.6 * internet + 0.5 * (method == "password")
             + rng.normal(0, 0.9, n_rows))
    is_suspicious = rng.random(n_rows) < 1 / (1 + np.exp(-logit))

    fail_p = np.where(is_suspicious, 0.65, 0.06)
    success = rng.random(n_rows) >= fail_p
    status = np.where(success, "SUCCESS", "FAILURE")

    bytes_sent = np.round(rng.lognormal(8, 1.2, n_rows) * np.where(is_suspicious, 3.0, 1.0))
    duration = np.round(rng.exponential(600, n_rows) * np.where(success, 1.0, 0.01), 1)

    # Attack traffic comes from a small pool of hostile IPs (so the same IP fails repeatedly);
    # normal users log in from one of a few addresses of their own.
    hostile_pool = np.array([f"185.{a}.{b}.{c}" for a, b, c in rng.integers(0, 255, (400, 3))])
    k = rng.integers(1, 4, n_rows)
    user_ip = ("10." + pd.Series(user_idx // 250).astype(str) + "." + pd.Series(user_idx % 250).astype(str)
               + "." + pd.Series(k).astype(str)).to_numpy()
    src_ip = np.where(risky_country | (is_suspicious & (rng.random(n_rows) < 0.5)),
                      rng.choice(hostile_pool, n_rows), user_ip)
    message = np.where(
        success,
        np.char.add("Accepted ", np.char.add(method.astype(str), np.char.add(" for ", username.astype(str)))),
        np.where(privileged | (rng.random(n_rows) < 0.3),
                 np.char.add("Failed password for invalid user ", username.astype(str)),
                 np.char.add("Failed password for ", username.astype(str))),
    )

    df = pd.DataFrame({
        "event_id": np.arange(n_rows),
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "username": username,
        "src_ip": src_ip,
        "src_country": country,
        "asset_id": asset_id,
        "auth_method": method,
        "status": status,
        "bytes_sent": bytes_sent,
        "session_duration_sec": duration,
        "user_agent": ua,
        "message": message,
        "is_suspicious": is_suspicious.astype(int),
    })

    # --- inject realistic data-quality problems -----------------------------------
    for col, frac in [("src_country", 0.03), ("bytes_sent", 0.02), ("session_duration_sec", 0.02),
                      ("user_agent", 0.01), ("auth_method", 0.01)]:
        df.loc[rng.random(n_rows) < frac, col] = np.nan
    lower = rng.random(n_rows) < 0.05
    df.loc[lower, "status"] = df.loc[lower, "status"].str.lower()      # inconsistent casing
    dupes = df.sample(frac=0.01, random_state=1)                        # duplicate records
    return pd.concat([df, dupes]).sample(frac=1, random_state=2).reset_index(drop=True)


def make_nvd_fallback(rng: np.random.Generator) -> dict:
    """Offline stand-in for the NVD API response (same JSON shape, obviously synthetic IDs)."""
    vulns = []
    for sw in SOFTWARE:
        for i in range(int(rng.integers(15, 60))):
            score = float(np.round(rng.uniform(2.0, 10.0), 1))
            vulns.append({"cve": {
                "id": f"CVE-SAMPLE-{sw[:3].upper()}-{i:04d}",
                "published": f"2025-{int(rng.integers(1, 13)):02d}-15T00:00:00.000",
                "descriptions": [{"lang": "en", "value": f"Synthetic sample vulnerability in {sw}."}],
                "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": score}}]},
                "_keyword": sw,
            }})
    return {"source": "OFFLINE SYNTHETIC SAMPLE - not real CVE data", "vulnerabilities": vulns}


def main(rows: int = 300_000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    assets = make_assets(rng)
    engine = create_engine(SOURCE_DB_URL)
    assets.to_sql("assets", engine, if_exists="replace", index=False)
    log.info("Wrote %d assets to %s", len(assets), SOURCE_DB_URL)

    events = make_events(rng, assets, rows)
    events.to_csv(AUTH_EVENTS_CSV, index=False)
    log.info("Wrote %d auth events to %s", len(events), AUTH_EVENTS_CSV)

    NVD_CACHE_FILE.write_text(json.dumps(make_nvd_fallback(rng)))
    log.info("Wrote NVD offline fallback to %s", NVD_CACHE_FILE)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.rows, a.seed)

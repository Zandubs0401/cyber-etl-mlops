"""TRANSFORM phase: clean, enrich, engineer features and normalize - in parallel with Dask.

The auth-event table is the large one, so it is processed as a partitioned Dask DataFrame.
The two small lookup tables (assets, vulnerability summary) are ordinary pandas DataFrames
and are broadcast-joined onto every partition.
"""
from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd

from . import config
from .logging_config import get_logger, log_duration

log = get_logger("transform")

PRIVILEGED_ACCOUNTS = ["root", "admin", "test", "oracle"]
HIGH_RISK_COUNTRIES = ["RU", "CN", "NG", "KP"]
SCRIPTED_AGENT_PATTERN = r"(?i)python-requests|curl|hydra|wget|go-http-client"
AUTH_METHODS = ["password", "mfa", "ssh_key", "sso"]

# The model's input contract. The API, the tests and training all import this list.
FEATURE_COLUMNS = [
    "hour", "is_off_hours", "is_weekend", "is_failure", "invalid_user_msg", "is_scripted_agent",
    "is_privileged_account", "is_high_risk_country", "bytes_sent_norm", "session_duration_norm",
    "user_fail_rate", "user_country_share", "ip_fail_count_log",
    "criticality", "internet_facing", "cve_count", "max_cvss", "critical_cve_count",
    "method_password", "method_mfa", "method_ssh_key", "method_sso",
]
TARGET = "is_suspicious"


def summarize_vulnerabilities(vulns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate CVEs to one row per software product."""
    v = vulns.dropna(subset=["keyword"]).copy()
    v["base_score"] = pd.to_numeric(v["base_score"], errors="coerce")
    return (v.groupby("keyword")
             .agg(cve_count=("cve_id", "nunique"),
                  max_cvss=("base_score", "max"),
                  critical_cve_count=("base_score", lambda s: int((s >= 9.0).sum())))
             .reset_index()
             .rename(columns={"keyword": "software"}))


def clean_events(ddf: dd.DataFrame) -> dd.DataFrame:
    """Handle duplicates, missing values and inconsistent formatting."""
    ddf = ddf.drop_duplicates(subset=["event_id"])
    ddf = ddf.dropna(subset=["event_id", "timestamp", "username", "asset_id"])
    ddf["status"] = ddf["status"].str.upper()
    ddf["src_country"] = ddf["src_country"].fillna("UNKNOWN")
    ddf["auth_method"] = ddf["auth_method"].fillna("unknown").str.lower()
    ddf["user_agent"] = ddf["user_agent"].fillna("unknown")
    # numeric gaps -> median (computed once across all partitions)
    medians = dd.compute(ddf["bytes_sent"].quantile(0.5), ddf["session_duration_sec"].quantile(0.5))
    ddf["bytes_sent"] = ddf["bytes_sent"].fillna(medians[0])
    ddf["session_duration_sec"] = ddf["session_duration_sec"].fillna(medians[1])
    ddf["timestamp"] = dd.to_datetime(ddf["timestamp"], utc=True)
    return ddf


def engineer_features(ddf: dd.DataFrame, assets: pd.DataFrame, vuln_summary: pd.DataFrame) -> dd.DataFrame:
    # --- time-based features
    ddf["hour"] = ddf["timestamp"].dt.hour
    ddf["is_off_hours"] = ((ddf["hour"] < 6) | (ddf["hour"] >= 22)).astype("int8")
    ddf["is_weekend"] = (ddf["timestamp"].dt.dayofweek >= 5).astype("int8")
    # --- features extracted from free text (unstructured log message / user agent)
    ddf["is_failure"] = (ddf["status"] == "FAILURE").astype("int8")
    ddf["invalid_user_msg"] = ddf["message"].str.contains("invalid user", case=False).astype("int8")
    ddf["is_scripted_agent"] = ddf["user_agent"].str.contains(SCRIPTED_AGENT_PATTERN, regex=True).astype("int8")
    ddf["is_privileged_account"] = ddf["username"].isin(PRIVILEGED_ACCOUNTS).astype("int8")
    ddf["is_high_risk_country"] = ddf["src_country"].isin(HIGH_RISK_COUNTRIES).astype("int8")
    for m in AUTH_METHODS:
        ddf[f"method_{m}"] = (ddf["auth_method"] == m).astype("int8")

    # --- behavioural aggregates (group-bys run in parallel across partitions)
    fail_int = ddf["is_failure"].astype("int64")
    ddf["_fail"] = fail_int
    user_stats = (ddf.groupby("username")
                    .agg({"_fail": "mean", "event_id": "count"})
                    .rename(columns={"_fail": "user_fail_rate", "event_id": "_user_events"})
                    .reset_index())
    # How normal is this country for this user? (share of the user's logins from that country)
    pair = (ddf.groupby(["username", "src_country"])["event_id"].count()
              .rename("_pair_events").reset_index())
    ip_stats = ddf.groupby("src_ip")["_fail"].sum().rename("ip_fail_count").reset_index()
    ddf = (ddf.merge(user_stats, on="username", how="left")
              .merge(pair, on=["username", "src_country"], how="left")
              .merge(ip_stats, on="src_ip", how="left"))
    ddf["user_country_share"] = (ddf["_pair_events"] / ddf["_user_events"]).astype("float64")
    ddf["ip_fail_count_log"] = np.log1p(ddf["ip_fail_count"].astype("float64"))

    # --- enrichment from the database and the API (small tables -> broadcast join)
    enrich = assets.merge(vuln_summary, on="software", how="left")
    enrich[["cve_count", "max_cvss", "critical_cve_count"]] = (
        enrich[["cve_count", "max_cvss", "critical_cve_count"]].fillna(0))
    enrich["internet_facing"] = enrich["internet_facing"].astype("int8")
    ddf = ddf.merge(enrich[["asset_id", "hostname", "software", "criticality", "internet_facing",
                            "cve_count", "max_cvss", "critical_cve_count"]],
                    on="asset_id", how="left", broadcast=True)
    return ddf


def normalize(ddf: dd.DataFrame) -> dd.DataFrame:
    """Log-transform heavily skewed columns, then min-max scale them to [0, 1]."""
    for src, dst in [("bytes_sent", "bytes_sent_norm"), ("session_duration_sec", "session_duration_norm")]:
        logged = np.log1p(ddf[src])
        lo, hi = dd.compute(logged.min(), logged.max())
        ddf[dst] = (logged - lo) / (hi - lo if hi > lo else 1.0)
    return ddf


def run_transform(staging: dict[str, str]) -> pd.DataFrame:
    with log_duration(log, "transform with Dask"):
        events = dd.read_parquet(staging["auth_events"]).repartition(npartitions=config.DASK_PARTITIONS)
        log.info("Loaded auth events as Dask DataFrame with %d partitions", events.npartitions)
        assets = pd.read_parquet(staging["assets"])
        vuln_summary = summarize_vulnerabilities(pd.read_parquet(staging["vulnerabilities"]))
        log.info("Vulnerability summary:\n%s", vuln_summary.to_string(index=False))

        ddf = normalize(engineer_features(clean_events(events), assets, vuln_summary))
        keep = ["event_id", "timestamp", "username", "src_ip", "asset_id", "hostname", "software",
                *FEATURE_COLUMNS, TARGET]
        result = ddf[keep].compute().sort_values("event_id").reset_index(drop=True)

        before = sum(len(pd.read_parquet(p)) for p in Path(staging["auth_events"]).glob("*.parquet"))
        log.info("Transform complete: %d raw rows -> %d clean rows (%d removed)",
                 before, len(result), before - len(result))
        if result[FEATURE_COLUMNS].isna().any().any():
            bad = result[FEATURE_COLUMNS].columns[result[FEATURE_COLUMNS].isna().any()].tolist()
            raise ValueError(f"Nulls remain in feature columns after transform: {bad}")
    return result

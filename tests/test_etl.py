import pandas as pd
import pytest

from cyber_pipeline import config
from cyber_pipeline.extract import extract_auth_events
from cyber_pipeline.load import read_features
from cyber_pipeline.transform import FEATURE_COLUMNS, TARGET, summarize_vulnerabilities


def test_pipeline_runs_and_loads_warehouse(pipeline_result):
    df = read_features()
    raw = pd.read_csv(config.AUTH_EVENTS_CSV)
    assert len(df) == raw["event_id"].nunique()          # duplicates removed, nothing else lost
    assert df["event_id"].is_unique
    assert set(FEATURE_COLUMNS + [TARGET]).issubset(df.columns)


def test_no_missing_values_in_features(pipeline_result):
    df = read_features()
    assert not df[FEATURE_COLUMNS].isna().any().any()


def test_normalized_columns_in_unit_range(pipeline_result):
    df = read_features()
    for col in ["bytes_sent_norm", "session_duration_norm", "user_country_share", "user_fail_rate"]:
        assert df[col].between(0, 1).all(), col


def test_csv_is_read_in_batches(pipeline_result):
    out = extract_auth_events(chunk_size=5_000)
    assert len(list(out.glob("part-*.parquet"))) >= 3


def test_vulnerability_summary():
    vulns = pd.DataFrame({"cve_id": ["a", "b", "c"], "keyword": ["nginx", "nginx", "OpenSSH"],
                          "base_score": [9.8, 5.0, 7.5]})
    s = summarize_vulnerabilities(vulns).set_index("software")
    assert s.loc["nginx", "cve_count"] == 2
    assert s.loc["nginx", "critical_cve_count"] == 1
    assert s.loc["OpenSSH", "max_cvss"] == pytest.approx(7.5)

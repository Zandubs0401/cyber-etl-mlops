import json

import pytest
from fastapi.testclient import TestClient

from cyber_pipeline import config
from cyber_pipeline.api import LoginFeatures, app
from cyber_pipeline.train import MIN_ROC_AUC

EXAMPLE = LoginFeatures.model_config["json_schema_extra"]["example"]


@pytest.fixture(scope="module")
def client(pipeline_result):
    with TestClient(app) as c:
        yield c


def test_model_passes_quality_gate(pipeline_result):
    meta = json.loads(config.MODEL_PATH.with_suffix(".json").read_text())
    assert meta["metrics"]["roc_auc"] >= MIN_ROC_AUC


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_flags_obvious_attack(client):
    benign = {**EXAMPLE, "hour": 14, "is_off_hours": 0, "is_failure": 0, "invalid_user_msg": 0,
              "is_scripted_agent": 0, "is_privileged_account": 0, "is_high_risk_country": 0,
              "user_country_share": 0.9, "ip_fail_count_log": 1.1, "bytes_sent_norm": 0.4,
              "session_duration_norm": 0.7, "method_password": 0, "method_mfa": 1}
    r = client.post("/predict", json={"events": [EXAMPLE, benign]})
    assert r.status_code == 200
    p = r.json()["predictions"]
    assert p[0]["suspicious_probability"] > p[1]["suspicious_probability"]
    assert p[0]["is_suspicious"] is True


def test_predict_rejects_bad_input(client):
    assert client.post("/predict", json={"events": [{**EXAMPLE, "hour": 99}]}).status_code == 422


def test_metrics_exposed(client):
    client.post("/predict", json={"events": [EXAMPLE]})
    body = client.get("/metrics").text
    assert "model_predictions_total" in body
    assert "api_request_latency_seconds" in body

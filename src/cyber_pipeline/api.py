"""FastAPI service that serves the champion model and exposes Prometheus metrics.

Run locally:   uvicorn cyber_pipeline.api:app --reload --port 8000
Docs:          http://localhost:8000/docs
Metrics:       http://localhost:8000/metrics   (scraped by Prometheus, charted in Grafana)
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info,
                               generate_latest)
from pydantic import BaseModel, Field

from . import config
from .logging_config import get_logger
from .transform import FEATURE_COLUMNS

log = get_logger("api")

# ------------------------------------------------------------------ Prometheus metrics
REQUESTS = Counter("api_requests_total", "HTTP requests", ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("model_predictions_total", "Predictions served", ["label"])
SCORES = Histogram("model_prediction_score", "Distribution of suspicious-probability scores",
                   buckets=[i / 10 for i in range(1, 11)])
ERRORS = Counter("model_prediction_errors_total", "Failed prediction requests")
MODEL_INFO = Info("model", "Currently loaded model")
MODEL_LOADED = Gauge("model_loaded", "1 if a model is loaded and ready")

STATE: dict = {}


def load_model() -> None:
    bundle = joblib.load(config.MODEL_PATH)
    STATE["model"], STATE["meta"] = bundle["model"], bundle["metadata"]
    meta = STATE["meta"]
    MODEL_INFO.info({"algorithm": meta["algorithm"], "version": str(meta["model_version"]),
                     "run_id": meta["mlflow_run_id"], "roc_auc": str(meta["metrics"]["roc_auc"])})
    MODEL_LOADED.set(1)
    log.info("Loaded model v%s (%s) from %s", meta["model_version"], meta["algorithm"], config.MODEL_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_model()
    except FileNotFoundError:
        MODEL_LOADED.set(0)
        log.error("No model found at %s - run the training step first", config.MODEL_PATH)
    yield


app = FastAPI(title="Suspicious Login Detection API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    path = request.url.path
    if path != "/metrics":
        LATENCY.labels(path).observe(time.perf_counter() - start)
        REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    return response


# ------------------------------------------------------------------ request/response schemas
class LoginFeatures(BaseModel):
    """Engineered features for one login event (the output of the ETL transform step)."""
    hour: int = Field(ge=0, le=23)
    is_off_hours: int = Field(ge=0, le=1)
    is_weekend: int = Field(ge=0, le=1)
    is_failure: int = Field(ge=0, le=1)
    invalid_user_msg: int = Field(ge=0, le=1)
    is_scripted_agent: int = Field(ge=0, le=1)
    is_privileged_account: int = Field(ge=0, le=1)
    is_high_risk_country: int = Field(ge=0, le=1)
    bytes_sent_norm: float = Field(ge=0, le=1)
    session_duration_norm: float = Field(ge=0, le=1)
    user_fail_rate: float = Field(ge=0, le=1)
    user_country_share: float = Field(ge=0, le=1)
    ip_fail_count_log: float = Field(ge=0)
    criticality: int = Field(ge=1, le=5)
    internet_facing: int = Field(ge=0, le=1)
    cve_count: float = Field(ge=0)
    max_cvss: float = Field(ge=0, le=10)
    critical_cve_count: float = Field(ge=0)
    method_password: int = Field(ge=0, le=1)
    method_mfa: int = Field(ge=0, le=1)
    method_ssh_key: int = Field(ge=0, le=1)
    method_sso: int = Field(ge=0, le=1)

    model_config = {"json_schema_extra": {"example": {
        "hour": 3, "is_off_hours": 1, "is_weekend": 0, "is_failure": 1, "invalid_user_msg": 1,
        "is_scripted_agent": 1, "is_privileged_account": 1, "is_high_risk_country": 1,
        "bytes_sent_norm": 0.71, "session_duration_norm": 0.05, "user_fail_rate": 0.34,
        "user_country_share": 0.02, "ip_fail_count_log": 4.2, "criticality": 5, "internet_facing": 1,
        "cve_count": 56, "max_cvss": 9.9, "critical_cve_count": 12,
        "method_password": 1, "method_mfa": 0, "method_ssh_key": 0, "method_sso": 0}}}


class PredictRequest(BaseModel):
    events: list[LoginFeatures] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    suspicious_probability: float
    is_suspicious: bool


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


assert set(LoginFeatures.model_fields) == set(FEATURE_COLUMNS), "API schema out of sync with ETL features"


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    ready = "model" in STATE
    return {"status": "ok" if ready else "degraded", "model_loaded": ready}


@app.get("/model-info")
def model_info():
    if "meta" not in STATE:
        raise HTTPException(503, "Model not loaded")
    return STATE["meta"]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if "model" not in STATE:
        ERRORS.inc()
        raise HTTPException(503, "Model not loaded")
    try:
        X = pd.DataFrame([e.model_dump() for e in req.events])[FEATURE_COLUMNS].astype("float64")
        proba = STATE["model"].predict_proba(X)[:, 1]
    except Exception as exc:  # pragma: no cover - defensive
        ERRORS.inc()
        log.exception("Prediction failed")
        raise HTTPException(500, f"Prediction failed: {exc}") from exc

    threshold = STATE["meta"]["threshold"]
    out = []
    for p in proba:
        label = bool(p >= threshold)
        SCORES.observe(float(p))
        PREDICTIONS.labels("suspicious" if label else "normal").inc()
        out.append(Prediction(suspicious_probability=round(float(p), 4), is_suspicious=label))
    log.info("Scored %d events (%d flagged)", len(out), sum(o.is_suspicious for o in out))
    return PredictResponse(model_version=str(STATE["meta"]["model_version"]), predictions=out)


@app.post("/reload")
def reload_model():
    """Hot-swap to a newly trained model without restarting the container."""
    load_model()
    return {"reloaded": True, "model_version": STATE["meta"]["model_version"]}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

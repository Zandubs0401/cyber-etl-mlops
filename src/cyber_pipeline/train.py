"""Train, tune and track a suspicious-login classifier with scikit-learn + MLflow.

Every hyperparameter combination is logged as a nested MLflow run (params, metrics, model).
The best run is registered in the MLflow Model Registry, given the alias "champion", and
exported to models/model.joblib for the FastAPI service. A minimum ROC-AUC quality gate
stops a bad model from being promoted (the CI pipeline relies on this).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import product

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .load import read_features
from .logging_config import get_logger, log_duration
from .transform import FEATURE_COLUMNS, TARGET

log = get_logger("train")
MIN_ROC_AUC = 0.80

SEARCH_SPACE = {
    "logistic_regression": (
        lambda p: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", **p)),
        {"C": [0.1, 1.0]},
    ),
    "random_forest": (
        lambda p: RandomForestClassifier(n_jobs=-1, random_state=42, class_weight="balanced", **p),
        {"n_estimators": [150], "max_depth": [8, 16]},
    ),
    "hist_gradient_boosting": (
        lambda p: HistGradientBoostingClassifier(random_state=42, **p),
        {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31]},
    ),
}


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Train on older events, test on the most recent ones - mimics real deployment."""
    df = df.sort_values("timestamp")
    cut = int(len(df) * (1 - test_frac))
    train, test = df.iloc[:cut], df.iloc[cut:]
    X = lambda d: d[FEATURE_COLUMNS].astype("float64")  # noqa: E731  (consistent schema for MLflow/API)
    return X(train), train[TARGET], X(test), test[TARGET]


def evaluate(model, X, y, threshold: float = 0.5) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
    }


def main(max_rows: int | None = None) -> dict:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    df = read_features()
    if max_rows:
        df = df.sample(n=min(max_rows, len(df)), random_state=42)
    X_train, y_train, X_test, y_test = time_split(df)
    log.info("Training rows: %d | test rows: %d | positive rate: %.3f",
             len(X_train), len(X_test), y_train.mean())

    best = {"roc_auc": -1.0}
    with mlflow.start_run(run_name=f"tuning-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"):
        mlflow.log_params({"train_rows": len(X_train), "test_rows": len(X_test),
                           "n_features": len(FEATURE_COLUMNS), "split": "time-based 80/20"})
        for algo, (build, grid) in SEARCH_SPACE.items():
            for values in product(*grid.values()):
                params = dict(zip(grid.keys(), values))
                with mlflow.start_run(run_name=f"{algo}-{params}", nested=True) as child, \
                        log_duration(log, f"train {algo} {params}"):
                    model = build(params).fit(X_train, y_train)
                    metrics = evaluate(model, X_test, y_test)
                    mlflow.log_param("algorithm", algo)
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(
                        model, name="model", serialization_format="cloudpickle",
                        signature=infer_signature(X_test.head(100), model.predict_proba(X_test.head(100))[:, 1]),
                        input_example=X_test.head(3),
                    )
                    log.info("  %s %s -> ROC-AUC %.4f | F1 %.4f", algo, params, metrics["roc_auc"], metrics["f1"])
                    if metrics["roc_auc"] > best["roc_auc"]:
                        best = {**metrics, "algorithm": algo, "params": params,
                                "run_id": child.info.run_id, "model": model}

        mlflow.log_metric("best_roc_auc", best["roc_auc"])
        mlflow.set_tag("best_run_id", best["run_id"])

    if best["roc_auc"] < MIN_ROC_AUC:
        raise RuntimeError(f"Quality gate failed: best ROC-AUC {best['roc_auc']:.3f} < {MIN_ROC_AUC}")

    # Register the winner and point the "champion" alias at it
    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", config.REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, "champion", mv.version)
    log.info("Registered %s version %s as 'champion'", config.REGISTERED_MODEL_NAME, mv.version)

    # Export a self-contained copy for the API container
    metadata = {
        "algorithm": best["algorithm"], "params": best["params"], "mlflow_run_id": best["run_id"],
        "registered_model": config.REGISTERED_MODEL_NAME, "model_version": mv.version,
        "features": FEATURE_COLUMNS, "threshold": 0.5,
        "metrics": {k: round(float(best[k]), 4) for k in ("roc_auc", "pr_auc", "precision", "recall", "f1")},
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump({"model": best["model"], "metadata": metadata}, config.MODEL_PATH)
    config.MODEL_PATH.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    log.info("Best model: %s %s ROC-AUC=%.4f saved to %s",
             best["algorithm"], best["params"], best["roc_auc"], config.MODEL_PATH)
    return metadata


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None, help="subsample for fast CI runs")
    main(ap.parse_args().max_rows)

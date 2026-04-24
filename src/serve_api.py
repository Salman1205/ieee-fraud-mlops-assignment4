from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

REQ_COUNT = Counter("fraud_api_requests_total", "Total API requests")
REQ_ERR = Counter("fraud_api_errors_total", "Total API errors")
LATENCY = Histogram("fraud_api_latency_seconds", "API latency")
PRED_CONF = Histogram("fraud_prediction_confidence", "Pred confidence")
RECALL_GAUGE = Gauge("fraud_recall_current", "Current fraud recall")
DRIFT_GAUGE = Gauge("fraud_drift_score", "Current drift score")

app = FastAPI(title="Fraud Inference API")


class PredictRequest(BaseModel):
    features: list[float]


def _load_model() -> Any:
    model_path = os.getenv("MODEL_PATH", "artifacts/models/best_model.pkl")
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    obj = joblib.load(p)
    return obj["model"] if isinstance(obj, dict) and "model" in obj else obj


MODEL = None


@app.on_event("startup")
def startup_event():
    global MODEL
    MODEL = _load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(req: PredictRequest):
    REQ_COUNT.inc()
    start = perf_counter()
    try:
        x = np.array(req.features).reshape(1, -1)
        score = float(MODEL.predict_proba(x)[0, 1])
        pred = int(score >= 0.5)
        PRED_CONF.observe(score)
        return {"fraud_probability": score, "prediction": pred}
    except Exception as ex:
        REQ_ERR.inc()
        raise HTTPException(status_code=500, detail=str(ex)) from ex
    finally:
        LATENCY.observe(perf_counter() - start)

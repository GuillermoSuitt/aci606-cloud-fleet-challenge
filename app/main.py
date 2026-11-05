from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import insert, select, func

from app.db import engine, telemetry, init_db

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
MODEL_VERSION = "v1"
FEATURES = ["speed", "battery", "motor_temp", "vibration", "motor_current", "obstacle_distance"]
model = None

START_TIME = time.time()
TELEMETRY_COUNT = 0
PREDICT_COUNT = 0
CLIENT_ERRORS = 0
SERVER_ERRORS = 0
PREDICT_LATENCIES: list[float] = []

app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="0.1.0-starter",
    description="Starter API for the ACI606 Cloud Fleet Challenge.",
)


@app.on_event("startup")
def on_startup() -> None:
    global model
    init_db()
    model = joblib.load(MODEL_PATH)


@app.middleware("http")
async def count_errors(request, call_next):
    global CLIENT_ERRORS, SERVER_ERRORS
    try:
        response = await call_next(request)
    except Exception:
        SERVER_ERRORS += 1
        raise
    if 400 <= response.status_code < 500:
        CLIENT_ERRORS += 1
    elif response.status_code >= 500:
        SERVER_ERRORS += 1
    return response


Status = Literal["NORMAL", "WARNING", "CRITICAL"]


class Telemetry(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    robot_id: str = Field(min_length=1, max_length=20)
    timestamp: float
    x: float
    y: float
    speed: float = Field(ge=0)
    battery: float = Field(ge=0, le=100)
    motor_temp: float
    vibration: float = Field(ge=0)
    motor_current: float = Field(ge=0)
    obstacle_distance: float = Field(ge=0)


class PredictionRequest(Telemetry):
    pass


def predecir_proba(X: pd.DataFrame) -> np.ndarray:
    """Probabilidades de cada clase, idénticas a model.predict_proba(X).

    Recorre los árboles del Random Forest directamente y promedia sus votos.
    Evita el costo fijo de joblib que scikit-learn agrega en cada llamada,
    que con 0,1 CPU en Render era la mayor parte de la latencia.
    """
    escalador, bosque = model.steps[0][1], model.steps[-1][1]
    xs = ((X[FEATURES].to_numpy(dtype=float) - escalador.mean_) / escalador.scale_).astype(np.float32)
    total = np.zeros((len(xs), len(bosque.classes_)))
    for arbol in bosque.estimators_:
        votos = arbol.tree_.predict(xs)
        total += votos / votos.sum(axis=1, keepdims=True)
    return total / len(bosque.estimators_)


def count_robots() -> int:
    stmt = select(func.count(func.distinct(telemetry.c.robot_id)))
    with engine.connect() as conn:
        return conn.execute(stmt).scalar()


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def root():
    return RedirectResponse(url="/dashboard")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "aci606-cloud-fleet",
        "version": app.version,
        "database": "postgresql",
        "model_loaded": model is not None,
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/telemetry")
def ingest_telemetry(sample: Telemetry) -> dict:
    global TELEMETRY_COUNT
    TELEMETRY_COUNT += 1
    with engine.begin() as conn:
        conn.execute(insert(telemetry).values(**sample.model_dump()))
    return {
        "accepted": True,
        "robot_id": sample.robot_id,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "note": "Stored in PostgreSQL (Neon).",
    }


@app.get("/robots")
def list_robots() -> list[dict]:
    stmt = (
        select(telemetry)
        .distinct(telemetry.c.robot_id)
        .order_by(telemetry.c.robot_id, telemetry.c.timestamp.desc())
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [dict(row) for row in rows]


@app.get("/robots/{robot_id}/history")
def robot_history(robot_id: str) -> list[dict]:
    stmt = (
        select(telemetry)
        .where(telemetry.c.robot_id == robot_id)
        .order_by(telemetry.c.timestamp)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    if not rows:
        raise HTTPException(status_code=404, detail="Robot not found")
    return [dict(row) for row in rows]


@app.post("/predict")
def predict(sample: PredictionRequest) -> dict:
    global PREDICT_COUNT
    PREDICT_COUNT += 1

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inicio = time.perf_counter()

    X = pd.DataFrame([sample.model_dump()])[FEATURES]
    probabilidades = predecir_proba(X)[0]
    idx = probabilidades.argmax()

    inference_ms = (time.perf_counter() - inicio) * 1000
    PREDICT_LATENCIES.append(inference_ms)

    return {
        "robot_id": sample.robot_id,
        "predicted_status": str(model.classes_[idx]),
        "confidence": round(float(probabilidades[idx]), 4),
        "model_version": MODEL_VERSION,
        "inference_ms": round(inference_ms, 2),
    }


@app.get("/metrics")
def metrics() -> dict:
    lat = sorted(PREDICT_LATENCIES)

    def percentil(p: float) -> float:
        if not lat:
            return 0.0
        return round(lat[min(len(lat) - 1, int(p * len(lat)))], 2)

    return {
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "requests": {
            "telemetry": TELEMETRY_COUNT,
            "predict": PREDICT_COUNT,
        },
        "errors": {
            "client_4xx": CLIENT_ERRORS,
            "server_5xx": SERVER_ERRORS,
        },
        "predict_latency_ms": {
            "avg": round(sum(lat) / len(lat), 2) if lat else 0.0,
            "p50": percentil(0.50),
            "p95": percentil(0.95),
        },
        "robots_tracked": count_robots(),
        "persistence": "POSTGRESQL",
        "model_version": MODEL_VERSION,
    }


@app.get("/dashboard/data")
def dashboard_data() -> dict:
    robots = list_robots()

    # Indicador histórico: cuántos registros tiene cada robot en la base
    stmt = select(telemetry.c.robot_id, func.count()).group_by(telemetry.c.robot_id)
    with engine.connect() as conn:
        conteos = dict(conn.execute(stmt).all())

    # Predicción de toda la flota en UNA sola llamada al modelo
    if robots and model is not None:
        X = pd.DataFrame(robots)[FEATURES]
        probs = predecir_proba(X)
        for robot, p in zip(robots, probs):
            i = p.argmax()
            robot["predicted_status"] = str(model.classes_[i])
            robot["confidence"] = round(float(p[i]), 3)

    for robot in robots:
        robot["samples"] = conteos.get(robot["robot_id"], 0)

    return {"robots": robots, "service": metrics()}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"
    return path.read_text(encoding="utf-8")

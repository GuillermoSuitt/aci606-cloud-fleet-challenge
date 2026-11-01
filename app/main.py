from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from app.db import engine, telemetry, init_db
from sqlalchemy import insert, select, func

app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="0.1.0-starter",
    description="Starter API for the ACI606 Cloud Fleet Challenge.",
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()

Status = Literal["NORMAL", "WARNING", "CRITICAL"]


class Telemetry(BaseModel):
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


REQUEST_COUNT = 0


def count_robots() -> int:
    stmt = select(func.count(func.distinct(telemetry.c.robot_id)))
    with engine.connect() as conn:
        return conn.execute(stmt).scalar()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "aci606-cloud-fleet",
        "version": app.version,
        "database": "postgresql",
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/telemetry")
def ingest_telemetry(sample: Telemetry) -> dict:
    global REQUEST_COUNT
    REQUEST_COUNT += 1
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
def predict(_: PredictionRequest) -> dict:
    # TODO E4: load the trained model and return a real prediction.
    # Required contract is documented in README.md and the student guide.
    raise HTTPException(
        status_code=501,
        detail="/predict is part of the challenge. Train and integrate your model.",
    )


@app.get("/metrics")
def metrics() -> dict:
    # TODO E4: extend with latency, errors, throughput and model metrics.
    return {
        "requests_received": REQUEST_COUNT,
        "robots_tracked": count_robots(),
        "persistence": "POSTGRESQL",
        "prediction_service": "NOT_IMPLEMENTED",
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"
    return path.read_text(encoding="utf-8")

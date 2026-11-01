import os
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float

engine = create_engine(
    os.environ["DATABASE_URL"],
    pool_pre_ping=True,
)

metadata = MetaData()

telemetry = Table(
    "telemetry",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("robot_id", String(20), nullable=False, index=True),
    Column("timestamp", Float, nullable=False),
    Column("x", Float, nullable=False),
    Column("y", Float, nullable=False),
    Column("speed" , Float, nullable=False),
    Column("battery", Float, nullable=False),
    Column("motor_temp", Float, nullable=False),
    Column("vibration", Float, nullable=False),
    Column("motor_current", Float, nullable=False),
    Column("obstacle_distance", Float, nullable=False),
)


def init_db() -> None:
    metadata.create_all(engine)
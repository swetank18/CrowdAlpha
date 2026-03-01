"""
db/models.py

SQLAlchemy ORM models for CrowdAlpha.
Uses SQLite for local dev (default), Postgres for production.
Connection string controlled by DATABASE_URL env var.
"""

from __future__ import annotations

import os
import json
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, String, Integer, Float,
    Boolean, DateTime, Text, ForeignKey, JSON
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker
from sqlalchemy.sql import func


DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./crowdalpha.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------

class SimulationModel(Base):
    __tablename__ = "simulations"

    id            = Column(String, primary_key=True)
    created_at    = Column(DateTime, server_default=func.now())
    status        = Column(String, default="running")   # running | stopped | finished
    config_json   = Column(Text, default="{}")
    n_ticks       = Column(Integer, default=0)
    final_mid     = Column(Float, nullable=True)

    agents        = relationship("AgentModel", back_populates="simulation")
    fills         = relationship("FillModel",  back_populates="simulation")


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class AgentModel(Base):
    __tablename__ = "agents"

    id            = Column(String, primary_key=True)
    simulation_id = Column(String, ForeignKey("simulations.id"))
    strategy_type = Column(String)
    config_json   = Column(Text, default="{}")
    is_user       = Column(Boolean, default=False)
    created_at    = Column(DateTime, server_default=func.now())

    # Final stats (written on sim stop)
    final_pnl     = Column(Float, nullable=True)
    final_sharpe  = Column(Float, nullable=True)
    final_inv     = Column(Integer, nullable=True)
    fill_count    = Column(Integer, default=0)

    simulation    = relationship("SimulationModel", back_populates="agents")


# ---------------------------------------------------------------------------
# Fills (trade log — append only)
# ---------------------------------------------------------------------------

class FillModel(Base):
    __tablename__ = "fills"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(String, ForeignKey("simulations.id"))
    tick          = Column(Integer)
    buy_agent_id  = Column(String)
    sell_agent_id = Column(String)
    price         = Column(Float)
    qty           = Column(Integer)
    timestamp_ns  = Column(Integer)

    simulation    = relationship("SimulationModel", back_populates="fills")


# ---------------------------------------------------------------------------
# Tick snapshots (sampled every 10 ticks for analytics)
# ---------------------------------------------------------------------------

class TickSnapshotModel(Base):
    __tablename__ = "tick_snapshots"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(String, ForeignKey("simulations.id"))
    tick          = Column(Integer)
    mid_price     = Column(Float, nullable=True)
    spread        = Column(Float, nullable=True)
    volatility    = Column(Float)
    crowding      = Column(Float)
    lfi           = Column(Float)
    regime        = Column(String)


# ---------------------------------------------------------------------------
# Crowding snapshots
# ---------------------------------------------------------------------------

class CrowdingSnapshotModel(Base):
    __tablename__ = "crowding_snapshots"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(String, ForeignKey("simulations.id"))
    tick          = Column(Integer)
    intensity     = Column(Float)
    matrix_json   = Column(Text)   # serialized crowding matrix


# ---------------------------------------------------------------------------
# Create all tables
# ---------------------------------------------------------------------------

def init_db() -> None:
    Base.metadata.create_all(bind=engine)

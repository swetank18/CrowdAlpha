"""
db/queries.py

Typed query helpers for common access patterns.
"""
from __future__ import annotations

from typing import List, Optional
from sqlalchemy.orm import Session
from .models import SimulationModel, AgentModel, FillModel, TickSnapshotModel


def get_simulation(db: Session, sim_id: str) -> Optional[SimulationModel]:
    return db.query(SimulationModel).filter(SimulationModel.id == sim_id).first()


def get_top_agents(db: Session, sim_id: str, n: int = 20) -> List[AgentModel]:
    return (
        db.query(AgentModel)
        .filter(AgentModel.simulation_id == sim_id)
        .order_by(AgentModel.final_sharpe.desc().nullslast())
        .limit(n)
        .all()
    )


def get_tick_history(
    db: Session, sim_id: str, start: int = 0, end: Optional[int] = None
) -> List[TickSnapshotModel]:
    q = db.query(TickSnapshotModel).filter(
        TickSnapshotModel.simulation_id == sim_id,
        TickSnapshotModel.tick >= start,
    )
    if end:
        q = q.filter(TickSnapshotModel.tick <= end)
    return q.order_by(TickSnapshotModel.tick).all()


def get_fills_by_agent(
    db: Session, sim_id: str, agent_id: str, limit: int = 100
) -> List[FillModel]:
    return (
        db.query(FillModel)
        .filter(
            FillModel.simulation_id == sim_id,
            (FillModel.buy_agent_id == agent_id)
            | (FillModel.sell_agent_id == agent_id)
        )
        .order_by(FillModel.tick.desc())
        .limit(limit)
        .all()
    )

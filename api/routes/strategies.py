"""
api/routes/strategies.py

REST endpoints for strategy/agent management.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter()


class DeployRequest(BaseModel):
    strategy_type: str
    agent_id:      Optional[str] = None
    config:        Dict[str, Any] = {}


class UserStrategyRequest(BaseModel):
    strategy_name: str
    code:          str
    deploy:        bool = False
    agent_id:      Optional[str] = None
    config:        Dict[str, Any] = {}


def _get_sim(request: Request):
    sim = getattr(request.app.state, "sim", None)
    if sim is None:
        raise HTTPException(503, "Simulation not ready")
    return sim


@router.get("")
def list_strategies(request: Request):
    """List all registered strategy types and live agents."""
    sim = _get_sim(request)
    return {
        "strategy_types": sim.registry.list_strategies(),
        "agents": [
            {
                "agent_id":      a.agent_id,
                "strategy_type": a.strategy_type,
                "inventory":     a.inventory,
                "pnl":           round(a.pnl, 2),
            }
            for a in sim.agents
        ]
    }


@router.post("", status_code=201)
def deploy_strategy(request: Request, body: DeployRequest):
    """Deploy a new agent into the running simulation."""
    sim = _get_sim(request)
    import uuid
    agent_id = body.agent_id or f"user_{str(uuid.uuid4())[:8]}"
    try:
        agent = sim.registry.create(body.strategy_type, agent_id, body.config)
        sim.add_agent(agent)
        return {"agent_id": agent_id, "strategy_type": body.strategy_type}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/user", status_code=201)
def register_user_strategy(request: Request, body: UserStrategyRequest):
    """Register a user-submitted Python strategy class."""
    sim = _get_sim(request)
    try:
        sim.registry.register_user_strategy(body.strategy_name, body.code)
        if not body.deploy:
            return {"registered": body.strategy_name, "deployed": False}

        import uuid

        agent_id = body.agent_id or f"user_{str(uuid.uuid4())[:8]}"
        agent = sim.registry.create(body.strategy_name, agent_id, body.config)
        sim.add_agent(agent)
        return {
            "registered": body.strategy_name,
            "deployed": True,
            "agent_id": agent_id,
            "strategy_type": body.strategy_name,
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/{agent_id}/stats")
def agent_stats(request: Request, agent_id: str):
    """Per-agent diagnostics."""
    sim = _get_sim(request)
    agent = next((a for a in sim.agents if a.agent_id == agent_id), None)
    if agent is None:
        raise HTTPException(404, f"Agent '{agent_id}' not found")
    diag = sim.diagnostics.compute(agent)
    return diag.to_dict()


@router.get("/leaderboard")
def leaderboard(request: Request):
    """All agents sorted by Sharpe ratio."""
    sim = _get_sim(request)
    return {"leaderboard": sim.diagnostics.snapshot_all(sim.agents)}

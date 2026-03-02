"""
api/routes/strategies.py

REST endpoints for strategy/agent management.
"""

from __future__ import annotations

import json
import math
import uuid
from typing import Dict, Any, Optional, Literal

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

from db.models import SessionLocal, StrategySubmissionModel

router = APIRouter()


class DeployRequest(BaseModel):
    strategy_type: str
    agent_id: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class UserStrategyRequest(BaseModel):
    strategy_name: str
    code: str
    deploy: bool = False
    agent_id: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class StrategyConfigSubmissionRequest(BaseModel):
    submitter_name: str = Field(..., min_length=2, max_length=80)
    contact: str = Field(..., min_length=3, max_length=120)
    mode: Literal["beginner", "intermediate", "advanced"] = "beginner"
    template_strategy: str = Field(..., min_length=3, max_length=64)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    clone_source_agent_id: Optional[str] = Field(default=None, max_length=64)
    visibility: Literal["public", "private"] = "public"
    notes: Optional[str] = Field(default=None, max_length=1000)


class SubmissionApproveRequest(BaseModel):
    deploy: bool = True
    agent_id: Optional[str] = None


def _get_sim(request: Request):
    sim = getattr(request.app.state, "sim", None)
    if sim is None:
        raise HTTPException(503, "Simulation not ready")
    return sim


def _clean_parameters(raw: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in list(raw.items())[:32]:
        k = str(key).strip()
        if not k or len(k) > 64:
            continue
        if isinstance(value, bool):
            out[k] = value
            continue
        if isinstance(value, int):
            out[k] = value
            continue
        if isinstance(value, float):
            if math.isfinite(value):
                out[k] = value
            continue
        if isinstance(value, str):
            out[k] = value[:200]
    return out


def _submission_to_dict(row: StrategySubmissionModel) -> Dict[str, Any]:
    try:
        parameters = json.loads(row.parameters_json or "{}")
    except json.JSONDecodeError:
        parameters = {}
    return {
        "submission_id": row.id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "submitter_name": row.submitter_name,
        "contact": row.contact,
        "mode": row.mode,
        "template_strategy": row.template_strategy,
        "parameters": parameters,
        "clone_source_agent_id": row.clone_source_agent_id,
        "visibility": row.visibility,
        "notes": row.notes,
        "status": row.status,
        "deployed_agent_id": row.deployed_agent_id,
    }


@router.get("")
def list_strategies(request: Request):
    """List all registered strategy types and live agents."""
    sim = _get_sim(request)
    return {
        "strategy_types": sim.registry.list_strategies(),
        "agents": [
            {
                "agent_id": a.agent_id,
                "strategy_type": a.strategy_type,
                "inventory": a.inventory,
                "pnl": round(a.pnl, 2),
            }
            for a in sim.agents
        ],
    }


@router.post("", status_code=201)
def deploy_strategy(request: Request, body: DeployRequest):
    """Deploy a new agent into the running simulation."""
    sim = _get_sim(request)
    agent_id = body.agent_id or f"user_{str(uuid.uuid4())[:8]}"
    try:
        agent = sim.registry.create(body.strategy_type, agent_id, body.config)
        sim.add_agent(agent)
        return {"agent_id": agent_id, "strategy_type": body.strategy_type}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/submit-config", status_code=201)
def submit_strategy_config(request: Request, body: StrategyConfigSubmissionRequest):
    """Persist a strategy config submission for manual review or admin deployment."""
    sim = _get_sim(request)
    strategy_types = set(sim.registry.list_strategies())
    if body.template_strategy not in strategy_types:
        raise HTTPException(400, f"Unknown template strategy '{body.template_strategy}'")

    row = StrategySubmissionModel(
        id=f"sub_{str(uuid.uuid4())[:10]}",
        submitter_name=body.submitter_name.strip(),
        contact=body.contact.strip(),
        mode=body.mode,
        template_strategy=body.template_strategy,
        parameters_json=json.dumps(_clean_parameters(body.parameters)),
        clone_source_agent_id=body.clone_source_agent_id,
        visibility=body.visibility,
        notes=body.notes,
        status="pending",
    )

    db = SessionLocal()
    try:
        db.add(row)
        db.commit()
        db.refresh(row)
        return {
            "submission_id": row.id,
            "status": row.status,
            "message": "Submission received. It will be reviewed before activation.",
        }
    finally:
        db.close()


@router.get("/submissions")
def list_submissions(status: Optional[str] = None, limit: int = 100):
    """List recent strategy submissions for admin review."""
    db = SessionLocal()
    try:
        q = db.query(StrategySubmissionModel)
        if status:
            q = q.filter(StrategySubmissionModel.status == status)
        rows = (
            q.order_by(StrategySubmissionModel.created_at.desc())
            .limit(max(1, min(limit, 200)))
            .all()
        )
        return {"items": [_submission_to_dict(row) for row in rows]}
    finally:
        db.close()


@router.post("/submissions/{submission_id}/approve")
def approve_submission(request: Request, submission_id: str, body: SubmissionApproveRequest):
    """Approve a pending submission and optionally deploy it into the live simulation."""
    sim = _get_sim(request)
    db = SessionLocal()
    try:
        row = db.query(StrategySubmissionModel).filter(StrategySubmissionModel.id == submission_id).first()
        if row is None:
            raise HTTPException(404, f"Submission '{submission_id}' not found")

        if row.status not in {"pending", "approved"}:
            raise HTTPException(400, f"Submission in status '{row.status}' cannot be approved")

        row.status = "approved"
        deployed_agent_id: Optional[str] = None
        if body.deploy:
            try:
                params = json.loads(row.parameters_json or "{}")
            except json.JSONDecodeError:
                params = {}
            deployed_agent_id = body.agent_id or f"usr_{str(uuid.uuid4())[:8]}"
            agent = sim.registry.create(row.template_strategy, deployed_agent_id, params)
            sim.add_agent(agent)
            row.status = "deployed"
            row.deployed_agent_id = deployed_agent_id

        db.add(row)
        db.commit()
        db.refresh(row)
        return {
            "submission": _submission_to_dict(row),
            "deployed": bool(deployed_agent_id),
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        db.close()


@router.post("/user", status_code=201)
def register_user_strategy(request: Request, body: UserStrategyRequest):
    """Register a user-submitted Python strategy class."""
    sim = _get_sim(request)
    try:
        sim.registry.register_user_strategy(body.strategy_name, body.code)
        if not body.deploy:
            return {"registered": body.strategy_name, "deployed": False}

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

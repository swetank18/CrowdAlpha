"""
api/routes/analytics.py

REST endpoints for crowding, alpha decay, regime, and fragility analytics.
"""
from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException

router = APIRouter()


def _get_sim(request: Request):
    sim = getattr(request.app.state, "sim", None)
    if sim is None:
        raise HTTPException(503, "Simulation not ready")
    return sim


@router.get("/crowding")
def get_crowding(request: Request):
    """Latest crowding matrix + intensity scalar + top crowded pairs."""
    sim = _get_sim(request)
    return sim.crowding_matrix.snapshot_for_api()


@router.get("/factor-space")
def get_factor_space(request: Request):
    """Factor vectors + 2D PCA projection for all agents."""
    sim = _get_sim(request)
    return sim.factor_space.snapshot_for_api()


@router.get("/decay")
def get_alpha_decay(request: Request):
    """Alpha decay model params + crowding intensity history."""
    sim = _get_sim(request)
    return sim.alpha_decay.snapshot_for_api()


@router.get("/regime")
def get_regime(request: Request):
    """Current market regime + recent regime history."""
    sim = _get_sim(request)
    return sim.regime_detector.snapshot_for_api()


@router.get("/fragility")
def get_fragility(request: Request):
    """Current LFI score, alert level, and history."""
    sim = _get_sim(request)
    return sim.fragility.snapshot_for_api()

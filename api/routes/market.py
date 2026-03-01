"""
api/routes/market.py

REST endpoints for live market data.
"""
from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException

router = APIRouter()


def _get_sim(request: Request):
    sim = getattr(request.app.state, "sim", None)
    if sim is None:
        raise HTTPException(503, "Simulation not ready")
    return sim


@router.get("/book")
def get_order_book(request: Request, levels: int = 10):
    """Return current order book depth snapshot."""
    sim = _get_sim(request)
    return sim.book.depth_snapshot(n_levels=levels)


@router.get("/trades")
def get_recent_trades(request: Request, limit: int = 200, since_tick: int = 0):
    """Return historical fills from the in-memory trade tape."""
    sim = _get_sim(request)
    limit = max(1, min(limit, 5000))
    tape = sim.trade_history
    if since_tick > 0:
        tape = [t for t in tape if int(t.get("tick", 0)) >= since_tick]
    return {"fills": tape[-limit:]}


@router.get("/metrics")
def get_metrics(request: Request):
    """Return key market metrics from the latest tick."""
    sim = _get_sim(request)
    state = sim.current_state
    if state is None:
        return {}
    return {
        "tick":       state.tick,
        "mid_price":  state.mid_price,
        "spread":     state.spread,
        "vwap":       state.vwap,
        "volatility": state.volatility,
        "regime":     state.regime,
        "lfi":        state.lfi,
        "lfi_alert":  state.lfi_alert,
        "crowding":   state.crowding,
    }


@router.get("/snapshot")
def get_full_snapshot(request: Request):
    """Return the full latest TickState, useful for client re-hydration."""
    sim = _get_sim(request)
    state = sim.current_state
    if state is None:
        raise HTTPException(404, "No tick data yet")
    return state.to_ws_event()

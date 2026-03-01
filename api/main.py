"""
api/main.py

FastAPI application entry point.

Lifespan:
  - startup: creates default simulation, starts async tick loop
  - shutdown: stops simulation loop

Endpoints are split across three routers:
  - /market   — live market data (book, trades, metrics)
  - /strategies — agent management
  - /analytics  — crowding, decay, regime, fragility
  - /ws/market  — WebSocket live feed
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from engine.simulation import Simulation, SimulationConfig
from api.routes import market, strategies, analytics
from api import websocket as ws_module


# ---------------------------------------------------------------------------
# Shared simulation instance (injected into routes via app.state)
# ---------------------------------------------------------------------------

_sim: Optional[Simulation] = None
_sim_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start simulation on startup, clean up on shutdown."""
    global _sim, _sim_task

    # Create default simulation
    config = SimulationConfig(
        n_momentum       = 3,
        n_mean_reversion = 3,
        n_market_makers  = 2,
        n_rl             = 0,   # RL agent disabled by default (no model file)
        tick_delay_ms    = 100.0,
        seed             = 42,
    )
    _sim = Simulation(config)
    app.state.sim = _sim

    # Wire WebSocket broadcaster
    async def broadcast(event: dict) -> None:
        await ws_module.manager.broadcast_json(event)

    # Start simulation loop in background
    _sim_task = asyncio.create_task(_sim.run_async(broadcast_fn=broadcast))

    yield

    # Shutdown
    if _sim:
        _sim.stop()
    if _sim_task:
        _sim_task.cancel()
        try:
            await _sim_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "CrowdAlpha API",
    description = "Multi-agent LOB simulation engine with crowding analytics",
    version     = "0.1.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Routers
app.include_router(market.router,     prefix="/market",     tags=["market"])
app.include_router(strategies.router, prefix="/strategies", tags=["strategies"])
app.include_router(analytics.router,  prefix="/analytics",  tags=["analytics"])
app.include_router(ws_module.router,  tags=["websocket"])


@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "service": "CrowdAlpha API"}


@app.get("/status", tags=["health"])
def status():
    sim = app.state.sim if hasattr(app.state, "sim") else None
    state = sim.current_state if sim else None
    return {
        "running":   sim.is_running() if sim else False,
        "tick":      state.tick if state else 0,
        "mid_price": state.mid_price if state else None,
        "regime":    state.regime if state else None,
    }

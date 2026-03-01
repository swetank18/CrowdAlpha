"""
api/websocket.py

WebSocket endpoint — streams live simulation events to connected clients.

Events:
    TICK           — every tick (100ms default): mid-price, spread, fills, agent stats
    FRAGILITY_WARN — when LFI crosses a threshold
    REGIME_CHANGE  — when market regime changes

Client connection: ws://localhost:8000/ws/market
"""

from __future__ import annotations

import asyncio
import json
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from engine.events import EventType, make_event

router = APIRouter()


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:

    def __init__(self) -> None:
        self._active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._active.discard(ws)

    async def broadcast_json(self, data: dict) -> None:
        if not self._active:
            return
        message = json.dumps(data)
        dead = set()
        for ws in list(self._active):
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._active.discard(ws)

    @property
    def n_connected(self) -> int:
        return len(self._active)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------

@router.get("/ws/schema")
def ws_schema():
    """WebSocket event contract used by the frontend."""
    base = {
        "schema_version": "int",
        "type": "string",
        "timestamp": "int(ns)",
        "payload": "object",
    }
    return {
        "base_message": base,
        "event_types": [e.value for e in EventType],
        "payloads": {
            "ORDER_SUBMITTED": "tick, order_id, agent_id, side, order_type, price, qty, cancel_target",
            "ORDER_FILLED": "tick, buy_order_id, sell_order_id, buy_agent_id, sell_agent_id, price, qty",
            "PRICE_UPDATED": "tick, prev_mid_price, mid_price, prev_spread, spread",
            "AGENT_STATE_CHANGED": "tick, agents[]",
            "CROWDING_MATRIX_UPDATED": "tick, crowding_intensity, agent_ids, matrix, agent_crowding_intensity, top_crowded_pairs",
            "REGIME_CHANGED": "tick, prev_regime, regime",
            "TICK": "tick, mid_price, spread, vwap, volatility, regime, lfi, lfi_near_depth_ratio, lfi_alert, crowding, order_book, recent_fills, agent_stats",
            "HEARTBEAT": "message|n_connected",
        },
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/market")
async def market_ws(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Keep alive — the simulation pushes data, client can send pings
        while True:
            try:
                # Wait for client message (ping / config change)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo pings back
                if data == "ping":
                    await websocket.send_text(json.dumps(make_event(EventType.HEARTBEAT, {"message": "pong"})))
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(
                    json.dumps(
                        make_event(
                            EventType.HEARTBEAT,
                            {"n_connected": manager.n_connected},
                        )
                    )
                )
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)

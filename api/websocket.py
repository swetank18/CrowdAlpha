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
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({"type": "HEARTBEAT"}))
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)

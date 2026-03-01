"""
engine/events.py

Shared event schema for WebSocket streaming.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, Optional


SCHEMA_VERSION = 1


class EventType(str, Enum):
    HEARTBEAT = "HEARTBEAT"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    PRICE_UPDATED = "PRICE_UPDATED"
    AGENT_STATE_CHANGED = "AGENT_STATE_CHANGED"
    CROWDING_MATRIX_UPDATED = "CROWDING_MATRIX_UPDATED"
    REGIME_CHANGED = "REGIME_CHANGED"
    TICK = "TICK"


def make_event(
    event_type: EventType | str,
    payload: Dict[str, Any],
    timestamp: Optional[int] = None,
) -> Dict[str, Any]:
    event_type_value = event_type.value if isinstance(event_type, EventType) else str(event_type)
    return {
        "schema_version": SCHEMA_VERSION,
        "type": event_type_value,
        "timestamp": int(timestamp if timestamp is not None else time.time_ns()),
        "payload": payload,
    }

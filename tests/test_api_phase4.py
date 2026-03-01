"""
tests/test_api_phase4.py

Phase 4 API/WebSocket event schema tests.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.events import EventType, make_event
from engine.simulation import Simulation, SimulationConfig


def test_make_event_has_required_fields():
    evt = make_event(EventType.TICK, {"tick": 1}, timestamp=123)
    assert evt["schema_version"] == 1
    assert evt["type"] == "TICK"
    assert evt["timestamp"] == 123
    assert evt["payload"]["tick"] == 1


def test_simulation_emits_typed_events_and_trade_history_has_tick():
    sim = Simulation(
        SimulationConfig(
            n_momentum=0,
            n_mean_reversion=0,
            n_market_makers=2,
            n_rl=0,
            seed=42,
            tick_delay_ms=0.0,
        )
    )

    sim.run(n_ticks=3, print_prices=False)
    events = sim.latest_events
    assert events, "latest_events should not be empty after run"

    for event in events:
        assert "schema_version" in event
        assert "type" in event
        assert "timestamp" in event
        assert "payload" in event

    types = {event["type"] for event in events}
    assert "ORDER_SUBMITTED" in types
    assert "CROWDING_MATRIX_UPDATED" in types
    assert "TICK" in types

    # Trades can be empty in very short runs, but schema must be stable when present.
    if sim.trade_history:
        assert "tick" in sim.trade_history[-1]

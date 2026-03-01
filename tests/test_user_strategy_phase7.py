"""
tests/test_user_strategy_phase7.py

Phase 7 tests:
- sandboxed user strategy registration and deployment
- safety checks for unsafe code
- timeout handling for pathological strategies
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from engine.agents.base_agent import MarketState
from engine.agents.registry import AgentRegistry
from engine.core.order import OrderType, Side


def _state(tick: int = 1, mid: float = 100.0, inv: int = 0) -> MarketState:
    best_bid = mid - 0.05
    best_ask = mid + 0.05
    return MarketState(
        tick=tick,
        mid_price=mid,
        best_bid=best_bid,
        best_ask=best_ask,
        spread=best_ask - best_bid,
        bid_levels=((best_bid, 10), (best_bid - 0.01, 8)),
        ask_levels=((best_ask, 10), (best_ask + 0.01, 8)),
        recent_trades=(),
        inventory=inv,
        cash=100_000.0,
        pnl=0.0,
        volatility=0.001,
        vwap=mid,
        crowding_intensity=0.3,
        crowding_side_pressure=0.1,
        impact_buy_mult=1.1,
        impact_sell_mult=1.0,
    )


def test_register_and_deploy_sandboxed_user_strategy():
    code = """
class UserMomentum(BaseAgent):
    def __init__(self, agent_id, strategy_type="user_momentum", initial_cash=100000.0, qty=3):
        super().__init__(agent_id=agent_id, strategy_type=strategy_type, initial_cash=initial_cash)
        self.qty = qty

    def on_tick(self, state):
        if state.best_ask is None:
            return []
        return [Order(
            agent_id=self.agent_id,
            side=Side.BID,
            order_type=OrderType.LIMIT,
            price=state.best_ask,
            qty=self.qty,
        )]

    def factor_vector(self):
        return np.array([0.6, 0.1, 0.8, 0.2, 0.4, 0.7], dtype=float)
"""
    registry = AgentRegistry()
    registry.register_user_strategy("user_momentum", code)
    agent = registry.create(
        "user_momentum",
        "user_a",
        {"qty": 4, "timeout_ms": 250},
    )
    out = agent.generate_orders(_state())
    assert out
    assert out[0].side == Side.BID
    assert out[0].order_type == OrderType.LIMIT
    assert out[0].qty == 4
    fv = agent.factor_vector()
    assert np.asarray(fv).size > 0
    agent.close()


def test_register_rejects_unsafe_imports():
    registry = AgentRegistry()
    bad = """
import os
class Bad(BaseAgent):
    def on_tick(self, state):
        return []
    def factor_vector(self):
        return np.zeros(6)
"""
    with pytest.raises(ValueError):
        registry.register_user_strategy("bad_import", bad)


def test_timeout_user_strategy_goes_silent():
    registry = AgentRegistry()
    slow = """
class Slow(BaseAgent):
    def on_tick(self, state):
        while True:
            pass
    def factor_vector(self):
        return np.zeros(6)
"""
    registry.register_user_strategy("slow_agent", slow)
    agent = registry.create("slow_agent", "slow_1", {"timeout_ms": 40})
    out = agent.generate_orders(_state())
    assert out == []
    # after timeout process is closed and remains silent
    out2 = agent.generate_orders(_state(tick=2))
    assert out2 == []

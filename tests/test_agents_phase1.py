"""
tests/test_agents_phase1.py

Phase 1 tests:
- BaseAgent interface enforcement
- Basic order generation for momentum, mean-reversion, and market-maker agents
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from engine.agents.base_agent import BaseAgent, MarketState
from engine.agents.market_maker import MarketMakerAgent
from engine.agents.mean_reversion import MeanReversionAgent
from engine.agents.momentum import MomentumAgent
from engine.core.order import Order, OrderType, Side


def make_state(
    tick: int,
    mid: float,
    best_bid: float | None = None,
    best_ask: float | None = None,
    inv: int = 0,
) -> MarketState:
    return MarketState(
        tick=tick,
        mid_price=mid,
        best_bid=best_bid,
        best_ask=best_ask,
        spread=(best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None,
        bid_levels=((best_bid, 10),) if best_bid is not None else (),
        ask_levels=((best_ask, 10),) if best_ask is not None else (),
        recent_trades=(),
        inventory=inv,
        cash=100_000.0,
        pnl=0.0,
        volatility=0.001,
        vwap=mid,
    )


class _BadTypeAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("bad_type", "bad_type")

    def on_tick(self, state: MarketState):
        return [123]  # invalid output

    def factor_vector(self):
        return np.zeros(5)


class _WrongOwnerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("owner_a", "wrong_owner")

    def on_tick(self, state: MarketState):
        return [
            Order(
                agent_id="owner_b",
                side=Side.BID,
                order_type=OrderType.LIMIT,
                price=100.0,
                qty=1,
            )
        ]

    def factor_vector(self):
        return np.zeros(5)


def test_base_agent_validates_output_types():
    agent = _BadTypeAgent()
    state = make_state(tick=1, mid=100.0)
    with pytest.raises(TypeError):
        agent.generate_orders(state)


def test_base_agent_validates_order_ownership():
    agent = _WrongOwnerAgent()
    state = make_state(tick=1, mid=100.0)
    with pytest.raises(ValueError):
        agent.generate_orders(state)


def test_momentum_generates_buy_order_in_uptrend():
    agent = MomentumAgent(
        "mom_test",
        fast_window=2,
        slow_window=4,
        entry_threshold=0.0001,
        market_threshold=0.0002,
    )
    out = []
    prices = [100.0, 100.1, 100.2, 100.35, 100.55]
    for i, price in enumerate(prices, start=1):
        out = agent.generate_orders(
            make_state(tick=i, mid=price, best_bid=price - 0.05, best_ask=price + 0.05)
        )
    assert out
    assert out[0].side == Side.BID
    assert out[0].order_type in (OrderType.LIMIT, OrderType.MARKET)


def test_mean_reversion_generates_buy_order_on_oversold():
    agent = MeanReversionAgent(
        "rev_test",
        window=5,
        threshold=1.0,
        market_z=1.8,
    )
    out = []
    prices = [100.0, 100.2, 100.1, 100.0, 98.4]
    for i, price in enumerate(prices, start=1):
        out = agent.generate_orders(
            make_state(tick=i, mid=price, best_bid=price - 0.05, best_ask=price + 0.05)
        )
    assert out
    assert out[0].side == Side.BID
    assert out[0].order_type in (OrderType.LIMIT, OrderType.MARKET)


def test_market_maker_posts_two_sided_quotes():
    agent = MarketMakerAgent("mm_test")
    out = agent.generate_orders(make_state(tick=1, mid=100.0, best_bid=99.9, best_ask=100.1))
    live_quotes = [o for o in out if o.order_type == OrderType.LIMIT]
    assert len(live_quotes) == 2
    assert {o.side for o in live_quotes} == {Side.BID, Side.ASK}


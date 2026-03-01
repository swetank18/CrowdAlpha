"""
tests/test_rl_agent_phase6.py

Phase 6 RL agent tests:
- Action interface validity
- Reward/transition accumulation in-sim
- Simulation compatibility with RL population enabled
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.agents.base_agent import MarketState
from engine.agents.rl_agent import RLAgent
from engine.simulation import Simulation, SimulationConfig


def _state(tick: int, mid: float, pnl: float, inventory: int = 0) -> MarketState:
    best_bid = mid - 0.05
    best_ask = mid + 0.05
    return MarketState(
        tick=tick,
        mid_price=mid,
        best_bid=best_bid,
        best_ask=best_ask,
        spread=best_ask - best_bid,
        bid_levels=((best_bid, 20), (best_bid - 0.01, 18), (best_bid - 0.02, 15)),
        ask_levels=((best_ask, 20), (best_ask + 0.01, 18), (best_ask + 0.02, 15)),
        recent_trades=(
            {"tick": tick, "price": mid + 0.01, "qty": 2},
            {"tick": tick, "price": mid - 0.01, "qty": 1},
        ),
        inventory=inventory,
        cash=100_000.0 + pnl,
        pnl=pnl,
        volatility=0.0015,
        vwap=mid - 0.02,
        crowding_intensity=0.4,
        crowding_side_pressure=0.1,
        impact_buy_mult=1.2,
        impact_sell_mult=1.1,
    )


def test_rl_agent_generates_valid_orders_and_builds_transitions():
    agent = RLAgent(
        "rl_test",
        enable_online_learning=False,  # test should pass with no SB3 install
        max_inv=40,
    )

    emitted = 0
    for t in range(1, 40):
        mid = 100.0 + 0.03 * t
        pnl = 5.0 * t
        inv = 5 if t % 2 == 0 else -4
        agent.sync_position(100_000.0 + pnl, inv, pnl)
        orders = agent.generate_orders(_state(tick=t, mid=mid, pnl=pnl, inventory=inv))
        emitted += len(orders)
        for order in orders:
            assert order.agent_id == "rl_test"

    # Transition/reward histories are populated after the first tick.
    assert len(agent._transitions) > 0
    assert len(agent._reward_history) > 0
    assert emitted > 0


def test_simulation_runs_with_rl_enabled_population():
    sim = Simulation(
        SimulationConfig(
            n_momentum=1,
            n_mean_reversion=1,
            n_market_makers=1,
            n_rl=1,
            seed=7,
            tick_delay_ms=0.0,
        )
    )

    states = sim.run(n_ticks=6, print_prices=False)
    assert len(states) == 6
    assert any(a.strategy_type == "rl_ppo" for a in sim.agents)

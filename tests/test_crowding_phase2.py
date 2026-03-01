"""
tests/test_crowding_phase2.py

Phase 2 crowding engine tests.
"""

import math
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.core.order import Fill, Order, OrderType, Side
from engine.crowding.alpha_decay import AlphaDecay
from engine.crowding.crowding_matrix import CrowdingMatrix
from engine.crowding.factor_space import FACTOR_NAMES, N_FACTORS, FactorSpace


@dataclass
class _AgentStub:
    agent_id: str


class _SharpeStub:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.current_sharpe = 1.0

    def rolling_sharpe(self):
        return self.current_sharpe


def test_factor_space_builds_six_dim_behavior_vectors():
    fs = FactorSpace(window=12)
    agents = [_AgentStub("a1"), _AgentStub("a2")]

    for tick in range(1, 15):
        mid = 100.0 + 0.05 * math.sin(tick / 2)
        orders = [
            Order("a1", Side.BID, OrderType.MARKET, price=mid, qty=5),
            Order("a2", Side.ASK, OrderType.LIMIT, price=mid + 0.2, qty=4),
        ]
        fills = [
            Fill(
                buy_order_id="bo",
                sell_order_id="so",
                buy_agent_id="a1",
                sell_agent_id="a2",
                price=mid,
                qty=2,
            )
        ]
        F = fs.update(
            agents=agents,
            tick=tick,
            mid_price=mid,
            best_bid=mid - 0.05,
            best_ask=mid + 0.05,
            orders=orders,
            fills=fills,
        )

    assert F.shape == (2, N_FACTORS)
    assert len(FACTOR_NAMES) == N_FACTORS

    # a1 is persistent buyer with more aggressive orders than a2.
    assert F[0, 2] > 0.2  # directional bias
    assert F[1, 2] < -0.2
    assert F[0, 4] > 0.2  # inventory skew
    assert F[1, 4] < -0.2
    assert F[0, 5] > F[1, 5]  # order aggressiveness


def test_crowding_intensity_is_activity_weighted():
    cm = CrowdingMatrix()
    F = np.array(
        [
            [1.0, 0.2, 0.8, 0.3, 0.7, 0.9],
            [1.0, 0.2, 0.8, 0.3, 0.7, 0.9],
            [-1.0, 0.0, -0.8, 0.1, -0.7, 0.2],
        ]
    )
    ids = ["a", "b", "c"]

    unweighted = cm.update(F, ids)
    weighted = cm.update(F, ids, activity_weights=[1.0, 1.0, 0.1])

    assert weighted > unweighted
    pairs = cm.top_pairs(1)
    assert pairs[0]["agent_a"] == "a"
    assert pairs[0]["agent_b"] == "b"


def test_alpha_decay_fits_half_life_and_updates_side_multipliers():
    ad = AlphaDecay(min_samples=12)
    a1 = _SharpeStub("a1")
    a2 = _SharpeStub("a2")
    agents = [a1, a2]

    for t in range(1, 40):
        crowd = min(1.0, t / 40)
        a1.current_sharpe = max(0.02, 1.6 * math.exp(-1.3 * crowd))
        a2.current_sharpe = max(0.02, 1.2 * math.exp(-0.9 * crowd))
        ad.update(
            crowding_intensity=crowd,
            agents=agents,
            agent_activity={"a1": 0.8, "a2": 0.5},
            order_flow_imbalance=0.6,
        )

    snap = ad.snapshot_for_api()
    assert len(snap["agent_decay_params"]) >= 1

    p = snap["agent_decay_params"][0]
    assert p["lambda"] >= 0.0
    assert p["half_life"] is None or p["half_life"] > 0

    ad.update(
        crowding_intensity=0.9,
        agents=agents,
        agent_activity={"a1": 0.9, "a2": 0.9},
        order_flow_imbalance=0.9,
    )
    buy1, sell1 = ad.current_impact_multipliers()
    assert buy1 >= sell1 >= 1.0

    ad.update(
        crowding_intensity=0.9,
        agents=agents,
        agent_activity={"a1": 0.9, "a2": 0.9},
        order_flow_imbalance=-0.9,
    )
    buy2, sell2 = ad.current_impact_multipliers()
    assert sell2 >= buy2 >= 1.0


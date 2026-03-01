"""
tests/test_analytics_phase3.py

Phase 3 analytics and regime detection tests.
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.analytics.diagnostics import Diagnostics
from engine.analytics.fragility import FragilityIndex
from engine.analytics.regime_detector import Regime, RegimeDetector
from engine.core.order import Fill


@dataclass
class _AgentStub:
    agent_id: str
    strategy_type: str
    initial_cash: float = 100_000.0


def test_diagnostics_computed_from_trade_stream():
    diag = Diagnostics(window=120)
    agents = [_AgentStub("a1", "momentum"), _AgentStub("a2", "mean_reversion")]

    mid = 100.0
    for tick in range(1, 90):
        mid_pre = mid
        mid = mid + 0.02
        fills = [
            Fill(
                buy_order_id=f"bo{tick}",
                sell_order_id=f"so{tick}",
                buy_agent_id="a1",
                sell_agent_id="a2",
                price=mid_pre + 0.03,
                qty=2,
            )
        ]
        diag.update(agents=agents, fills=fills, mid_pre=mid_pre, mid_post=mid)

    a1 = diag.compute_by_id("a1", fallback_strategy="momentum").to_dict()
    a2 = diag.compute_by_id("a2", fallback_strategy="mean_reversion").to_dict()

    assert a1["fill_count"] > 0
    assert a1["avg_fill_size"] > 0
    assert a1["spread_cost"] > 0
    assert a1["market_impact_cost"] > 0
    assert a1["turnover"] > 0
    assert a1["sharpe"] is not None
    assert a2["fill_count"] > 0


def test_fragility_spikes_when_near_depth_ratio_and_imbalance_spike():
    frag = FragilityIndex()

    # Baseline: most depth far from mid.
    baseline_lfi = []
    for t in range(1, 70):
        bids = [(90.00 - 0.01 * i, 20) for i in range(8)]
        asks = [(110.00 + 0.01 * i, 20) for i in range(8)]
        lfi = frag.update(
            tick=t,
            bid_levels=bids,
            ask_levels=asks,
            mid_price=100.0,
            fills=[],
        )
        baseline_lfi.append(lfi)

    # Stress: depth concentrated near touch + strongly one-sided prints.
    stress_fills = [
        Fill("bo1", "so1", "a", "b", price=100.15, qty=10),
        Fill("bo2", "so2", "a", "b", price=100.20, qty=10),
    ]
    stressed = frag.update(
        tick=71,
        bid_levels=[(99.99, 120), (99.98, 120), (99.97, 100)],
        ask_levels=[(100.01, 120), (100.02, 120), (100.03, 100)],
        mid_price=100.0,
        fills=stress_fills,
    )

    assert stressed > max(baseline_lfi[-5:])
    assert frag.fill_imbalance > 0


def test_regime_detector_marks_crash_prone_under_fragile_crowded_conditions():
    det = RegimeDetector(calibration_window=50)
    mid = 100.0

    # Calm calibration period.
    for t in range(1, 70):
        mid += 0.001
        det.update(
            tick=t,
            mid_price=mid,
            spread=0.05,
            depth=2000.0,
            lfi=0.1,
            crowding=0.2,
            fill_imbalance=0.05,
        )

    # Stress period.
    regime = Regime.CALM
    spread = 0.2
    depth = 1200.0
    for t in range(70, 85):
        mid += 0.08 if t % 2 == 0 else -0.07
        spread *= 1.12
        depth *= 0.90
        regime = det.update(
            tick=t,
            mid_price=mid,
            spread=spread,
            depth=depth,
            lfi=0.9,
            crowding=0.85,
            fill_imbalance=0.9,
        )

    assert regime == Regime.CRASH_PRONE


"""
engine/core/execution.py

Fill execution: applies fills to agent positions.

Also models market impact — the idea that large orders move price
against the aggressor. We use a simple square-root impact model:

    price_impact = eta * sqrt(fill_qty / avg_daily_volume)

The impacted price is stored on the Fill and used for PnL accounting,
but the book's resting price is NOT modified (that would break the LOB invariant).
Instead impact manifests through the agent's effective cost basis.

This module is intentionally thin — it does the accounting math and
returns structured position updates, it does not touch the book or agents directly.
Simulation.py is the glue that calls execution and passes results back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import math

from .order import Fill, Side


# ---------------------------------------------------------------------------
# Position tracking per agent
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """
    Live position for a single agent.
    Cash and inventory are updated after every fill.
    """
    agent_id:  str
    cash:      float = 0.0
    inventory: int   = 0

    # Running PnL components
    realized_pnl:   float = 0.0
    total_buy_qty:  int   = 0
    total_sell_qty: int   = 0
    fill_count:     int   = 0

    def mark_to_market(self, mid_price: float) -> float:
        """
        Unrealized PnL = inventory × mid_price + realized_pnl + cash
        (cash starts at 0; buying reduces cash, selling increases it)
        Full PnL = cash + inventory * mid_price
        """
        return self.cash + self.inventory * mid_price

    def sharpe(self, pnl_history: List[float], risk_free: float = 0.0) -> Optional[float]:
        """Rolling Sharpe from a list of per-tick PnL values."""
        if len(pnl_history) < 2:
            return None
        import statistics
        mean_ret = statistics.mean(pnl_history)
        std_ret  = statistics.stdev(pnl_history)
        if std_ret == 0:
            return None
        return (mean_ret - risk_free) / std_ret


# ---------------------------------------------------------------------------
# Impact model
# ---------------------------------------------------------------------------

class ImpactModel:
    """
    Linear square-root market impact.

        impact = eta * sign * sqrt(qty)

    eta is a calibration constant — start with 0.01.
    Larger eta = more illiquid market.
    Impact is symmetric: buys push price up, sells push price down.
    """

    def __init__(self, eta: float = 0.01) -> None:
        self.eta = eta

    def impact(self, qty: int, side: str) -> float:
        """
        Returns the signed price impact of a trade.
        Positive for buys (cost more), negative for sells (receive less).
        """
        sign = 1.0 if side == "BID" else -1.0
        return sign * self.eta * math.sqrt(qty)


# ---------------------------------------------------------------------------
# Execution engine — applies fills to positions
# ---------------------------------------------------------------------------

class ExecutionEngine:

    def __init__(self, impact_model: Optional[ImpactModel] = None) -> None:
        self.impact = impact_model or ImpactModel()
        self.positions: Dict[str, Position] = {}

    def get_or_create(self, agent_id: str, initial_cash: float = 100_000.0) -> Position:
        if agent_id not in self.positions:
            self.positions[agent_id] = Position(agent_id=agent_id, cash=initial_cash)
        return self.positions[agent_id]

    def apply_fills(self, fills: List[Fill], mid_price: float) -> None:
        """
        Apply a list of fills to agent positions.
        Called by simulation.py after each matching engine batch.
        """
        for fill in fills:
            self._apply_single(fill, mid_price)

    def _apply_single(self, fill: Fill, mid_price: float) -> None:
        buyer  = self.get_or_create(fill.buy_agent_id)
        seller = self.get_or_create(fill.sell_agent_id)

        # Impact adjustment: buyer pays a bit more, seller receives a bit less
        buy_impact  = self.impact.impact(fill.qty, "BID")
        sell_impact = self.impact.impact(fill.qty, "ASK")

        effective_buy_price  = fill.price + abs(buy_impact)
        effective_sell_price = fill.price - abs(sell_impact)

        # Buyer: loses cash, gains inventory
        cost = effective_buy_price * fill.qty
        buyer.cash      -= cost
        buyer.inventory += fill.qty
        buyer.total_buy_qty += fill.qty
        buyer.fill_count += 1

        # Seller: gains cash, loses inventory
        proceeds = effective_sell_price * fill.qty
        seller.cash      += proceeds
        seller.inventory -= fill.qty
        seller.total_sell_qty += fill.qty
        seller.fill_count += 1

    def snapshot(self, mid_price: float) -> List[Dict]:
        """Return position snapshot for all agents — used by leaderboard/API."""
        return [
            {
                "agent_id":      pos.agent_id,
                "cash":          round(pos.cash, 4),
                "inventory":     pos.inventory,
                "pnl":           round(pos.mark_to_market(mid_price), 4),
                "fill_count":    pos.fill_count,
                "total_buy_qty": pos.total_buy_qty,
                "total_sell_qty":pos.total_sell_qty,
            }
            for pos in self.positions.values()
        ]

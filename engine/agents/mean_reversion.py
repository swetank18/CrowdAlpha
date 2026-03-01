"""
engine/agents/mean_reversion.py

Mean reversion agent — trades on rolling z-score signal.

Strategy:
  - Computes rolling mean and std of mid-price over `window` ticks
  - z-score = (mid - mean) / std
  - Buys when z < -threshold (price is "too low")
  - Sells when z > +threshold (price is "too high")
  - Limit orders resting at a target reversion price
  - Hard inventory cap prevents unlimited accumulation

The interaction between momentum and mean-reversion agents is a primary
driver of realistic-looking price dynamics: momentum pushes price away,
mean-reversion pulls it back. Their balance determines autocorrelation.
"""

from __future__ import annotations

import numpy as np
from typing import List
from engine.core.order import Order, Side, OrderType
from .base_agent import BaseAgent, MarketState


class MeanReversionAgent(BaseAgent):

    def __init__(
        self,
        agent_id: str,
        window:    int   = 30,
        threshold: float = 1.5,    # z-score entry threshold
        order_qty: int   = 5,
        max_inv:   int   = 40,
        initial_cash: float = 100_000.0,
    ) -> None:
        super().__init__(agent_id, "mean_reversion", initial_cash,
                         window=window, threshold=threshold)
        self.window    = window
        self.threshold = threshold
        self.order_qty = order_qty
        self.max_inv   = max_inv

        self._price_history: List[float] = []
        self._last_zscore:   float       = 0.0

    # ------------------------------------------------------------------

    def on_tick(self, state: MarketState) -> List[Order]:
        mid = state.mid_or_last
        self._price_history.append(mid)

        if len(self._price_history) < self.window:
            return []

        recent = self._price_history[-self.window:]
        mean   = np.mean(recent)
        std    = np.std(recent)

        if std < 1e-8:
            return []

        z = (mid - mean) / std
        self._last_zscore = z

        orders = []

        if z < -self.threshold and self.inventory < self.max_inv:
            # Price is cheap relative to recent history → BUY
            # Place limit order at mid (expect mean reversion to fill us)
            orders.append(Order(
                agent_id   = self.agent_id,
                side       = Side.BID,
                order_type = OrderType.LIMIT,
                price      = round(mid, 4),
                qty        = self.order_qty,
            ))

        elif z > self.threshold and self.inventory > -self.max_inv:
            # Price is expensive → SELL
            orders.append(Order(
                agent_id   = self.agent_id,
                side       = Side.ASK,
                order_type = OrderType.LIMIT,
                price      = round(mid, 4),
                qty        = self.order_qty,
            ))

        # Keep history bounded
        if len(self._price_history) > self.window * 3:
            self._price_history.pop(0)

        return orders

    def factor_vector(self) -> np.ndarray:
        """
        Factor vector:
          [0, mean_rev_signal, bid_agg, ask_agg, 0]
        """
        sig = np.clip(-self._last_zscore / self.threshold, -1.0, 1.0)
        bid_agg = 0.5 if self._last_zscore < -self.threshold else 0.0
        ask_agg = 0.5 if self._last_zscore >  self.threshold else 0.0
        return np.array([0.0, sig, bid_agg, ask_agg, 0.0])

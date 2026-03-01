"""
engine/agents/momentum.py

Momentum agent — trades on EMA crossover signal.

Strategy:
  - Maintains a fast EMA and slow EMA of mid-price
  - When fast EMA crosses above slow EMA → uptrend → BUY
  - When fast EMA crosses below slow EMA → downtrend → SELL
  - Uses limit orders priced aggressively (crosses the spread by a small tick)
  - Inventory limits prevent runaway positions

This agent produces one of the strongest crowding risks: when many
momentum agents use similar EMA windows, they all enter/exit simultaneously,
amplifying price moves and compressing their collective alpha.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional
from engine.core.order import Order, Side, OrderType
from .base_agent import BaseAgent, MarketState


class MomentumAgent(BaseAgent):

    def __init__(
        self,
        agent_id: str,
        fast_window: int   = 5,
        slow_window: int   = 20,
        order_qty:   int   = 5,
        max_inv:     int   = 50,
        aggression:  float = 0.001,  # fraction above/below mid to price limit
        initial_cash: float = 100_000.0,
    ) -> None:
        super().__init__(agent_id, "momentum", initial_cash,
                         fast_window=fast_window, slow_window=slow_window)
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.order_qty   = order_qty
        self.max_inv     = max_inv
        self.aggression  = aggression

        self._price_history: List[float] = []
        self._last_signal:   float       = 0.0   # stored for factor vector

    # ------------------------------------------------------------------

    def on_tick(self, state: MarketState) -> List[Order]:
        mid = state.mid_or_last
        self._price_history.append(mid)

        # Need at least slow_window prices
        if len(self._price_history) < self.slow_window:
            return []

        fast_ema = self._ema(self._fast_series(), self.fast_window)
        slow_ema = self._ema(self._slow_series(), self.slow_window)
        signal   = (fast_ema - slow_ema) / (slow_ema + 1e-9)
        self._last_signal = signal

        orders = []

        if signal > 0.0001 and self.inventory < self.max_inv:
            # Uptrend: buy
            buy_price = round(mid * (1 + self.aggression), 4)
            orders.append(Order(
                agent_id   = self.agent_id,
                side       = Side.BID,
                order_type = OrderType.LIMIT,
                price      = buy_price,
                qty        = self.order_qty,
            ))

        elif signal < -0.0001 and self.inventory > -self.max_inv:
            # Downtrend: sell
            sell_price = round(mid * (1 - self.aggression), 4)
            orders.append(Order(
                agent_id   = self.agent_id,
                side       = Side.ASK,
                order_type = OrderType.LIMIT,
                price      = sell_price,
                qty        = self.order_qty,
            ))

        # Keep price history bounded
        if len(self._price_history) > self.slow_window * 3:
            self._price_history.pop(0)

        return orders

    def factor_vector(self) -> np.ndarray:
        """
        Factor vector for crowding module:
          [momentum_signal, 0 (no mean-rev), bid_agg, ask_agg, 0]
        """
        sig = np.clip(self._last_signal * 100, -1.0, 1.0)
        bid_agg = self.aggression if sig > 0 else 0.0
        ask_agg = self.aggression if sig < 0 else 0.0
        return np.array([sig, 0.0, bid_agg, ask_agg, 0.0])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fast_series(self) -> List[float]:
        return self._price_history[-self.fast_window * 2:]

    def _slow_series(self) -> List[float]:
        return self._price_history[-self.slow_window * 2:]

    @staticmethod
    def _ema(prices: List[float], window: int) -> float:
        """Exponential moving average."""
        alpha = 2.0 / (window + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = alpha * p + (1 - alpha) * ema
        return ema

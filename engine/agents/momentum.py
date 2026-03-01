"""
engine/agents/momentum.py

Momentum agent based on fast/slow EMA trend.
"""

from __future__ import annotations

from typing import List

import numpy as np

from engine.core.order import Order, OrderType, Side
from .base_agent import BaseAgent, MarketState


class MomentumAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        fast_window: int = 6,
        slow_window: int = 24,
        order_qty: int = 5,
        max_inv: int = 50,
        aggression: float = 0.0012,
        entry_threshold: float = 0.0004,
        market_threshold: float = 0.0012,
        initial_cash: float = 100_000.0,
    ) -> None:
        super().__init__(
            agent_id,
            "momentum",
            initial_cash,
            fast_window=fast_window,
            slow_window=slow_window,
            entry_threshold=entry_threshold,
            market_threshold=market_threshold,
        )
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.order_qty = order_qty
        self.max_inv = max_inv
        self.aggression = aggression
        self.entry_threshold = entry_threshold
        self.market_threshold = market_threshold

        self._price_history: List[float] = []
        self._last_signal: float = 0.0
        self._last_bid_agg: float = 0.0
        self._last_ask_agg: float = 0.0
        self._last_turnover: float = 0.0

    def on_tick(self, state: MarketState) -> List[Order]:
        mid = state.mid_or_last
        self._price_history.append(mid)

        self._last_bid_agg = 0.0
        self._last_ask_agg = 0.0
        self._last_turnover = 0.0

        if len(self._price_history) < self.slow_window:
            return []

        fast_ema = self._ema(self._price_history[-self.fast_window * 2 :], self.fast_window)
        slow_ema = self._ema(self._price_history[-self.slow_window * 2 :], self.slow_window)
        signal = (fast_ema - slow_ema) / (slow_ema + 1e-9)

        # Inventory feedback prevents runaway directional accumulation.
        inventory_pressure = (self.inventory / max(self.max_inv, 1)) * 0.0008
        adjusted_signal = signal - inventory_pressure
        self._last_signal = adjusted_signal

        strength = min(abs(adjusted_signal) / max(self.market_threshold, 1e-9), 2.0)
        qty = max(1, int(round(self.order_qty * (1.0 + 0.5 * strength))))

        if adjusted_signal > self.market_threshold and self.inventory < self.max_inv:
            self._last_bid_agg = 1.0
            self._last_turnover = 1.0
            return [
                Order(
                    agent_id=self.agent_id,
                    side=Side.BID,
                    order_type=OrderType.MARKET,
                    price=mid,
                    qty=qty,
                )
            ]

        if adjusted_signal > self.entry_threshold and self.inventory < self.max_inv:
            buy_price = self._aggressive_buy_price(state, mid)
            self._last_bid_agg = min(self.aggression * 200.0, 1.0)
            self._last_turnover = 0.5
            return [
                Order(
                    agent_id=self.agent_id,
                    side=Side.BID,
                    order_type=OrderType.LIMIT,
                    price=buy_price,
                    qty=qty,
                )
            ]

        if adjusted_signal < -self.market_threshold and self.inventory > -self.max_inv:
            self._last_ask_agg = 1.0
            self._last_turnover = 1.0
            return [
                Order(
                    agent_id=self.agent_id,
                    side=Side.ASK,
                    order_type=OrderType.MARKET,
                    price=mid,
                    qty=qty,
                )
            ]

        if adjusted_signal < -self.entry_threshold and self.inventory > -self.max_inv:
            sell_price = self._aggressive_sell_price(state, mid)
            self._last_ask_agg = min(self.aggression * 200.0, 1.0)
            self._last_turnover = 0.5
            return [
                Order(
                    agent_id=self.agent_id,
                    side=Side.ASK,
                    order_type=OrderType.LIMIT,
                    price=sell_price,
                    qty=qty,
                )
            ]

        if len(self._price_history) > self.slow_window * 4:
            self._price_history.pop(0)
        return []

    def factor_vector(self) -> np.ndarray:
        sig = np.clip(self._last_signal * 250.0, -1.0, 1.0)
        return np.array(
            [
                sig,
                0.0,
                self._last_bid_agg,
                self._last_ask_agg,
                self._last_turnover,
            ]
        )

    @staticmethod
    def _ema(prices: List[float], window: int) -> float:
        alpha = 2.0 / (window + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return float(ema)

    def _aggressive_buy_price(self, state: MarketState, mid: float) -> float:
        if state.best_ask is not None:
            return round(state.best_ask, 4)
        return round(mid * (1.0 + self.aggression), 4)

    def _aggressive_sell_price(self, state: MarketState, mid: float) -> float:
        if state.best_bid is not None:
            return round(state.best_bid, 4)
        return round(mid * (1.0 - self.aggression), 4)


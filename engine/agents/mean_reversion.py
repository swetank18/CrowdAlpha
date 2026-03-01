"""
engine/agents/mean_reversion.py

Mean-reversion agent using rolling z-score.
"""

from __future__ import annotations

from typing import List

import numpy as np

from engine.core.order import Order, OrderType, Side
from .base_agent import BaseAgent, MarketState


class MeanReversionAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        window: int = 30,
        threshold: float = 1.4,
        order_qty: int = 5,
        max_inv: int = 40,
        aggression: float = 0.0008,
        market_z: float = 2.4,
        initial_cash: float = 100_000.0,
    ) -> None:
        super().__init__(
            agent_id,
            "mean_reversion",
            initial_cash,
            window=window,
            threshold=threshold,
            market_z=market_z,
        )
        self.window = window
        self.threshold = threshold
        self.order_qty = order_qty
        self.max_inv = max_inv
        self.aggression = aggression
        self.market_z = market_z

        self._price_history: List[float] = []
        self._last_zscore: float = 0.0
        self._last_bid_agg: float = 0.0
        self._last_ask_agg: float = 0.0
        self._last_turnover: float = 0.0

    def on_tick(self, state: MarketState) -> List[Order]:
        mid = state.mid_or_last
        self._price_history.append(mid)

        self._last_bid_agg = 0.0
        self._last_ask_agg = 0.0
        self._last_turnover = 0.0

        if len(self._price_history) < self.window:
            return []

        recent = np.array(self._price_history[-self.window :], dtype=float)
        mean = float(np.mean(recent))
        std = float(np.std(recent))
        if std < 1e-8:
            return []

        z = (mid - mean) / std
        # Inventory feedback: lean less into the same side when loaded.
        z -= (self.inventory / max(self.max_inv, 1)) * 0.5
        self._last_zscore = z

        if len(self._price_history) > self.window * 4:
            self._price_history.pop(0)

        if z < -self.market_z and self.inventory < self.max_inv:
            self._last_bid_agg = 1.0
            self._last_turnover = 1.0
            return [
                Order(
                    agent_id=self.agent_id,
                    side=Side.BID,
                    order_type=OrderType.MARKET,
                    price=mid,
                    qty=self.order_qty + 2,
                )
            ]

        if z < -self.threshold and self.inventory < self.max_inv:
            self._last_bid_agg = min(self.aggression * 300.0, 1.0)
            self._last_turnover = 0.5
            return [
                Order(
                    agent_id=self.agent_id,
                    side=Side.BID,
                    order_type=OrderType.LIMIT,
                    price=self._buy_price(state, mid, mean),
                    qty=self.order_qty,
                )
            ]

        if z > self.market_z and self.inventory > -self.max_inv:
            self._last_ask_agg = 1.0
            self._last_turnover = 1.0
            return [
                Order(
                    agent_id=self.agent_id,
                    side=Side.ASK,
                    order_type=OrderType.MARKET,
                    price=mid,
                    qty=self.order_qty + 2,
                )
            ]

        if z > self.threshold and self.inventory > -self.max_inv:
            self._last_ask_agg = min(self.aggression * 300.0, 1.0)
            self._last_turnover = 0.5
            return [
                Order(
                    agent_id=self.agent_id,
                    side=Side.ASK,
                    order_type=OrderType.LIMIT,
                    price=self._sell_price(state, mid, mean),
                    qty=self.order_qty,
                )
            ]

        return []

    def factor_vector(self) -> np.ndarray:
        sig = np.clip(-self._last_zscore / max(self.threshold, 1e-9), -1.0, 1.0)
        return np.array(
            [
                0.0,
                sig,
                self._last_bid_agg,
                self._last_ask_agg,
                self._last_turnover,
            ]
        )

    def _buy_price(self, state: MarketState, mid: float, mean: float) -> float:
        target = min(mid, mean)
        if state.best_bid is not None:
            target = max(target, state.best_bid + 0.0001)
        return round(max(target, mid * (1.0 - self.aggression)), 4)

    def _sell_price(self, state: MarketState, mid: float, mean: float) -> float:
        target = max(mid, mean)
        if state.best_ask is not None:
            target = min(target, state.best_ask - 0.0001)
        return round(min(target, mid * (1.0 + self.aggression)), 4)


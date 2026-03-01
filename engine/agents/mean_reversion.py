"""
engine/agents/mean_reversion.py

Mean-reversion agent using rolling z-score.
"""

from __future__ import annotations

import random
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
        self._last_threshold_used: float = threshold
        self._last_tick_seen: int = 0
        self._cooldown_until_tick: int = 0

    def on_tick(self, state: MarketState) -> List[Order]:
        self._last_tick_seen = int(state.tick)
        if state.tick < self._cooldown_until_tick:
            return []

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
        threshold_scale = 1.0 + self._signed_noise(0.15, 0.20)
        target_scale = 1.0 + self._signed_noise(0.15, 0.20)
        threshold_used = max(self.threshold * threshold_scale, 1e-6)
        market_threshold_used = max(self.market_z * threshold_scale, 1e-6)
        self._last_threshold_used = threshold_used

        if len(self._price_history) > self.window * 4:
            self._price_history.pop(0)

        if z < -market_threshold_used and self.inventory < self.max_inv:
            self._last_bid_agg = 1.0
            self._last_turnover = 1.0
            qty = self._cap_qty_to_quote_depth(state, Side.BID, self.order_qty + 2)
            return [
                Order(
                    agent_id=self.agent_id,
                    side=Side.BID,
                    order_type=OrderType.MARKET,
                    price=mid,
                    qty=qty,
                )
            ]

        if z < -threshold_used and self.inventory < self.max_inv:
            self._last_bid_agg = min(self.aggression * 300.0, 1.0)
            self._last_turnover = 0.5
            qty = self._cap_qty_to_quote_depth(state, Side.BID, self.order_qty)
            return [
                Order(
                    agent_id=self.agent_id,
                    side=Side.BID,
                    order_type=OrderType.LIMIT,
                    price=self._buy_price(state, mid, mean, target_scale),
                    qty=qty,
                )
            ]

        if z > market_threshold_used and self.inventory > -self.max_inv:
            self._last_ask_agg = 1.0
            self._last_turnover = 1.0
            qty = self._cap_qty_to_quote_depth(state, Side.ASK, self.order_qty + 2)
            return [
                Order(
                    agent_id=self.agent_id,
                    side=Side.ASK,
                    order_type=OrderType.MARKET,
                    price=mid,
                    qty=qty,
                )
            ]

        if z > threshold_used and self.inventory > -self.max_inv:
            self._last_ask_agg = min(self.aggression * 300.0, 1.0)
            self._last_turnover = 0.5
            qty = self._cap_qty_to_quote_depth(state, Side.ASK, self.order_qty)
            return [
                Order(
                    agent_id=self.agent_id,
                    side=Side.ASK,
                    order_type=OrderType.LIMIT,
                    price=self._sell_price(state, mid, mean, target_scale),
                    qty=qty,
                )
            ]

        return []

    def factor_vector(self) -> np.ndarray:
        sig = np.clip(-self._last_zscore / max(self._last_threshold_used, 1e-9), -1.0, 1.0)
        return np.array(
            [
                0.0,
                sig,
                self._last_bid_agg,
                self._last_ask_agg,
                self._last_turnover,
            ]
        )

    def sync_position(self, cash: float, inventory: int, pnl: float) -> None:
        prev_inventory = self.inventory
        super().sync_position(cash, inventory, pnl)
        if inventory != prev_inventory:
            self._cooldown_until_tick = self._last_tick_seen + random.randint(5, 15)

    def _buy_price(self, state: MarketState, mid: float, mean: float, target_scale: float) -> float:
        base_target = min(mid, mean)
        target = mid + (base_target - mid) * target_scale
        if state.best_bid is not None:
            target = max(target, state.best_bid + 0.0001)
        return round(max(target, mid * (1.0 - self.aggression)), 4)

    def _sell_price(self, state: MarketState, mid: float, mean: float, target_scale: float) -> float:
        base_target = max(mid, mean)
        target = mid + (base_target - mid) * target_scale
        if state.best_ask is not None:
            target = min(target, state.best_ask - 0.0001)
        return round(min(target, mid * (1.0 + self.aggression)), 4)

    @staticmethod
    def _signed_noise(min_abs: float, max_abs: float) -> float:
        mag = random.uniform(min_abs, max_abs)
        return mag if random.random() < 0.5 else -mag

    @staticmethod
    def _cap_qty_to_quote_depth(state: MarketState, side: Side, desired_qty: int) -> int:
        if desired_qty <= 0:
            return 1
        if side == Side.BID:
            if state.ask_levels:
                depth = int(state.ask_levels[0][1])
            else:
                depth = 0
        else:
            if state.bid_levels:
                depth = int(state.bid_levels[0][1])
            else:
                depth = 0
        if depth <= 0:
            return int(desired_qty)
        cap = max(1, int(depth * 0.30))
        return int(max(1, min(desired_qty, cap)))

"""
engine/agents/market_maker.py

Inventory-aware two-sided market maker.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from engine.core.order import Order, OrderType, Side
from .base_agent import BaseAgent, MarketState


class MarketMakerAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        base_spread: float = 0.08,
        min_half_spread: float = 0.02,
        vol_mult: float = 70.0,
        inventory_skew: float = 0.015,
        inv_penalty: float = 0.002,
        quote_jitter: float = 0.03,
        order_qty: int = 8,
        max_inv: int = 60,
        vol_lookback: int = 30,
        initial_cash: float = 100_000.0,
    ) -> None:
        super().__init__(
            agent_id,
            "market_maker",
            initial_cash,
            base_spread=base_spread,
            min_half_spread=min_half_spread,
            vol_mult=vol_mult,
            inventory_skew=inventory_skew,
            inv_penalty=inv_penalty,
            quote_jitter=quote_jitter,
        )
        self.base_spread = base_spread
        self.min_half_spread = min_half_spread
        self.vol_mult = vol_mult
        self.inventory_skew = inventory_skew
        self.inv_penalty = inv_penalty
        self.quote_jitter = quote_jitter
        self.order_qty = order_qty
        self.max_inv = max_inv
        self.vol_lookback = vol_lookback

        self._price_history: List[float] = []
        self._resting_bid_id: Optional[str] = None
        self._resting_ask_id: Optional[str] = None
        self._last_bid_agg: float = 0.0
        self._last_ask_agg: float = 0.0
        self._last_turnover: float = 0.0

    def on_tick(self, state: MarketState) -> List[Order]:
        mid = state.mid_or_last
        self._price_history.append(mid)

        orders: List[Order] = []

        # Refresh quotes every tick.
        if self._resting_bid_id:
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    side=Side.BID,
                    order_type=OrderType.CANCEL,
                    price=mid,
                    qty=1,
                    cancel_target=self._resting_bid_id,
                )
            )
            self._resting_bid_id = None
        if self._resting_ask_id:
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    side=Side.ASK,
                    order_type=OrderType.CANCEL,
                    price=mid,
                    qty=1,
                    cancel_target=self._resting_ask_id,
                )
            )
            self._resting_ask_id = None

        vol = self._realized_vol()
        inv_ratio = self.inventory / max(self.max_inv, 1)

        # Reservation price shifts away from accumulated inventory.
        reservation = mid - (self.inventory_skew * self.inventory)
        reservation += float(np.random.normal(0.0, self.quote_jitter))

        # Lean with recent trade pressure to avoid stale quoting.
        if state.recent_trades:
            signed_flow = 0.0
            for fill in state.recent_trades[-8:]:
                qty = float(fill.get("qty", 0.0))
                signed_flow += qty if fill.get("buy_agent") != self.agent_id[:8] else -qty
            reservation += 0.001 * signed_flow

        dynamic_half = (
            self.base_spread
            + self.min_half_spread
            + self.vol_mult * vol
            + self.inv_penalty * abs(self.inventory)
        )
        crowd_scale = 1.0 + 1.25 * max(state.crowding_intensity, 0.0)
        half_spread = max(dynamic_half * crowd_scale, self.min_half_spread)

        bid_price = reservation - half_spread
        ask_price = reservation + half_spread

        pressure = float(np.clip(state.crowding_side_pressure, -1.0, 1.0))
        if pressure > 0:
            # Buy-side crowding: widen ask side.
            ask_price += half_spread * pressure * 0.8
        elif pressure < 0:
            # Sell-side crowding: widen bid side.
            bid_price -= half_spread * abs(pressure) * 0.8

        # Keep quotes sensible relative to visible top of book.
        if state.best_bid is not None:
            bid_price = min(bid_price, state.best_bid + 0.01)
        if state.best_ask is not None:
            ask_price = max(ask_price, state.best_ask - 0.01)
        if bid_price >= ask_price:
            center = (bid_price + ask_price) / 2.0
            bid_price = center - self.min_half_spread
            ask_price = center + self.min_half_spread

        # Scale order size down near inventory limits.
        inv_scale = max(0.25, 1.0 - abs(inv_ratio))
        quote_qty = max(1, int(round(self.order_qty * inv_scale)))

        self._last_bid_agg = min(half_spread / max(mid, 1e-9), 1.0)
        self._last_ask_agg = self._last_bid_agg
        self._last_turnover = 0.3

        if self.inventory < self.max_inv and bid_price > 0:
            bid = Order(
                agent_id=self.agent_id,
                side=Side.BID,
                order_type=OrderType.LIMIT,
                price=round(bid_price, 4),
                qty=quote_qty,
            )
            orders.append(bid)
            self._resting_bid_id = bid.order_id

        if self.inventory > -self.max_inv and ask_price > 0:
            ask = Order(
                agent_id=self.agent_id,
                side=Side.ASK,
                order_type=OrderType.LIMIT,
                price=round(ask_price, 4),
                qty=quote_qty,
            )
            orders.append(ask)
            self._resting_ask_id = ask.order_id

        if len(self._price_history) > self.vol_lookback * 4:
            self._price_history.pop(0)

        return orders

    def factor_vector(self) -> np.ndarray:
        return np.array(
            [
                0.0,
                0.0,
                self._last_bid_agg,
                self._last_ask_agg,
                self._last_turnover,
            ]
        )

    def _realized_vol(self) -> float:
        if len(self._price_history) < 2:
            return 0.0
        recent = self._price_history[-self.vol_lookback :]
        if len(recent) < 2:
            return 0.0
        log_returns = np.diff(np.log(np.array(recent) + 1e-9))
        return float(np.std(log_returns))

"""
engine/agents/market_maker.py

Market maker agent — posts two-sided quotes around mid-price.

This is the most complex agent because it must balance three competing concerns:
  1. Earning the spread (post tight quotes that fill)
  2. Inventory risk (accumulated inventory is directional risk)
  3. Adverse selection (smart agents will pick off stale quotes)

Implementation uses a simplified Avellaneda-Stoikov (2008) heuristic:
  - Reservation price = mid - γ × σ² × inventory
      (skews reservation price away from large inventory)
  - Spread = γ × σ² + (2/γ) × ln(1 + γ/k)
      (widens with volatility; we simplify to: spread = base_spread + inv_penalty)
  - Posts bid at: reservation - half_spread
  - Posts ask at: reservation + half_spread

The market maker is the primary liquidity provider — without it the book
runs dry and spreads blow out. It is also the agent most harmed by crowding,
because when momentum agents all move in the same direction the MM accumulates
toxic inventory.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional
from engine.core.order import Order, Side, OrderType
from .base_agent import BaseAgent, MarketState


class MarketMakerAgent(BaseAgent):

    def __init__(
        self,
        agent_id:       str,
        base_spread:    float = 0.5,     # minimum half-spread in price units
        inv_penalty:    float = 0.05,    # additional spread per unit of |inventory|
        order_qty:      int   = 8,
        max_inv:        int   = 60,
        vol_lookback:   int   = 20,
        initial_cash:   float = 100_000.0,
    ) -> None:
        super().__init__(agent_id, "market_maker", initial_cash)
        self.base_spread  = base_spread
        self.inv_penalty  = inv_penalty
        self.order_qty    = order_qty
        self.max_inv      = max_inv
        self.vol_lookback = vol_lookback

        self._price_history: List[float] = []
        self._last_bid_agg: float = 0.0
        self._last_ask_agg: float = 0.0
        self._resting_bid_id: Optional[str] = None
        self._resting_ask_id: Optional[str] = None

    # ------------------------------------------------------------------

    def on_tick(self, state: MarketState) -> List[Order]:
        mid = state.mid_or_last
        self._price_history.append(mid)

        orders = []

        # Cancel previous resting quotes (refresh each tick)
        if self._resting_bid_id:
            orders.append(Order(
                agent_id=self.agent_id, side=Side.BID,
                order_type=OrderType.CANCEL, price=0.0, qty=1,
                cancel_target=self._resting_bid_id,
            ))
            self._resting_bid_id = None
        if self._resting_ask_id:
            orders.append(Order(
                agent_id=self.agent_id, side=Side.ASK,
                order_type=OrderType.CANCEL, price=0.0, qty=1,
                cancel_target=self._resting_ask_id,
            ))
            self._resting_ask_id = None

        # Compute realized volatility
        vol = self._realized_vol()

        # Reservation price (skew away from large inventory)
        gamma = 0.1
        reservation = mid - gamma * (vol ** 2) * self.inventory

        # Half-spread: base + inventory penalty + vol adjustment
        inventory_skew = abs(self.inventory) * self.inv_penalty
        half_spread = self.base_spread + inventory_skew + vol * 0.5

        bid_price = round(reservation - half_spread, 4)
        ask_price = round(reservation + half_spread, 4)

        self._last_bid_agg = half_spread / (mid + 1e-9)
        self._last_ask_agg = half_spread / (mid + 1e-9)

        # Only post bid if we're not too long
        if self.inventory < self.max_inv and bid_price > 0:
            bid = Order(
                agent_id=self.agent_id, side=Side.BID,
                order_type=OrderType.LIMIT,
                price=bid_price, qty=self.order_qty,
            )
            orders.append(bid)
            self._resting_bid_id = bid.order_id

        # Only post ask if we're not too short
        if self.inventory > -self.max_inv:
            ask = Order(
                agent_id=self.agent_id, side=Side.ASK,
                order_type=OrderType.LIMIT,
                price=ask_price, qty=self.order_qty,
            )
            orders.append(ask)
            self._resting_ask_id = ask.order_id

        # Bound history
        if len(self._price_history) > self.vol_lookback * 3:
            self._price_history.pop(0)

        return orders

    def factor_vector(self) -> np.ndarray:
        """
        Factor vector:
          [0, 0, bid_aggressiveness, ask_aggressiveness, 0]
        Market makers have no directional signal, only spread posting behavior.
        """
        return np.array([0.0, 0.0, self._last_bid_agg, self._last_ask_agg, 0.0])

    # ------------------------------------------------------------------

    def _realized_vol(self) -> float:
        if len(self._price_history) < 2:
            return 1.0
        recent = self._price_history[-self.vol_lookback:]
        if len(recent) < 2:
            return 1.0
        log_returns = np.diff(np.log(np.array(recent) + 1e-9))
        return float(np.std(log_returns))

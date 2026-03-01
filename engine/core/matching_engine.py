"""
engine/core/matching_engine.py

The matching engine: the single authority that executes trades.

Responsibilities:
  1. Accept incoming orders each tick (limit, market, cancel)
  2. Route cancels immediately
  3. For limit orders: try to cross against the opposing side first,
     then rest any unfilled remainder on the book
  4. For market orders: take liquidity until filled or book is exhausted
  5. Produce Fill records for every execution
  6. Track queue position (arrival order within the same price level)

Design contract:
  - Stateless between process() calls — all state lives in the OrderBook
  - Returns a list of Fills from this batch
  - Caller is responsible for applying fills to agent positions (execution.py)
"""

from __future__ import annotations

from typing import List, Callable, Optional
from .order import Order, Fill, Side, OrderType
from .order_book import OrderBook


# Type alias for event callbacks
FillCallback = Callable[[Fill], None]


class MatchingEngine:

    def __init__(
        self,
        book: OrderBook,
        on_fill: Optional[FillCallback] = None,
    ) -> None:
        self.book = book
        self._on_fill = on_fill   # optional real-time callback

    # ------------------------------------------------------------------
    # Main entry point — called once per tick by simulation.py
    # ------------------------------------------------------------------

    def process(self, orders: List[Order]) -> List[Fill]:
        """
        Process a batch of orders. Orders are processed in the order
        they arrive (simulation enforces randomised submission order
        per tick to prevent first-mover bias).

        Returns all fills generated this tick.
        """
        fills: List[Fill] = []

        for order in orders:
            if order.order_type == OrderType.CANCEL:
                self._process_cancel(order)
            elif order.order_type == OrderType.MARKET:
                fills.extend(self._process_market(order))
            elif order.order_type == OrderType.LIMIT:
                fills.extend(self._process_limit(order))

        return fills

    # ------------------------------------------------------------------
    # Market orders — consume until filled or book exhausted
    # ------------------------------------------------------------------

    def _process_market(self, order: Order) -> List[Fill]:
        fills: List[Fill] = []
        if order.side == Side.BID:
            # Buy market order hits resting asks (lowest first)
            while order.remaining_qty > 0:
                best = self.book.best_ask
                if best is None:
                    break
                fill = self._execute(order, best)
                fills.append(fill)
        else:
            # Sell market order hits resting bids (highest first)
            while order.remaining_qty > 0:
                best = self.book.best_bid
                if best is None:
                    break
                fill = self._execute(order, best)
                fills.append(fill)
        return fills

    # ------------------------------------------------------------------
    # Limit orders — cross then rest
    # ------------------------------------------------------------------

    def _process_limit(self, order: Order) -> List[Fill]:
        fills: List[Fill] = []

        if order.side == Side.BID:
            # Bid limit crosses against resting asks at ask_price <= bid_price
            while order.remaining_qty > 0:
                best_ask = self.book.best_ask
                if best_ask is None or best_ask.price > order.price:
                    break  # no crossing
                fill = self._execute(order, best_ask)
                fills.append(fill)
        else:
            # Ask limit crosses against resting bids at bid_price >= ask_price
            while order.remaining_qty > 0:
                best_bid = self.book.best_bid
                if best_bid is None or best_bid.price < order.price:
                    break  # no crossing
                fill = self._execute(order, best_bid)
                fills.append(fill)

        # Rest any unfilled portion on the book
        if order.remaining_qty > 0:
            self.book.add(order)

        return fills

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def _process_cancel(self, order: Order) -> None:
        if order.cancel_target:
            self.book.cancel(order.cancel_target)

    # ------------------------------------------------------------------
    # Core fill execution — called every time two orders cross
    # ------------------------------------------------------------------

    def _execute(self, aggressor: Order, resting: Order) -> Fill:
        """
        Execute a trade between the aggressor (incoming) and the resting
        order. The fill price is always the resting order's price
        (standard convention: price improvement goes to the aggressor).

        Handles partial fills: whichever side is smaller fills fully,
        the larger side's remaining_qty is decremented.
        """
        fill_qty = min(aggressor.remaining_qty, resting.remaining_qty)
        fill_price = resting.price

        aggressor.remaining_qty -= fill_qty
        resting.remaining_qty -= fill_qty

        # If resting order is fully consumed, remove from book
        if resting.remaining_qty == 0:
            self.book.mark_dead(resting.order_id)

        # Identify buyer and seller
        if aggressor.side == Side.BID:
            buy_order_id, buy_agent_id   = aggressor.order_id, aggressor.agent_id
            sell_order_id, sell_agent_id = resting.order_id,   resting.agent_id
        else:
            buy_order_id, buy_agent_id   = resting.order_id,   resting.agent_id
            sell_order_id, sell_agent_id = aggressor.order_id, aggressor.agent_id

        fill = Fill(
            buy_order_id=buy_order_id,
            sell_order_id=sell_order_id,
            buy_agent_id=buy_agent_id,
            sell_agent_id=sell_agent_id,
            price=fill_price,
            qty=fill_qty,
        )

        if self._on_fill:
            self._on_fill(fill)

        return fill

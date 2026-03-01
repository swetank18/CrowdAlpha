"""
engine/core/order_book.py

Limit Order Book (LOB) — maintains the resting bid and ask queues.

Design:
  - Bids: max-heap (highest price has priority)
  - Asks: min-heap (lowest price has priority)
  - Within the same price level: FIFO by timestamp (price-time priority)
  - Heap entries: (priority_key, timestamp, order) to break ties deterministically

The book does NOT execute matches — that is the matching engine's job.
The book's responsibility: store orders, expose best prices, answer
depth queries, and cancel orders by id.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
from .order import Order, Side


# ---------------------------------------------------------------------------
# Internal heap key helpers
# ---------------------------------------------------------------------------

def _bid_key(order: Order) -> tuple:
    """
    Bids: highest price first, earliest timestamp second.
    Negate price so Python's min-heap behaves as a max-heap.
    """
    return (-order.price, order.timestamp)


def _ask_key(order: Order) -> tuple:
    """
    Asks: lowest price first, earliest timestamp second.
    """
    return (order.price, order.timestamp)


# ---------------------------------------------------------------------------
# OrderBook
# ---------------------------------------------------------------------------

class OrderBook:
    """
    Central limit order book.

    Internally we keep:
      _bid_heap / _ask_heap  — priority queues for fast best-bid/ask lookup
      _orders                — dict id→Order for O(1) cancel lookup
      _cancelled             — set of cancelled order ids (lazy removal)

    Lazy removal: instead of removing from the heap on cancel (expensive),
    we mark the id as cancelled and skip it when we pop.
    """

    def __init__(self) -> None:
        # (priority_key, order) tuples
        self._bid_heap: List[Tuple] = []
        self._ask_heap: List[Tuple] = []

        # id → Order (all live resting orders)
        self._orders: Dict[str, Order] = {}

        # set of order ids that have been cancelled or fully filled
        self._dead: set = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, order: Order) -> None:
        """Add a resting limit order to the book."""
        self._orders[order.order_id] = order
        if order.side == Side.BID:
            heapq.heappush(self._bid_heap, (_bid_key(order), order))
        else:
            heapq.heappush(self._ask_heap, (_ask_key(order), order))

    def cancel(self, order_id: str) -> Optional[Order]:
        """
        Cancel a resting order. Returns the order if found, None if unknown.
        Uses lazy removal — just marks as dead, heap cleanup happens on peek/pop.
        """
        if order_id in self._orders:
            order = self._orders.pop(order_id)
            self._dead.add(order_id)
            return order
        return None

    def mark_dead(self, order_id: str) -> None:
        """Mark an order as no longer live (used by matching engine on fill)."""
        self._dead.add(order_id)
        self._orders.pop(order_id, None)

    @property
    def best_bid(self) -> Optional[Order]:
        """Highest-priority resting bid, or None if book is empty."""
        return self._peek(self._bid_heap)

    @property
    def best_ask(self) -> Optional[Order]:
        """Lowest-priced resting ask, or None if book is empty."""
        return self._peek(self._ask_heap)

    @property
    def mid_price(self) -> Optional[float]:
        bb = self.best_bid
        ba = self.best_ask
        if bb and ba:
            return (bb.price + ba.price) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        bb = self.best_bid
        ba = self.best_ask
        if bb and ba:
            return ba.price - bb.price
        return None

    def pop_best_bid(self) -> Optional[Order]:
        return self._pop(self._bid_heap)

    def pop_best_ask(self) -> Optional[Order]:
        return self._pop(self._ask_heap)

    def depth_snapshot(self, n_levels: int = 10) -> Dict:
        """
        Return a dict with bid and ask price levels (up to n_levels each).
        Format: {
          'bids': [(price, total_qty), ...],   # sorted high→low
          'asks': [(price, total_qty), ...],   # sorted low→high
        }
        Does NOT consume or modify the book.
        """
        bids: Dict[float, int] = defaultdict(int)
        asks: Dict[float, int] = defaultdict(int)

        for order in self._orders.values():
            if order.remaining_qty <= 0:
                continue
            if order.side == Side.BID:
                bids[order.price] += order.remaining_qty
            else:
                asks[order.price] += order.remaining_qty

        sorted_bids = sorted(bids.items(), key=lambda x: -x[0])[:n_levels]
        sorted_asks = sorted(asks.items(), key=lambda x:  x[0])[:n_levels]

        return {
            "bids": sorted_bids,
            "asks": sorted_asks,
            "mid_price": self.mid_price,
            "spread": self.spread,
        }

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def __len__(self) -> int:
        return len(self._orders)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _peek(self, heap: List) -> Optional[Order]:
        """Return the top-priority live order without removing it."""
        while heap:
            _, order = heap[0]
            if order.order_id not in self._dead and order.remaining_qty > 0:
                return order
            heapq.heappop(heap)
        return None

    def _pop(self, heap: List) -> Optional[Order]:
        """Pop and return the top-priority live order."""
        while heap:
            _, order = heapq.heappop(heap)
            if order.order_id not in self._dead and order.remaining_qty > 0:
                return order
        return None

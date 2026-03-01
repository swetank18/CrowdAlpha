"""
engine/core/order.py

Order dataclass — the atomic unit of the simulation.
Every agent action produces an Order. The matching engine
consumes Orders and produces Fills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import uuid
import time


class Side(Enum):
    BID = "BID"   # buy
    ASK = "ASK"   # sell


class OrderType(Enum):
    LIMIT  = "LIMIT"   # rests on book if not immediately filled
    MARKET = "MARKET"  # takes liquidity at best available price
    CANCEL = "CANCEL"  # removes a resting order from the book


@dataclass
class Order:
    """
    Immutable order record. Fields are set at creation; only
    `remaining_qty` is mutated in-place by the matching engine
    as partial fills occur.
    """
    agent_id:      str
    side:          Side
    order_type:    OrderType
    price:         float           # ignored for MARKET and CANCEL
    qty:           int             # original quantity
    order_id:      str       = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:     int       = field(default_factory=lambda: time.time_ns())
    remaining_qty: int       = field(init=False)

    # For CANCEL orders: the id of the order to cancel
    cancel_target: Optional[str] = None

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"Order qty must be positive, got {self.qty}")
        self.remaining_qty = self.qty

    @property
    def is_filled(self) -> bool:
        return self.remaining_qty == 0

    @property
    def filled_qty(self) -> int:
        return self.qty - self.remaining_qty

    def __repr__(self) -> str:
        return (
            f"Order({self.order_type.value} {self.side.value} "
            f"qty={self.qty} @{self.price:.4f} "
            f"rem={self.remaining_qty} agent={self.agent_id[:8]})"
        )


@dataclass
class Fill:
    """
    Produced by the matching engine whenever two orders cross.
    Records both sides of the trade.
    """
    buy_order_id:  str
    sell_order_id: str
    buy_agent_id:  str
    sell_agent_id: str
    price:         float
    qty:           int
    timestamp:     int = field(default_factory=lambda: time.time_ns())

    def __repr__(self) -> str:
        return (
            f"Fill(qty={self.qty} @{self.price:.4f} "
            f"buyer={self.buy_agent_id[:8]} "
            f"seller={self.sell_agent_id[:8]})"
        )

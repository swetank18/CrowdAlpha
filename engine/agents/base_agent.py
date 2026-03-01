"""
engine/agents/base_agent.py

Standardized interface for all trading agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from engine.core.order import Order


@dataclass(frozen=True)
class MarketState:
    """
    Snapshot delivered to each agent on each tick.
    This is the only market context an agent receives.
    """

    tick: int
    mid_price: Optional[float]
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]

    # Depth levels sorted by priority.
    bid_levels: tuple[tuple[float, int], ...] = field(default_factory=tuple)
    ask_levels: tuple[tuple[float, int], ...] = field(default_factory=tuple)

    # Recent fills from the trade tape.
    recent_trades: tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    # Agent's own position snapshot.
    inventory: int = 0
    cash: float = 0.0
    pnl: float = 0.0

    # Engine-level rolling metrics.
    volatility: float = 0.0
    vwap: float = 0.0
    crowding_intensity: float = 0.0
    agent_crowding_intensity: float = 0.0
    crowding_side_pressure: float = 0.0
    impact_buy_mult: float = 1.0
    impact_sell_mult: float = 1.0
    agent_impact_mult: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bid_levels", tuple(self.bid_levels))
        object.__setattr__(self, "ask_levels", tuple(self.ask_levels))
        object.__setattr__(self, "recent_trades", tuple(self.recent_trades))

    @property
    def mid_or_last(self) -> float:
        if self.mid_price is not None:
            return self.mid_price
        if self.recent_trades:
            return float(self.recent_trades[-1]["price"])
        return 100.0


class BaseAgent(ABC):
    """
    Abstract base for all simulation agents.
    """

    def __init__(
        self,
        agent_id: str,
        strategy_type: str,
        initial_cash: float = 100_000.0,
        **kwargs: Any,
    ) -> None:
        self.agent_id = agent_id
        self.strategy_type = strategy_type
        self.initial_cash = initial_cash

        self.cash: float = initial_cash
        self.inventory: int = 0
        self.pnl: float = 0.0

        self._pnl_history: List[float] = []
        self.config: Dict[str, Any] = kwargs

    @abstractmethod
    def on_tick(self, state: MarketState) -> list:
        """
        Strategy logic.
        Must consume only MarketState and return Order objects.
        """
        ...

    @abstractmethod
    def factor_vector(self) -> np.ndarray:
        """
        Fixed-length behavior vector for crowding analytics.
        """
        ...

    def generate_orders(self, state: MarketState) -> List[Order]:
        """
        Runtime interface guard:
        - input must be MarketState
        - output must be sequence of Order
        - emitted orders must belong to this agent
        """
        if not isinstance(state, MarketState):
            raise TypeError(
                f"{self.agent_id}: on_tick requires MarketState, got {type(state)!r}"
            )

        raw_orders = self.on_tick(state)
        if raw_orders is None:
            return []
        if not isinstance(raw_orders, Sequence):
            raise TypeError(
                f"{self.agent_id}: on_tick must return sequence[Order], got {type(raw_orders)!r}"
            )

        out: List[Order] = []
        for idx, order in enumerate(raw_orders):
            if not isinstance(order, Order):
                raise TypeError(
                    f"{self.agent_id}: item {idx} from on_tick is not Order: {type(order)!r}"
                )
            if order.agent_id != self.agent_id:
                raise ValueError(
                    f"{self.agent_id}: emitted order with mismatched agent_id={order.agent_id!r}"
                )
            out.append(order)
        return out

    def sync_position(self, cash: float, inventory: int, pnl: float) -> None:
        self.cash = cash
        self.inventory = inventory
        self.pnl = pnl
        self._pnl_history.append(pnl)
        if len(self._pnl_history) > 100:
            self._pnl_history.pop(0)

    def rolling_sharpe(self) -> Optional[float]:
        if len(self._pnl_history) < 10:
            return None
        arr = np.array(self._pnl_history)
        returns = np.diff(arr)
        std = returns.std()
        if std == 0:
            return None
        return float(returns.mean() / std)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.agent_id[:8]}, "
            f"inv={self.inventory}, "
            f"cash={self.cash:.0f})"
        )

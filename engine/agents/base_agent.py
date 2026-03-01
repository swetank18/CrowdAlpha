"""
engine/agents/base_agent.py

The standardized interface every agent must implement.

Design constraints (enforced at interface level):
  - An agent receives ONLY a MarketState snapshot per tick
  - It knows nothing outside that snapshot + its own position
  - It returns a list of Orders (may be empty)
  - It cannot reach into the order book, other agents, or global state

This constraint is what makes the simulation fair and what allows
the crowding module to extract meaningful factor vectors from behavior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


# ---------------------------------------------------------------------------
# MarketState — the only information an agent sees
# ---------------------------------------------------------------------------

@dataclass
class MarketState:
    """
    Snapshot of market conditions delivered to each agent at every tick.
    All agents receive the same snapshot (symmetric information).
    """
    tick:          int
    mid_price:     Optional[float]
    best_bid:      Optional[float]
    best_ask:      Optional[float]
    spread:        Optional[float]

    # Depth-of-book: list of (price, qty) tuples, sorted by priority
    bid_levels:    List[tuple] = field(default_factory=list)   # high→low
    ask_levels:    List[tuple] = field(default_factory=list)   # low→high

    # Recent trades (up to last N fills)
    recent_trades: List[Dict] = field(default_factory=list)

    # Agent's own position (injected by simulation.py, not the book)
    inventory:     int   = 0
    cash:          float = 0.0
    pnl:           float = 0.0

    # Rolling metrics computed by simulation
    volatility:    float = 0.0   # realized vol (rolling 20-tick std of returns)
    vwap:          float = 0.0   # volume-weighted average price

    @property
    def mid_or_last(self) -> float:
        """Best guess at current price — mid if available, else last trade."""
        if self.mid_price is not None:
            return self.mid_price
        if self.recent_trades:
            return self.recent_trades[-1]["price"]
        return 100.0  # fallback initial price


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """
    Abstract base for all agents in the simulation.

    Subclasses must implement:
      on_tick(state) -> list[Order]
      factor_vector() -> np.ndarray   (for the crowding module)
    """

    def __init__(
        self,
        agent_id: str,
        strategy_type: str,
        initial_cash: float = 100_000.0,
        **kwargs,
    ) -> None:
        self.agent_id      = agent_id
        self.strategy_type = strategy_type
        self.initial_cash  = initial_cash

        # Internal state (synced from execution engine by simulation.py)
        self.cash:      float = initial_cash
        self.inventory: int   = 0
        self.pnl:       float = 0.0

        # PnL history for Sharpe calculation (per-tick mark-to-market)
        self._pnl_history: List[float] = []

        # Config params passed as kwargs
        self.config: Dict[str, Any] = kwargs

    # ------------------------------------------------------------------
    # Must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def on_tick(self, state: MarketState) -> list:
        """
        Called every tick. Receives market snapshot.
        Returns a list of Order objects (import from engine.core.order).
        MUST NOT modify state or access any global.
        """
        ...

    @abstractmethod
    def factor_vector(self) -> np.ndarray:
        """
        Returns a fixed-length numpy vector describing this agent's
        current behavioral profile. Used by the crowding module.

        Standard factor vector components (length 5):
          [0] momentum_signal      — signed EMA trend, normalized [-1, 1]
          [1] mean_rev_signal      — signed z-score, normalized [-1, 1]
          [2] bid_aggressiveness   — avg distance below mid for bids [0, 1]
          [3] ask_aggressiveness   — avg distance above mid for asks [0, 1]
          [4] turnover_rate        — fills / ticks as fraction [0, 1]
        """
        ...

    # ------------------------------------------------------------------
    # Sync helpers — called by simulation.py, not by the agent itself
    # ------------------------------------------------------------------

    def sync_position(self, cash: float, inventory: int, pnl: float) -> None:
        """Update internal state from execution engine."""
        self.cash      = cash
        self.inventory = inventory
        self.pnl       = pnl
        self._pnl_history.append(pnl)
        # Keep rolling window of 100 ticks
        if len(self._pnl_history) > 100:
            self._pnl_history.pop(0)

    def rolling_sharpe(self) -> Optional[float]:
        """Rolling Sharpe ratio from stored PnL history."""
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

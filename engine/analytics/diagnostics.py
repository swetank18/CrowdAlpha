"""
engine/analytics/diagnostics.py

Per-agent performance diagnostics: Sharpe, Sortino, max drawdown,
skewness, kurtosis, hit rate, average fill size.

All metrics operate on per-tick PnL histories streamed from simulation.py.
Designed to be called each tick (cheap incremental updates) or on-demand.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Agent diagnostics snapshot
# ---------------------------------------------------------------------------

@dataclass
class AgentDiagnostics:
    agent_id:      str
    strategy_type: str
    pnl:           float = 0.0
    sharpe:        Optional[float] = None
    sortino:       Optional[float] = None
    max_drawdown:  float = 0.0
    skewness:      float = 0.0
    kurtosis:      float = 0.0
    hit_rate:      float = 0.0     # fraction of ticks with positive return
    avg_fill_size: float = 0.0
    fill_count:    int   = 0
    inventory:     int   = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id":      self.agent_id,
            "strategy_type": self.strategy_type,
            "pnl":           round(self.pnl, 2),
            "sharpe":        round(self.sharpe, 4) if self.sharpe else None,
            "sortino":       round(self.sortino, 4) if self.sortino else None,
            "max_drawdown":  round(self.max_drawdown, 4),
            "skewness":      round(self.skewness, 4),
            "kurtosis":      round(self.kurtosis, 4),
            "hit_rate":      round(self.hit_rate, 4),
            "avg_fill_size": round(self.avg_fill_size, 2),
            "fill_count":    self.fill_count,
            "inventory":     self.inventory,
        }


# ---------------------------------------------------------------------------
# Diagnostics engine
# ---------------------------------------------------------------------------

class Diagnostics:

    RISK_FREE = 0.0   # per-tick risk-free rate

    def __init__(self) -> None:
        self._pnl_history: Dict[str, List[float]] = {}
        self._peak_pnl:    Dict[str, float]       = {}
        self._max_dd:      Dict[str, float]       = {}

    def update(self, agents: List["BaseAgent"]) -> None:
        """Record current PnL for each agent. Called every tick."""
        for agent in agents:
            aid = agent.agent_id
            if aid not in self._pnl_history:
                self._pnl_history[aid] = []
                self._peak_pnl[aid]    = agent.pnl
                self._max_dd[aid]      = 0.0

            self._pnl_history[aid].append(agent.pnl)

            # Rolling max drawdown
            if agent.pnl > self._peak_pnl[aid]:
                self._peak_pnl[aid] = agent.pnl
            dd = self._peak_pnl[aid] - agent.pnl
            if dd > self._max_dd[aid]:
                self._max_dd[aid] = dd

            # Keep bounded
            if len(self._pnl_history[aid]) > 1000:
                self._pnl_history[aid].pop(0)

    def compute(self, agent: "BaseAgent") -> AgentDiagnostics:
        """Compute full diagnostics snapshot for one agent."""
        aid = agent.agent_id
        history = self._pnl_history.get(aid, [])

        diag = AgentDiagnostics(
            agent_id      = aid,
            strategy_type = agent.strategy_type,
            pnl           = agent.pnl,
            fill_count    = getattr(agent, 'fill_count', 0),
            inventory     = agent.inventory,
            max_drawdown  = self._max_dd.get(aid, 0.0),
        )

        if len(history) < 2:
            return diag

        arr = np.array(history)
        returns = np.diff(arr)
        if len(returns) == 0:
            return diag

        mean_r = returns.mean()
        std_r  = returns.std()

        # Sharpe
        if std_r > 1e-9:
            diag.sharpe = float((mean_r - self.RISK_FREE) / std_r)

        # Sortino (downside deviation only)
        downside = returns[returns < 0]
        if len(downside) > 0:
            down_std = downside.std()
            if down_std > 1e-9:
                diag.sortino = float((mean_r - self.RISK_FREE) / down_std)

        # Skewness and excess kurtosis
        if std_r > 1e-9 and len(returns) >= 3:
            diag.skewness = float(((returns - mean_r) ** 3).mean() / std_r ** 3)
            diag.kurtosis = float(((returns - mean_r) ** 4).mean() / std_r ** 4 - 3)

        # Hit rate
        diag.hit_rate = float((returns > 0).mean())

        return diag

    def snapshot_all(self, agents: List["BaseAgent"]) -> List[Dict]:
        """Compute diagnostics for all agents, sorted by Sharpe desc."""
        diags = [self.compute(a).to_dict() for a in agents]
        diags.sort(key=lambda d: (d["sharpe"] or -999), reverse=True)
        return diags

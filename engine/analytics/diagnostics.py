"""
engine/analytics/diagnostics.py

Trade-stream diagnostics computed from observable fills and mark-to-market path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from engine.core.order import Fill

if TYPE_CHECKING:
    from engine.agents.base_agent import BaseAgent


@dataclass
class AgentDiagnostics:
    agent_id: str
    strategy_type: str
    pnl: float = 0.0
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    max_drawdown: float = 0.0
    turnover: float = 0.0
    spread_cost: float = 0.0
    market_impact_cost: float = 0.0
    kurtosis: float = 0.0
    vol_autocorr: float = 0.0
    hit_rate: float = 0.0
    avg_fill_size: float = 0.0
    fill_count: int = 0
    inventory: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "strategy_type": self.strategy_type,
            "pnl": round(self.pnl, 2),
            "sharpe": round(self.sharpe, 4) if self.sharpe is not None else None,
            "sortino": round(self.sortino, 4) if self.sortino is not None else None,
            "max_drawdown": round(self.max_drawdown, 4),
            "turnover": round(self.turnover, 4),
            "spread_cost": round(self.spread_cost, 4),
            "market_impact_cost": round(self.market_impact_cost, 4),
            "kurtosis": round(self.kurtosis, 4),
            "vol_autocorr": round(self.vol_autocorr, 4),
            "hit_rate": round(self.hit_rate, 4),
            "avg_fill_size": round(self.avg_fill_size, 2),
            "fill_count": self.fill_count,
            "inventory": self.inventory,
        }


class Diagnostics:
    def __init__(
        self,
        window: int = 250,
        min_sharpe_ticks: int = 50,
        ticks_per_year: float = 31_536_000.0,
    ) -> None:
        self.window = max(30, window)
        self.min_sharpe_ticks = max(50, int(min_sharpe_ticks))
        self.ticks_per_year = max(float(ticks_per_year), 1.0)
        self._cash: Dict[str, float] = {}
        self._inventory: Dict[str, int] = {}
        self._pnl_history: Dict[str, List[float]] = {}
        self._trade_qty_history: Dict[str, List[float]] = {}
        self._inv_abs_history: Dict[str, List[float]] = {}
        self._spread_cost: Dict[str, float] = {}
        self._impact_cost: Dict[str, float] = {}
        self._fill_count: Dict[str, int] = {}
        self._fill_qty: Dict[str, float] = {}
        self._strategy_type: Dict[str, str] = {}

    def update(
        self,
        agents: List["BaseAgent"],
        fills: List[Fill],
        mid_pre: float,
        mid_post: float,
    ) -> None:
        for agent in agents:
            aid = agent.agent_id
            if aid not in self._cash:
                self._cash[aid] = float(agent.initial_cash)
                self._inventory[aid] = 0
                self._pnl_history[aid] = []
                self._trade_qty_history[aid] = []
                self._inv_abs_history[aid] = []
                self._spread_cost[aid] = 0.0
                self._impact_cost[aid] = 0.0
                self._fill_count[aid] = 0
                self._fill_qty[aid] = 0.0
            self._strategy_type[aid] = agent.strategy_type

        traded_this_tick: Dict[str, float] = {a.agent_id: 0.0 for a in agents}
        mid_delta = float(mid_post - mid_pre)

        for fill in fills:
            buyer = fill.buy_agent_id
            seller = fill.sell_agent_id
            qty = int(fill.qty)
            price = float(fill.price)

            for aid in (buyer, seller):
                if aid not in self._cash:
                    # Safety for agents not present in current list.
                    self._cash[aid] = 100_000.0
                    self._inventory[aid] = 0
                    self._pnl_history[aid] = []
                    self._trade_qty_history[aid] = []
                    self._inv_abs_history[aid] = []
                    self._spread_cost[aid] = 0.0
                    self._impact_cost[aid] = 0.0
                    self._fill_count[aid] = 0
                    self._fill_qty[aid] = 0.0
                    self._strategy_type.setdefault(aid, "unknown")
                traded_this_tick.setdefault(aid, 0.0)

            # Fill-accounting from trade stream.
            self._cash[buyer] -= price * qty
            self._inventory[buyer] += qty
            self._cash[seller] += price * qty
            self._inventory[seller] -= qty

            traded_this_tick[buyer] += qty
            traded_this_tick[seller] += qty
            self._fill_count[buyer] += 1
            self._fill_count[seller] += 1
            self._fill_qty[buyer] += qty
            self._fill_qty[seller] += qty

            # Cost decomposition around observed pre/post mid.
            self._spread_cost[buyer] += max(price - mid_pre, 0.0) * qty
            self._spread_cost[seller] += max(mid_pre - price, 0.0) * qty
            self._impact_cost[buyer] += max(mid_delta, 0.0) * qty
            self._impact_cost[seller] += max(-mid_delta, 0.0) * qty

        for aid in list(self._cash.keys()):
            pnl = self._cash[aid] + self._inventory[aid] * mid_post
            self._pnl_history.setdefault(aid, []).append(float(pnl))
            self._trade_qty_history.setdefault(aid, []).append(float(traded_this_tick.get(aid, 0.0)))
            self._inv_abs_history.setdefault(aid, []).append(float(abs(self._inventory[aid])))

            if len(self._pnl_history[aid]) > self.window:
                self._pnl_history[aid].pop(0)
            if len(self._trade_qty_history[aid]) > self.window:
                self._trade_qty_history[aid].pop(0)
            if len(self._inv_abs_history[aid]) > self.window:
                self._inv_abs_history[aid].pop(0)

    def compute(self, agent: "BaseAgent") -> AgentDiagnostics:
        aid = agent.agent_id
        return self.compute_by_id(aid, fallback_strategy=agent.strategy_type)

    def compute_by_id(self, agent_id: str, fallback_strategy: str = "unknown") -> AgentDiagnostics:
        hist = self._pnl_history.get(agent_id, [])
        strategy = self._strategy_type.get(agent_id, fallback_strategy)
        diag = AgentDiagnostics(
            agent_id=agent_id,
            strategy_type=strategy,
            pnl=float(hist[-1]) if hist else 0.0,
            fill_count=int(self._fill_count.get(agent_id, 0)),
            inventory=int(self._inventory.get(agent_id, 0)),
            spread_cost=float(self._spread_cost.get(agent_id, 0.0)),
            market_impact_cost=float(self._impact_cost.get(agent_id, 0.0)),
            avg_fill_size=(
                float(self._fill_qty.get(agent_id, 0.0)) / max(float(self._fill_count.get(agent_id, 0)), 1.0)
            ),
        )

        if len(hist) < 3:
            return diag

        arr = np.array(hist, dtype=float)
        rets = np.diff(arr)
        if len(rets) < self.min_sharpe_ticks:
            return diag

        mean_r = float(np.mean(rets))
        std_r = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
        if std_r > 1e-9:
            diag.sharpe = (mean_r / std_r) * float(np.sqrt(self.ticks_per_year))

        downside = rets[rets < 0.0]
        if len(downside) > 1:
            down_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
            if down_std > 1e-9:
                diag.sortino = (mean_r / down_std) * float(np.sqrt(self.ticks_per_year))

        equity = arr
        running_max = np.maximum.accumulate(equity)
        dd = running_max - equity
        diag.max_drawdown = float(np.max(dd)) if len(dd) else 0.0

        diag.hit_rate = float(np.mean(rets > 0.0))

        if len(rets) >= 4 and std_r > 1e-9:
            centered = rets - mean_r
            m2 = float(np.mean(centered**2))
            m4 = float(np.mean(centered**4))
            diag.kurtosis = (m4 / (m2 * m2) - 3.0) if m2 > 1e-12 else 0.0

        abs_rets = np.abs(rets)
        if len(abs_rets) >= 3:
            x = abs_rets[:-1]
            y = abs_rets[1:]
            if float(np.std(x)) > 1e-12 and float(np.std(y)) > 1e-12:
                diag.vol_autocorr = float(np.corrcoef(x, y)[0, 1])

        qty_hist = np.array(self._trade_qty_history.get(agent_id, []), dtype=float)
        inv_hist = np.array(self._inv_abs_history.get(agent_id, []), dtype=float)
        if len(qty_hist) > 0 and len(inv_hist) > 0:
            avg_qty = float(np.mean(qty_hist))
            avg_inv = float(np.mean(inv_hist))
            diag.turnover = avg_qty / max(avg_inv, 1.0)

        return diag

    def snapshot_all(self, agents: List["BaseAgent"]) -> List[Dict[str, Any]]:
        out = [self.compute(agent).to_dict() for agent in agents]
        out.sort(key=lambda x: (x["sharpe"] if x["sharpe"] is not None else -999.0), reverse=True)
        return out

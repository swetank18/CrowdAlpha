"""
engine/crowding/alpha_decay.py

Crowding-driven alpha decay and slippage amplification model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from engine.agents.base_agent import BaseAgent


@dataclass
class DecayParams:
    agent_id: str
    alpha_max: float
    lambda_: float
    half_life: Optional[float]
    r_squared: float
    n_samples: int


class AlphaDecay:
    def __init__(
        self,
        min_samples: int = 20,
        slippage_scale: float = 1.6,
        side_scale: float = 0.9,
        max_impact_mult: float = 4.0,
    ) -> None:
        self.min_samples = min_samples
        self.slippage_scale = slippage_scale
        self.side_scale = side_scale
        self.max_impact_mult = max_impact_mult

        self._crowding_history: List[float] = []
        self._side_pressure_history: List[float] = []
        self._agent_crowding: Dict[str, List[float]] = {}
        self._agent_sharpe: Dict[str, List[float]] = {}
        self._params: Dict[str, DecayParams] = {}

        self._impact_buy_mult: float = 1.0
        self._impact_sell_mult: float = 1.0
        self._side_pressure: float = 0.0

    def update(
        self,
        crowding_intensity: float,
        agents: List["BaseAgent"],
        agent_activity: Optional[Dict[str, float]] = None,
        order_flow_imbalance: float = 0.0,
    ) -> None:
        crowd = float(np.clip(crowding_intensity, -1.0, 1.0))
        self._crowding_history.append(crowd)
        if len(self._crowding_history) > 1000:
            self._crowding_history.pop(0)

        activity = agent_activity or {}
        self._side_pressure = float(np.clip(order_flow_imbalance * max(crowd, 0.0), -1.0, 1.0))
        self._side_pressure_history.append(self._side_pressure)
        if len(self._side_pressure_history) > 1000:
            self._side_pressure_history.pop(0)

        self._update_impact_multipliers(crowd)

        for agent in agents:
            sharpe = agent.rolling_sharpe()
            if sharpe is None:
                continue

            agent_id = agent.agent_id
            a = float(np.clip(activity.get(agent_id, 0.0), 0.0, 1.0))
            experienced = float(np.clip(max(crowd, 0.0) * (0.5 + 0.5 * a), 0.0, 1.0))

            # Alpha proxy for exponential fit must be positive.
            alpha_proxy = max(float(sharpe), 1e-3)

            self._agent_crowding.setdefault(agent_id, []).append(experienced)
            self._agent_sharpe.setdefault(agent_id, []).append(alpha_proxy)

            if len(self._agent_crowding[agent_id]) > 1000:
                self._agent_crowding[agent_id].pop(0)
            if len(self._agent_sharpe[agent_id]) > 1000:
                self._agent_sharpe[agent_id].pop(0)

        self._fit_all()

    def current_impact_multipliers(self) -> tuple[float, float]:
        return self._impact_buy_mult, self._impact_sell_mult

    @property
    def side_pressure(self) -> float:
        return self._side_pressure

    def _update_impact_multipliers(self, crowd: float) -> None:
        base_amp = 1.0 + self.slippage_scale * (max(crowd, 0.0) ** 2)
        buy_side = 1.0 + self.side_scale * max(self._side_pressure, 0.0)
        sell_side = 1.0 + self.side_scale * max(-self._side_pressure, 0.0)

        buy = base_amp * buy_side
        sell = base_amp * sell_side

        self._impact_buy_mult = float(np.clip(buy, 1.0, self.max_impact_mult))
        self._impact_sell_mult = float(np.clip(sell, 1.0, self.max_impact_mult))

    def _fit_all(self) -> None:
        for agent_id, s_hist in self._agent_sharpe.items():
            c_hist = self._agent_crowding.get(agent_id, [])
            n = min(len(s_hist), len(c_hist))
            if n < self.min_samples:
                continue

            c = np.array(c_hist[-n:], dtype=float)
            s = np.array(s_hist[-n:], dtype=float)

            params = self._fit_exponential(agent_id, c, s)
            if params is not None:
                self._params[agent_id] = params

    def _fit_exponential(
        self,
        agent_id: str,
        crowding: np.ndarray,
        alpha_proxy: np.ndarray,
    ) -> Optional[DecayParams]:
        valid = alpha_proxy > 1e-6
        if int(np.sum(valid)) < max(6, self.min_samples // 2):
            return None

        c = crowding[valid]
        y = np.log(alpha_proxy[valid])

        try:
            x = np.column_stack([np.ones(len(c)), c])
            coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        except np.linalg.LinAlgError:
            return None

        b0, b1 = float(coeffs[0]), float(coeffs[1])
        alpha_max = max(math.exp(b0), 1e-6)
        lambda_ = max(-b1, 0.0)

        pred = x @ coeffs
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        half_life = (math.log(2.0) / lambda_) if lambda_ > 1e-8 else None
        return DecayParams(
            agent_id=agent_id,
            alpha_max=round(alpha_max, 4),
            lambda_=round(lambda_, 4),
            half_life=round(half_life, 2) if half_life is not None else None,
            r_squared=round(float(np.clip(r2, -1.0, 1.0)), 4),
            n_samples=int(np.sum(valid)),
        )

    def decay_curve(self, agent_id: str, n_points: int = 50) -> List[Dict[str, float]]:
        p = self._params.get(agent_id)
        if p is None:
            return []
        xs = np.linspace(0.0, 1.0, n_points)
        ys = p.alpha_max * np.exp(-p.lambda_ * xs)
        return [
            {"crowding": round(float(x), 4), "predicted_sharpe": round(float(y), 4)}
            for x, y in zip(xs, ys)
        ]

    def snapshot_for_api(self) -> Dict[str, Any]:
        return {
            "crowding_intensity_history": [round(float(x), 4) for x in self._crowding_history[-100:]],
            "side_pressure_history": [round(float(x), 4) for x in self._side_pressure_history[-100:]],
            "impact_multipliers": {
                "buy": round(self._impact_buy_mult, 4),
                "sell": round(self._impact_sell_mult, 4),
                "side_pressure": round(self._side_pressure, 4),
            },
            "agent_decay_params": [
                {
                    "agent_id": p.agent_id,
                    "alpha_max": p.alpha_max,
                    "lambda": p.lambda_,
                    "half_life": p.half_life,
                    "r_squared": p.r_squared,
                    "n_samples": p.n_samples,
                    "curve": self.decay_curve(p.agent_id, n_points=20),
                }
                for p in self._params.values()
            ],
        }


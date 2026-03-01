"""
engine/crowding/alpha_decay.py

Models the causal relationship between crowding intensity and Sharpe ratio.

Core model (per agent):
    Sharpe(t) = α_max × exp(-λ × crowding(t))

Where:
    α_max  = maximum Sharpe (achievable when crowding → 0)
    λ      = half-life decay coefficient (the key academic finding)
    crowding(t) = crowding intensity at time t ∈ [0, 1]

Half-life: when crowding doubles the decay, Sharpe halves.
    t½ = ln(2) / λ

A flat λ ≈ 0 means the strategy is crowding-resistant.
A high λ means it degrades rapidly as the market becomes crowded.

This module fits the model from observed data (rolling Sharpe + crowding
intensity history) and generates the decay curves for visualization.

Fitting method: least-squares exponential fit via log-linearization.
    log(Sharpe) = log(α_max) - λ × crowding
    → linear regression on (crowding, log(Sharpe))
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DecayParams:
    """Fitted decay model for one agent."""
    agent_id:  str
    alpha_max: float          # max Sharpe at zero crowding
    lambda_:   float          # decay coefficient
    half_life: Optional[float]  # ticks until Sharpe halves; None if λ ≈ 0
    r_squared: float          # goodness of fit ∈ [0, 1]
    n_samples: int


# ---------------------------------------------------------------------------
# AlphaDecay
# ---------------------------------------------------------------------------

class AlphaDecay:

    def __init__(self, min_samples: int = 20) -> None:
        self.min_samples = min_samples

        # Rolling parallel history: (crowding_intensity, per_agent_sharpe)
        self._crowding_history: List[float] = []
        self._sharpe_history:   Dict[str, List[float]] = {}  # agent_id → sharpes

        # Fitted params
        self._params: Dict[str, DecayParams] = {}

    # ------------------------------------------------------------------
    # Update (called each tick by simulation.py)
    # ------------------------------------------------------------------

    def update(
        self,
        crowding_intensity: float,
        agents: List["BaseAgent"],
    ) -> None:
        """
        Record this tick's crowding intensity and each agent's rolling Sharpe.
        Refits the decay model every tick (cheap because we have few agents).
        """
        self._crowding_history.append(crowding_intensity)
        if len(self._crowding_history) > 1000:
            self._crowding_history.pop(0)

        for agent in agents:
            sharpe = agent.rolling_sharpe()
            if sharpe is None:
                continue
            if agent.agent_id not in self._sharpe_history:
                self._sharpe_history[agent.agent_id] = []
            self._sharpe_history[agent.agent_id].append(sharpe)
            if len(self._sharpe_history[agent.agent_id]) > 1000:
                self._sharpe_history[agent.agent_id].pop(0)

        # Refit if we have enough data
        if len(self._crowding_history) >= self.min_samples:
            self._fit_all()

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def _fit_all(self) -> None:
        """Fit exponential decay model for every agent with enough data."""
        crowd = np.array(self._crowding_history)

        for agent_id, sharpes in self._sharpe_history.items():
            if len(sharpes) < self.min_samples:
                continue

            # Align lengths (crowd and sharpe may differ if agent joined late)
            n = min(len(crowd), len(sharpes))
            c = crowd[-n:]
            s = np.array(sharpes[-n:])

            params = self._fit_exponential(agent_id, c, s)
            if params:
                self._params[agent_id] = params

    def _fit_exponential(
        self, agent_id: str, crowding: np.ndarray, sharpe: np.ndarray
    ) -> Optional[DecayParams]:
        """
        Fit: log(|Sharpe|) = log(α_max) - λ × crowding
        via least squares.
        """
        # Filter out non-positive Sharpe (log undefined)
        valid = sharpe > 0
        if valid.sum() < 5:
            return None

        c_valid = crowding[valid]
        s_valid = sharpe[valid]

        log_s = np.log(s_valid)

        # Linear regression: log_s = b0 - lambda * crowding
        try:
            X = np.column_stack([np.ones(len(c_valid)), c_valid])
            coeffs, residuals, _, _ = np.linalg.lstsq(X, log_s, rcond=None)
        except np.linalg.LinAlgError:
            return None

        b0, neg_lambda = coeffs
        alpha_max = math.exp(b0)
        lambda_   = max(-neg_lambda, 0.0)  # force non-negative decay

        # R² for goodness of fit
        ss_res  = np.sum((log_s - (X @ coeffs)) ** 2)
        ss_tot  = np.sum((log_s - log_s.mean()) ** 2)
        r2      = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.0

        half_life = (math.log(2) / lambda_) if lambda_ > 1e-6 else None

        return DecayParams(
            agent_id  = agent_id,
            alpha_max = round(alpha_max, 4),
            lambda_   = round(lambda_, 4),
            half_life = round(half_life, 2) if half_life else None,
            r_squared = round(r2, 4),
            n_samples = len(c_valid),
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, DecayParams]:
        return dict(self._params)

    def decay_curve(self, agent_id: str, n_points: int = 50) -> List[Dict]:
        """
        Generate model prediction curve for plotting:
          crowding ∈ [0, 1] → predicted Sharpe
        Returns list of {crowding, predicted_sharpe} dicts.
        """
        params = self._params.get(agent_id)
        if params is None:
            return []

        crowding_vals = np.linspace(0.0, 1.0, n_points)
        predicted = params.alpha_max * np.exp(-params.lambda_ * crowding_vals)

        return [
            {"crowding": round(float(c), 4), "predicted_sharpe": round(float(s), 4)}
            for c, s in zip(crowding_vals, predicted)
        ]

    def snapshot_for_api(self) -> Dict[str, Any]:
        """Serializable snapshot for analytics API."""
        return {
            "crowding_intensity_history": [
                round(x, 4) for x in self._crowding_history[-100:]
            ],
            "agent_decay_params": [
                {
                    "agent_id":  p.agent_id,
                    "alpha_max": p.alpha_max,
                    "lambda":    p.lambda_,
                    "half_life": p.half_life,
                    "r_squared": p.r_squared,
                    "curve":     self.decay_curve(p.agent_id, n_points=20),
                }
                for p in self._params.values()
            ],
        }

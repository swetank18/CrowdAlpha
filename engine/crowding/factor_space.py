"""
engine/crowding/factor_space.py

Extract behavioral factor vectors from agents over a rolling window.

The factor matrix F ∈ ℝ^(N_agents × N_factors) is the input to the
crowding module. Each row is an agent's behavioral fingerprint.

Standard factor vector (length 5, all normalized to [-1, 1] or [0, 1]):
  [0] momentum_signal      — directional EMA trend signal
  [1] mean_rev_signal      — signed z-score signal
  [2] bid_aggressiveness   — how far inside spread bids are placed
  [3] ask_aggressiveness   — how far inside spread asks are placed
  [4] turnover_rate        — fills per tick

The factor space is also used to produce a 2D PCA projection for
the frontend FactorSpace scatter plot.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.agents.base_agent import BaseAgent


# Number of factors in the standard factor vector — must match base_agent.py
N_FACTORS = 5
FACTOR_NAMES = [
    "momentum_signal",
    "mean_rev_signal",
    "bid_aggressiveness",
    "ask_aggressiveness",
    "turnover_rate",
]


class FactorSpace:
    """
    Collects and stores factor vectors from all agents.
    Called once per tick by simulation.py.

    Also maintains a rolling history for trend analysis and
    computes PCA projection for visualization.
    """

    def __init__(self, window: int = 50) -> None:
        self.window = window
        # agent_id → list of factor vectors (rolling history)
        self._history: Dict[str, List[np.ndarray]] = {}
        # Latest factor matrix
        self._current_matrix: Optional[np.ndarray] = None
        self._agent_ids: List[str] = []

    def update(self, agents: List["BaseAgent"]) -> np.ndarray:
        """
        Call every tick. Collects factor vectors from all agents,
        stores rolling history, returns current factor matrix.

        Returns:
            F: np.ndarray of shape (N_agents, N_FACTORS)
        """
        vectors = []
        ids     = []

        for agent in agents:
            vec = agent.factor_vector()
            if len(vec) != N_FACTORS:
                raise ValueError(
                    f"Agent {agent.agent_id} returned factor vector of length "
                    f"{len(vec)}, expected {N_FACTORS}"
                )
            # Clamp to valid range
            vec = np.clip(vec, -1.0, 1.0)

            if agent.agent_id not in self._history:
                self._history[agent.agent_id] = []
            self._history[agent.agent_id].append(vec)
            if len(self._history[agent.agent_id]) > self.window:
                self._history[agent.agent_id].pop(0)

            vectors.append(vec)
            ids.append(agent.agent_id)

        self._agent_ids = ids
        self._current_matrix = np.array(vectors)  # shape: (N, 5)
        return self._current_matrix

    def current_matrix(self) -> Optional[np.ndarray]:
        """Latest factor matrix, shape (N_agents, N_FACTORS)."""
        return self._current_matrix

    def pca_projection(self) -> Optional[np.ndarray]:
        """
        2D PCA projection of the current factor matrix for visualization.
        Returns array of shape (N_agents, 2), or None if fewer than 2 agents.
        """
        F = self._current_matrix
        if F is None or F.shape[0] < 2:
            return None

        # Center
        centered = F - F.mean(axis=0)

        # SVD-based PCA
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            projection = centered @ Vt[:2].T  # (N, 2)
            return projection
        except np.linalg.LinAlgError:
            return None

    def snapshot_for_api(self) -> Dict[str, Any]:
        """
        Serializable snapshot for the analytics API endpoint.
        """
        F = self._current_matrix
        if F is None:
            return {"agents": [], "factor_names": FACTOR_NAMES}

        pca = self.pca_projection()
        agents_out = []
        for i, agent_id in enumerate(self._agent_ids):
            entry: Dict[str, Any] = {
                "agent_id": agent_id,
                "factors": dict(zip(FACTOR_NAMES, F[i].tolist())),
            }
            if pca is not None:
                entry["pca"] = {"x": float(pca[i, 0]), "y": float(pca[i, 1])}
            agents_out.append(entry)

        return {
            "agents":       agents_out,
            "factor_names": FACTOR_NAMES,
        }

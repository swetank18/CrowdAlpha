"""
engine/crowding/crowding_matrix.py

Pairwise cosine similarity between all agents' factor vectors.

The crowding matrix C ∈ ℝ^(N×N) where C[i,j] is the cosine similarity
between agent i's and agent j's factor vectors.

  C[i,j] = (F[i] · F[j]) / (|F[i]| × |F[j]|)

Diagonal is always 1 (an agent is perfectly similar to itself).
Off-diagonal values ∈ [-1, 1]:
  +1  = strategies are identical (fully crowded)
   0  = orthogonal (uncorrelated behavior)
  -1  = opposite strategies (e.g., pure buyer vs pure seller)

Crowding intensity (scalar):
  The mean of the upper triangle of C (excluding diagonal).
  Range: [-1, 1], higher = more crowded market.

Academic note:
  This is a market-level crowding metric, different from
  portfolio-level crowding measures (e.g., return correlation).
  It measures BEHAVIORAL similarity, which is the causal precursor
  to return correlation and the alpha decay modeled in alpha_decay.py.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict, Any


class CrowdingMatrix:

    def __init__(self) -> None:
        self._matrix:    Optional[np.ndarray] = None
        self._intensity: float = 0.0
        self._agent_ids: List[str] = []
        self._intensity_history: List[float] = []  # rolling for alpha_decay

    # ------------------------------------------------------------------

    def update(self, factor_matrix: np.ndarray, agent_ids: List[str]) -> float:
        """
        Compute pairwise cosine similarity from factor matrix.

        Args:
            factor_matrix: shape (N_agents, N_factors)
            agent_ids: list of agent ids, same order as rows

        Returns:
            crowding_intensity: scalar ∈ [-1, 1]
        """
        self._agent_ids = agent_ids
        N = factor_matrix.shape[0]

        if N == 0:
            self._matrix    = np.zeros((0, 0))
            self._intensity = 0.0
            return 0.0

        # L2-normalize each row
        norms = np.linalg.norm(factor_matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)  # avoid div-by-zero
        F_norm = factor_matrix / norms

        # Cosine similarity matrix (N×N)
        C = F_norm @ F_norm.T
        C = np.clip(C, -1.0, 1.0)

        self._matrix = C

        # Crowding intensity = mean of upper triangle (exclude diagonal)
        if N > 1:
            iu = np.triu_indices(N, k=1)
            self._intensity = float(np.mean(C[iu]))
        else:
            self._intensity = 1.0  # single agent → fully "crowded" with itself

        self._intensity_history.append(self._intensity)
        if len(self._intensity_history) > 500:
            self._intensity_history.pop(0)

        return self._intensity

    @property
    def intensity(self) -> float:
        return self._intensity

    @property
    def matrix(self) -> Optional[np.ndarray]:
        return self._matrix

    @property
    def intensity_history(self) -> List[float]:
        return list(self._intensity_history)

    # ------------------------------------------------------------------

    def top_pairs(self, n: int = 5) -> List[Dict]:
        """
        Return the n most-crowded agent pairs (highest cosine similarity,
        excluding self-pairs). Useful for API + UI highlighting.
        """
        C = self._matrix
        ids = self._agent_ids
        if C is None or len(ids) < 2:
            return []

        N = C.shape[0]
        pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                pairs.append({
                    "agent_a":    ids[i],
                    "agent_b":    ids[j],
                    "similarity": round(float(C[i, j]), 4),
                })

        pairs.sort(key=lambda x: -x["similarity"])
        return pairs[:n]

    def snapshot_for_api(self) -> Dict[str, Any]:
        """Serializable snapshot for analytics API."""
        C = self._matrix
        return {
            "agent_ids":           self._agent_ids,
            "matrix":              C.tolist() if C is not None else [],
            "crowding_intensity":  round(self._intensity, 4),
            "top_crowded_pairs":   self.top_pairs(5),
        }

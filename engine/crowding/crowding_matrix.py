"""
engine/crowding/crowding_matrix.py

Pairwise cosine similarity and activity-weighted crowding intensity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class CrowdingMatrix:
    def __init__(self) -> None:
        self._matrix: Optional[np.ndarray] = None
        self._intensity: float = 0.0
        self._agent_ids: List[str] = []
        self._activity_weights: List[float] = []
        self._agent_intensity: Dict[str, float] = {}
        self._intensity_history: List[float] = []

    def update(
        self,
        factor_matrix: np.ndarray,
        agent_ids: List[str],
        activity_weights: Optional[List[float]] = None,
    ) -> float:
        self._agent_ids = list(agent_ids)
        n = factor_matrix.shape[0]

        if n == 0:
            self._matrix = np.zeros((0, 0), dtype=float)
            self._intensity = 0.0
            self._activity_weights = []
            self._agent_intensity = {}
            return 0.0

        norms = np.linalg.norm(factor_matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        fn = factor_matrix / norms
        c = np.clip(fn @ fn.T, -1.0, 1.0)
        self._matrix = c

        if activity_weights is None or len(activity_weights) != n:
            w = np.ones(n, dtype=float) / float(n)
        else:
            w = np.array(activity_weights, dtype=float)
            w = np.where(w < 0.0, 0.0, w)
            sw = float(np.sum(w))
            if sw < 1e-9:
                w = np.ones(n, dtype=float) / float(n)
            else:
                w = w / sw

        self._activity_weights = w.tolist()

        # Agent-level crowding:
        # Phi_i = (1/(N-1)) * sum_{j!=i} C_ij * w_j
        # where w_j is recent trading volume share.
        phi = np.zeros(n, dtype=float)
        if n > 1:
            for i in range(n):
                numer = float(np.sum(c[i, :] * w) - (c[i, i] * w[i]))
                phi[i] = numer / float(n - 1)
        self._agent_intensity = {
            aid: float(np.clip(phi[idx], -1.0, 1.0))
            for idx, aid in enumerate(self._agent_ids)
        }

        if n == 1:
            self._intensity = float(phi[0])
        else:
            self._intensity = float(np.mean(phi))

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

    @property
    def agent_intensity_map(self) -> Dict[str, float]:
        return dict(self._agent_intensity)

    def impact_multipliers(
        self,
        kappa: float,
        min_mult: float = 0.2,
        max_mult: float = 6.0,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for aid, phi in self._agent_intensity.items():
            mult = 1.0 + float(kappa) * float(phi)
            out[aid] = float(np.clip(mult, min_mult, max_mult))
        return out

    def top_pairs(self, n: int = 5) -> List[Dict[str, Any]]:
        c = self._matrix
        ids = self._agent_ids
        if c is None or len(ids) < 2:
            return []

        out: List[Dict[str, Any]] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                out.append(
                    {
                        "agent_a": ids[i],
                        "agent_b": ids[j],
                        "similarity": round(float(c[i, j]), 4),
                        "pair_activity": round(
                            float(self._activity_weights[i] * self._activity_weights[j])
                            if i < len(self._activity_weights) and j < len(self._activity_weights)
                            else 0.0,
                            4,
                        ),
                    }
                )
        out.sort(key=lambda x: (-x["similarity"], -x["pair_activity"]))
        return out[:n]

    def snapshot_for_api(self) -> Dict[str, Any]:
        c = self._matrix
        return {
            "agent_ids": self._agent_ids,
            "matrix": c.tolist() if c is not None else [],
            "activity_weights": [round(float(w), 4) for w in self._activity_weights],
            "crowding_intensity": round(self._intensity, 4),
            "agent_crowding_intensity": {
                k: round(float(v), 6) for k, v in self._agent_intensity.items()
            },
            "top_crowded_pairs": self.top_pairs(5),
        }

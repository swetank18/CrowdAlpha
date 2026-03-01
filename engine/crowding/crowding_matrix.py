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
            return 0.0

        norms = np.linalg.norm(factor_matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        fn = factor_matrix / norms
        c = np.clip(fn @ fn.T, -1.0, 1.0)
        self._matrix = c

        if activity_weights is None or len(activity_weights) != n:
            w = np.ones(n, dtype=float)
        else:
            w = np.array(activity_weights, dtype=float)
            w = np.where(w < 0.0, 0.0, w)
            if float(np.sum(w)) < 1e-9:
                w = np.ones(n, dtype=float)

        self._activity_weights = w.tolist()

        if n == 1:
            self._intensity = 1.0
        else:
            iu = np.triu_indices(n, k=1)
            pair_w = (w[iu[0]] * w[iu[1]]).astype(float)
            denom = float(np.sum(pair_w))
            if denom < 1e-9:
                self._intensity = float(np.mean(c[iu]))
            else:
                self._intensity = float(np.sum(pair_w * c[iu]) / denom)

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
            "top_crowded_pairs": self.top_pairs(5),
        }


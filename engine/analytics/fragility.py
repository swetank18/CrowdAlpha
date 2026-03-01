"""
engine/analytics/fragility.py

Liquidity Fragility Index (LFI) — a composite score that measures
how close the market is to a liquidity crisis.

LFI ∈ [0, 1] where:
    0.0 = perfectly healthy market
    1.0 = fully fragile (illiquid, wide spreads, no fills)

LFI is a weighted composite of three normalized sub-scores:
    1. Spread fragility   — how much the spread has widened from baseline
    2. Depth collapse     — how much the book depth has dropped from baseline
    3. Fill rate drop     — how much fill activity has declined from baseline

Each sub-score is a ratio vs a rolling "healthy baseline" computed
from the first N ticks. This means LFI is adaptive: it measures
deterioration relative to what was normal for THIS simulation.

When LFI crosses thresholds, a FRAGILITY_WARNING event is emitted
via the API WebSocket.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

LFI_WARNING  = 0.5    # yellow alert
LFI_DANGER   = 0.75   # red alert
LFI_CRITICAL = 0.9    # crash risk

# Weights for composite score
W_SPREAD    = 0.35
W_DEPTH     = 0.40
W_FILL_RATE = 0.25

# Baseline window — ticks used to establish "normal"
BASELINE_WINDOW = 50


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class FragilitySnapshot:
    tick:          int
    lfi:           float
    spread_score:  float
    depth_score:   float
    fill_score:    float
    alert_level:   str     # "NORMAL", "WARNING", "DANGER", "CRITICAL"


# ---------------------------------------------------------------------------
# FragilityIndex
# ---------------------------------------------------------------------------

class FragilityIndex:

    def __init__(self) -> None:
        self._spreads:    List[float] = []
        self._depths:     List[float] = []
        self._fill_rates: List[float] = []
        self._lfi_history: List[FragilitySnapshot] = []

        # Established baselines (set after BASELINE_WINDOW ticks)
        self._baseline_spread:    Optional[float] = None
        self._baseline_depth:     Optional[float] = None
        self._baseline_fill_rate: Optional[float] = None

        self._tick = 0
        self._current_lfi: float = 0.0

    # ------------------------------------------------------------------

    def update(
        self,
        tick:       int,
        spread:     Optional[float],
        depth:      float,
        fill_count: int,
    ) -> float:
        """
        Update LFI with new tick data.
        Returns current LFI score ∈ [0, 1].
        """
        self._tick = tick
        sp = spread if spread is not None else 0.0

        self._spreads.append(sp)
        self._depths.append(depth)
        self._fill_rates.append(float(fill_count))

        if len(self._spreads) > 500:
            self._spreads.pop(0)
            self._depths.pop(0)
            self._fill_rates.pop(0)

        # Establish baseline from first BASELINE_WINDOW ticks
        if len(self._spreads) == BASELINE_WINDOW and self._baseline_spread is None:
            self._baseline_spread    = max(float(np.mean(self._spreads)),    1e-9)
            self._baseline_depth     = max(float(np.mean(self._depths)),     1e-9)
            self._baseline_fill_rate = max(float(np.mean(self._fill_rates)), 1e-9)

        if self._baseline_spread is None:
            # Pre-baseline: LFI = 0
            self._current_lfi = 0.0
            return 0.0

        # Rolling short window for current conditions
        window = 10
        cur_spread    = float(np.mean(self._spreads[-window:]))
        cur_depth     = float(np.mean(self._depths[-window:]))
        cur_fill_rate = float(np.mean(self._fill_rates[-window:]))

        # Sub-scores: normalized deviation from baseline
        spread_score    = np.clip((cur_spread / self._baseline_spread - 1.0), 0.0, 4.0) / 4.0
        depth_score     = np.clip(1.0 - cur_depth / self._baseline_depth,     0.0, 1.0)
        fill_score      = np.clip(1.0 - cur_fill_rate / self._baseline_fill_rate, 0.0, 1.0)

        lfi = (
            W_SPREAD    * spread_score  +
            W_DEPTH     * depth_score   +
            W_FILL_RATE * fill_score
        )
        lfi = float(np.clip(lfi, 0.0, 1.0))
        self._current_lfi = lfi

        alert = self._alert_level(lfi)
        snap = FragilitySnapshot(
            tick         = tick,
            lfi          = round(lfi, 4),
            spread_score = round(float(spread_score), 4),
            depth_score  = round(float(depth_score), 4),
            fill_score   = round(float(fill_score), 4),
            alert_level  = alert,
        )
        self._lfi_history.append(snap)
        if len(self._lfi_history) > 500:
            self._lfi_history.pop(0)

        return lfi

    # ------------------------------------------------------------------

    @property
    def lfi(self) -> float:
        return self._current_lfi

    @property
    def is_warning(self) -> bool:
        return self._current_lfi >= LFI_WARNING

    @property
    def is_critical(self) -> bool:
        return self._current_lfi >= LFI_CRITICAL

    @property
    def alert_level(self) -> str:
        return self._alert_level(self._current_lfi)

    def _alert_level(self, lfi: float) -> str:
        if lfi >= LFI_CRITICAL: return "CRITICAL"
        if lfi >= LFI_DANGER:   return "DANGER"
        if lfi >= LFI_WARNING:  return "WARNING"
        return "NORMAL"

    def snapshot_for_api(self) -> Dict[str, Any]:
        return {
            "lfi":          round(self._current_lfi, 4),
            "alert_level":  self.alert_level,
            "history": [
                {
                    "tick":         s.tick,
                    "lfi":          s.lfi,
                    "spread_score": s.spread_score,
                    "depth_score":  s.depth_score,
                    "fill_score":   s.fill_score,
                    "alert_level":  s.alert_level,
                }
                for s in self._lfi_history[-100:]
            ]
        }

"""
engine/analytics/regime_detector.py

4-state market regime classifier.

States:
    CALM      — low volatility, narrow spread, normal fill rate
    TRENDING  — directional drift, moderate volatility
    UNSTABLE  — high volatility, wide and erratic spread
    CRASH     — extreme volatility spike, liquidity withdrawal

Classification uses a rule-based heuristic (fast, deterministic,
explainable — appropriate for a simulation where ground truth is
known). A Bayesian HMM could be fitted offline and loaded as a
lookup table if desired.

Inputs per tick:
    - realized_vol  (rolling 20-tick std of log returns)
    - spread        (current bid-ask spread)
    - fill_rate     (fills per tick, rolling 10-tick mean)
    - depth         (total book depth, rolling 10-tick mean)

Regime changes emit a REGIME_CHANGE event through the API WebSocket.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


class Regime(str, Enum):
    CALM     = "CALM"
    TRENDING = "TRENDING"
    UNSTABLE = "UNSTABLE"
    CRASH    = "CRASH"


@dataclass
class RegimeSnapshot:
    tick:     int
    regime:   Regime
    vol:      float
    spread:   float
    fill_rate: float
    depth:    float


class RegimeDetector:

    # Classification thresholds (calibrate after observing simulation output)
    VOL_CALM        = 0.002    # below this → base calm signal
    VOL_TRENDING    = 0.005    # moderate vol
    VOL_UNSTABLE    = 0.010    # high vol
    VOL_CRASH       = 0.020    # extreme vol

    SPREAD_WIDE     = 3.0      # spread above this suggests stress
    FILL_RATE_LOW   = 0.3      # fills per tick below this → illiquidity
    DEPTH_LOW       = 10       # total depth below this → depth collapse

    VOL_WINDOW      = 20
    FILL_WINDOW     = 10
    DEPTH_WINDOW    = 10

    def __init__(self) -> None:
        self._log_returns:  List[float] = []
        self._fills_per_tick: List[int] = []
        self._depths:       List[float] = []
        self._current:      Regime      = Regime.CALM
        self._history:      List[RegimeSnapshot] = []
        self._tick:         int         = 0

    # ------------------------------------------------------------------

    def update(
        self,
        tick:          int,
        mid_price:     Optional[float],
        spread:        Optional[float],
        fills_this_tick: int,
        book_depth:    float,
        prev_mid:      Optional[float],
    ) -> Regime:
        """
        Update with new tick data. Returns the detected regime.
        """
        self._tick = tick

        # Log return
        if mid_price and prev_mid and prev_mid > 0:
            log_ret = np.log(mid_price / prev_mid)
        else:
            log_ret = 0.0
        self._log_returns.append(log_ret)
        if len(self._log_returns) > self.VOL_WINDOW:
            self._log_returns.pop(0)

        self._fills_per_tick.append(fills_this_tick)
        if len(self._fills_per_tick) > self.FILL_WINDOW:
            self._fills_per_tick.pop(0)

        self._depths.append(book_depth)
        if len(self._depths) > self.DEPTH_WINDOW:
            self._depths.pop(0)

        # Compute rolling stats
        vol        = float(np.std(self._log_returns)) if self._log_returns else 0.0
        avg_fills  = float(np.mean(self._fills_per_tick)) if self._fills_per_tick else 0.0
        avg_depth  = float(np.mean(self._depths)) if self._depths else 0.0
        cur_spread = spread or 0.0

        new_regime = self._classify(vol, cur_spread, avg_fills, avg_depth)
        self._current = new_regime

        snap = RegimeSnapshot(
            tick=tick, regime=new_regime,
            vol=round(vol, 6), spread=round(cur_spread, 4),
            fill_rate=round(avg_fills, 2), depth=round(avg_depth, 2),
        )
        self._history.append(snap)
        if len(self._history) > 500:
            self._history.pop(0)

        return new_regime

    def _classify(
        self,
        vol:       float,
        spread:    float,
        fill_rate: float,
        depth:     float,
    ) -> Regime:
        """Rule-based classifier."""
        # CRASH — extreme conditions
        if vol > self.VOL_CRASH or (depth < self.DEPTH_LOW and spread > self.SPREAD_WIDE * 2):
            return Regime.CRASH

        # UNSTABLE — high vol or stressed liquidity
        if vol > self.VOL_UNSTABLE or (spread > self.SPREAD_WIDE and fill_rate < self.FILL_RATE_LOW):
            return Regime.UNSTABLE

        # TRENDING — moderate vol (price moving, but orderly)
        if vol > self.VOL_TRENDING:
            return Regime.TRENDING

        # CALM — everything normal
        return Regime.CALM

    @property
    def current(self) -> Regime:
        return self._current

    @property
    def history(self) -> List[RegimeSnapshot]:
        return list(self._history)

    def snapshot_for_api(self) -> Dict[str, Any]:
        return {
            "current_regime": self._current.value,
            "tick":           self._tick,
            "history": [
                {
                    "tick":      s.tick,
                    "regime":    s.regime.value,
                    "vol":       s.vol,
                    "spread":    s.spread,
                    "fill_rate": s.fill_rate,
                    "depth":     s.depth,
                }
                for s in self._history[-50:]  # last 50 regime readings
            ]
        }

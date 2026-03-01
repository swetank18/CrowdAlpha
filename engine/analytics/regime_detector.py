"""
engine/analytics/regime_detector.py

Observable-signal regime classification:
vol level, vol autocorrelation, LFI, spread elasticity, crowding.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class Regime(str, Enum):
    CALM = "CALM"
    TRENDING = "TRENDING"
    UNSTABLE = "UNSTABLE"
    CRASH_PRONE = "CRASH_PRONE"


@dataclass
class RegimeSnapshot:
    tick: int
    regime: Regime
    vol_level: float
    vol_autocorr: float
    lfi: float
    spread_elasticity: float
    crowding: float
    drift: float


class RegimeDetector:
    def __init__(self, calibration_window: int = 80) -> None:
        self.calibration_window = max(40, calibration_window)

        self._returns: List[float] = []
        self._spreads: List[float] = []
        self._depths: List[float] = []
        self._lfi: List[float] = []
        self._crowding: List[float] = []
        self._elasticity: List[float] = []
        self._drift: List[float] = []
        self._history: List[RegimeSnapshot] = []
        self._current: Regime = Regime.CALM
        self._tick: int = 0
        self._prev_mid: Optional[float] = None

    def update(
        self,
        tick: int,
        mid_price: Optional[float],
        spread: Optional[float],
        depth: float,
        lfi: float,
        crowding: float,
        fill_imbalance: float,
    ) -> Regime:
        self._tick = tick
        sp = float(spread or 0.0)
        dp = float(max(depth, 0.0))
        lfi_val = float(np.clip(lfi, 0.0, 1.0))
        crowd = float(np.clip(crowding, -1.0, 1.0))

        ret = 0.0
        if mid_price is not None and self._prev_mid is not None and self._prev_mid > 0:
            ret = float(np.log(mid_price / self._prev_mid))
        if mid_price is not None:
            self._prev_mid = float(mid_price)

        self._returns.append(ret)
        self._spreads.append(sp)
        self._depths.append(dp)
        self._lfi.append(lfi_val)
        self._crowding.append(crowd)

        if len(self._returns) > 1000:
            self._returns.pop(0)
            self._spreads.pop(0)
            self._depths.pop(0)
            self._lfi.pop(0)
            self._crowding.pop(0)

        vol_level = self._rolling_vol()
        vol_autocorr = self._vol_autocorr()
        drift = self._rolling_drift()
        elasticity = self._spread_elasticity(fill_imbalance)
        self._elasticity.append(elasticity)
        self._drift.append(abs(drift))
        if len(self._elasticity) > 1000:
            self._elasticity.pop(0)
            self._drift.pop(0)

        regime = self._classify(
            vol_level=vol_level,
            vol_autocorr=vol_autocorr,
            lfi=lfi_val,
            spread_elasticity=elasticity,
            crowding=crowd,
            drift=drift,
        )
        self._current = regime

        snap = RegimeSnapshot(
            tick=tick,
            regime=regime,
            vol_level=round(vol_level, 6),
            vol_autocorr=round(vol_autocorr, 4),
            lfi=round(lfi_val, 4),
            spread_elasticity=round(elasticity, 4),
            crowding=round(crowd, 4),
            drift=round(drift, 6),
        )
        self._history.append(snap)
        if len(self._history) > 600:
            self._history.pop(0)
        return regime

    @property
    def current(self) -> Regime:
        return self._current

    def snapshot_for_api(self) -> Dict[str, Any]:
        return {
            "current_regime": self._current.value,
            "tick": self._tick,
            "history": [
                {
                    "tick": s.tick,
                    "regime": s.regime.value,
                    "vol_level": s.vol_level,
                    "vol_autocorr": s.vol_autocorr,
                    "lfi": s.lfi,
                    "spread_elasticity": s.spread_elasticity,
                    "crowding": s.crowding,
                    "drift": s.drift,
                }
                for s in self._history[-120:]
            ],
        }

    def _rolling_vol(self, window: int = 40) -> float:
        r = self._returns[-window:]
        return float(np.std(r)) if len(r) >= 2 else 0.0

    def _vol_autocorr(self, window: int = 60) -> float:
        r = np.abs(np.array(self._returns[-window:], dtype=float))
        if len(r) < 4:
            return 0.0
        x = r[:-1]
        y = r[1:]
        if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    def _rolling_drift(self, window: int = 20) -> float:
        r = self._returns[-window:]
        return float(np.mean(r)) if len(r) >= 2 else 0.0

    def _spread_elasticity(self, fill_imbalance: float) -> float:
        if len(self._spreads) < 2 or len(self._depths) < 2:
            return 0.0
        s0, s1 = self._spreads[-2], self._spreads[-1]
        d0, d1 = self._depths[-2], self._depths[-1]

        spread_chg = (s1 - s0) / max(abs(s0), 1e-6)
        depth_chg = (d1 - d0) / max(abs(d0), 1e-6)

        if depth_chg < 0:
            base = max(spread_chg, 0.0) / max(abs(depth_chg), 1e-6)
        else:
            base = max(spread_chg, 0.0)
        return float(np.clip(base * (1.0 + abs(fill_imbalance)), 0.0, 5.0))

    def _calibrated(self, values: List[float], fallback: float, q: float) -> float:
        if len(values) < self.calibration_window:
            return fallback
        return float(np.quantile(np.array(values[-self.calibration_window :], dtype=float), q))

    def _classify(
        self,
        vol_level: float,
        vol_autocorr: float,
        lfi: float,
        spread_elasticity: float,
        crowding: float,
        drift: float,
    ) -> Regime:
        vol_q75 = self._calibrated([abs(x) for x in self._returns], 0.0006, 0.75)
        vol_q90 = self._calibrated([abs(x) for x in self._returns], 0.0012, 0.90)
        vac_q75 = self._calibrated([max(x, 0.0) for x in self._elasticity], 0.45, 0.75)
        lfi_q75 = self._calibrated(self._lfi, 0.45, 0.75)
        lfi_q90 = self._calibrated(self._lfi, 0.7, 0.90)
        crowd_q75 = self._calibrated([max(x, 0.0) for x in self._crowding], 0.55, 0.75)
        elas_q75 = self._calibrated(self._elasticity, 0.7, 0.75)
        elas_q90 = self._calibrated(self._elasticity, 1.2, 0.90)
        drift_q75 = self._calibrated(self._drift, 0.00008, 0.75)

        # Crash-prone: high fragility + elastic spread + crowded or clustered vol shock.
        if (
            lfi >= max(0.75, lfi_q90)
            and spread_elasticity >= max(1.0, elas_q90)
            and crowding >= max(0.55, crowd_q75)
        ) or (
            vol_level >= max(0.0012, vol_q90)
            and vol_autocorr >= max(0.25, vac_q75)
            and lfi >= max(0.60, lfi_q75)
        ):
            return Regime.CRASH_PRONE

        # Unstable: elevated vol with clustering or fragile liquidity.
        if vol_level >= max(0.0008, vol_q75) and (
            vol_autocorr >= 0.18
            or spread_elasticity >= max(0.6, elas_q75)
            or lfi >= max(0.50, lfi_q75)
        ):
            return Regime.UNSTABLE

        # Trending: directional drift under manageable fragility.
        if abs(drift) >= max(0.00008, drift_q75) and lfi < max(0.60, lfi_q90):
            return Regime.TRENDING

        return Regime.CALM


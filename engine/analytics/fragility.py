"""
engine/analytics/fragility.py

Liquidity Fragility Index from depth concentration near top-of-book
and recent fill-side imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from engine.core.order import Fill


LFI_WARNING = 0.45
LFI_DANGER = 0.65
LFI_CRITICAL = 0.85
BASELINE_WINDOW = 50


@dataclass
class FragilitySnapshot:
    tick: int
    lfi: float
    near_depth_ratio: float
    fill_imbalance: float
    adjusted_ratio: float
    alert_level: str


class FragilityIndex:
    def __init__(self, near_ticks: int = 5) -> None:
        self.near_ticks = near_ticks
        self._ratio_hist: List[float] = []
        self._imb_hist: List[float] = []
        self._raw_hist: List[float] = []
        self._history: List[FragilitySnapshot] = []
        self._baseline_mean: Optional[float] = None
        self._baseline_std: Optional[float] = None
        self._current_lfi: float = 0.0
        self._current_imbalance: float = 0.0
        self._current_near_depth_ratio: float = 0.0
        self._current_adjusted_ratio: float = 0.0

    def update(
        self,
        tick: int,
        bid_levels: Sequence[Tuple[float, float]],
        ask_levels: Sequence[Tuple[float, float]],
        mid_price: Optional[float],
        fills: Sequence[Fill],
    ) -> float:
        ratio = self._near_depth_ratio(bid_levels, ask_levels, mid_price)
        imb = self._fill_imbalance(mid_price, fills)
        raw = ratio * (1.0 + imb)

        self._ratio_hist.append(ratio)
        self._imb_hist.append(imb)
        self._raw_hist.append(raw)
        self._current_imbalance = imb
        self._current_near_depth_ratio = ratio
        self._current_adjusted_ratio = raw

        if len(self._ratio_hist) > 800:
            self._ratio_hist.pop(0)
            self._imb_hist.pop(0)
            self._raw_hist.pop(0)

        if self._baseline_mean is None and len(self._raw_hist) >= BASELINE_WINDOW:
            init = np.array(self._raw_hist[:BASELINE_WINDOW], dtype=float)
            self._baseline_mean = float(np.mean(init))
            self._baseline_std = float(np.std(init))

        lfi = 0.0
        if self._baseline_mean is not None:
            smoothed_raw = float(np.mean(self._raw_hist[-5:]))
            denom = max(2.0 * (self._baseline_std or 0.0), 1e-6)
            z = (smoothed_raw - self._baseline_mean) / denom
            lfi = float(np.clip(z, 0.0, 1.0))

        self._current_lfi = lfi
        snap = FragilitySnapshot(
            tick=tick,
            lfi=round(lfi, 4),
            near_depth_ratio=round(ratio, 4),
            fill_imbalance=round(imb, 4),
            adjusted_ratio=round(raw, 4),
            alert_level=self.alert_level,
        )
        self._history.append(snap)
        if len(self._history) > 500:
            self._history.pop(0)

        return lfi

    @property
    def lfi(self) -> float:
        return self._current_lfi

    @property
    def fill_imbalance(self) -> float:
        return self._current_imbalance

    @property
    def near_depth_ratio(self) -> float:
        return self._current_near_depth_ratio

    @property
    def adjusted_ratio(self) -> float:
        return self._current_adjusted_ratio

    @property
    def alert_level(self) -> str:
        if self._current_lfi >= LFI_CRITICAL:
            return "CRITICAL"
        if self._current_lfi >= LFI_DANGER:
            return "DANGER"
        if self._current_lfi >= LFI_WARNING:
            return "WARNING"
        return "NORMAL"

    def snapshot_for_api(self) -> Dict[str, Any]:
        return {
            "lfi": round(self._current_lfi, 4),
            "fill_imbalance": round(self._current_imbalance, 4),
            "alert_level": self.alert_level,
            "history": [
                {
                    "tick": s.tick,
                    "lfi": s.lfi,
                    "near_depth_ratio": s.near_depth_ratio,
                    "fill_imbalance": s.fill_imbalance,
                    "adjusted_ratio": s.adjusted_ratio,
                    "alert_level": s.alert_level,
                }
                for s in self._history[-120:]
            ],
        }

    def _near_depth_ratio(
        self,
        bid_levels: Sequence[Tuple[float, float]],
        ask_levels: Sequence[Tuple[float, float]],
        mid_price: Optional[float],
    ) -> float:
        total_depth = float(sum(q for _, q in bid_levels) + sum(q for _, q in ask_levels))
        if total_depth <= 1e-9:
            return 0.0

        if mid_price is None:
            if bid_levels and ask_levels:
                mid = (float(bid_levels[0][0]) + float(ask_levels[0][0])) / 2.0
            elif bid_levels:
                mid = float(bid_levels[0][0])
            elif ask_levels:
                mid = float(ask_levels[0][0])
            else:
                return 0.0
        else:
            mid = float(mid_price)

        tick_size = self._infer_tick_size(bid_levels, ask_levels, mid)
        band = self.near_ticks * tick_size

        near = 0.0
        for p, q in bid_levels:
            if abs(float(p) - mid) <= band:
                near += float(q)
        for p, q in ask_levels:
            if abs(float(p) - mid) <= band:
                near += float(q)

        return float(np.clip(near / total_depth, 0.0, 1.0))

    def _fill_imbalance(self, mid_price: Optional[float], fills: Sequence[Fill]) -> float:
        if not fills:
            return 0.0
        if mid_price is None:
            mid = float(np.mean([f.price for f in fills]))
        else:
            mid = float(mid_price)

        buy_pressure = 0.0
        sell_pressure = 0.0
        for f in fills:
            qty = float(f.qty)
            if f.price >= mid:
                buy_pressure += qty
            else:
                sell_pressure += qty
        total = buy_pressure + sell_pressure
        if total <= 1e-9:
            return 0.0
        return float(abs(buy_pressure - sell_pressure) / total)

    def _infer_tick_size(
        self,
        bid_levels: Sequence[Tuple[float, float]],
        ask_levels: Sequence[Tuple[float, float]],
        mid: float,
    ) -> float:
        candidates: List[float] = []
        bid_prices = [float(p) for p, _ in bid_levels]
        ask_prices = [float(p) for p, _ in ask_levels]

        for prices in (sorted(bid_prices, reverse=True), sorted(ask_prices)):
            for i in range(1, len(prices)):
                d = abs(prices[i] - prices[i - 1])
                if d > 1e-9:
                    candidates.append(d)
        if candidates:
            return max(min(candidates), 1e-4)
        return max(mid * 0.0005, 0.01)

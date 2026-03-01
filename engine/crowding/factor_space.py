"""
engine/crowding/factor_space.py

Builds behavior-derived factor vectors from observable order/fill history.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from engine.core.order import Fill, Order, OrderType, Side

if TYPE_CHECKING:
    from engine.agents.base_agent import BaseAgent


N_FACTORS = 6
FACTOR_NAMES = [
    "turnover_rate",
    "avg_holding_period",
    "directional_bias",
    "volatility_exposure",
    "inventory_skew",
    "order_aggressiveness",
]


class FactorSpace:
    """
    Factor engine based only on observable behavior:
    orders, fills, and market prices over a rolling window.
    """

    def __init__(self, window: int = 50) -> None:
        self.window = max(10, window)

        self._order_events: Deque[Dict[str, Any]] = deque()
        self._fill_events: Deque[Dict[str, Any]] = deque()
        self._mid_by_tick: Dict[int, float] = {}

        self._current_matrix: Optional[np.ndarray] = None
        self._agent_ids: List[str] = []
        self._activity_map: Dict[str, float] = {}
        self._volume_share_map: Dict[str, float] = {}

    def update(
        self,
        agents: List["BaseAgent"],
        tick: int,
        mid_price: float,
        best_bid: Optional[float],
        best_ask: Optional[float],
        orders: List[Order],
        fills: List[Fill],
    ) -> np.ndarray:
        self._record_mid(tick, mid_price)
        self._record_orders(tick, mid_price, best_bid, best_ask, orders)
        self._record_fills(tick, fills)
        self._evict_old(tick)

        agent_ids = [a.agent_id for a in agents]
        raw = self._compute_raw_features(agent_ids, tick)
        F = self._normalize_features(raw, agent_ids)

        self._agent_ids = agent_ids
        self._current_matrix = F
        return F

    def latest_activity_map(self) -> Dict[str, float]:
        return dict(self._activity_map)

    def activity_weights(self, agent_ids: List[str]) -> List[float]:
        if not agent_ids:
            return []
        weights = [max(0.0, self._volume_share_map.get(agent_id, 0.0)) for agent_id in agent_ids]
        total = float(sum(weights))
        if total < 1e-12:
            uniform = 1.0 / float(len(agent_ids))
            return [uniform for _ in agent_ids]
        return [w / total for w in weights]

    def current_matrix(self) -> Optional[np.ndarray]:
        return self._current_matrix

    def pca_projection(self) -> Optional[np.ndarray]:
        F = self._current_matrix
        if F is None or F.shape[0] < 2:
            return None
        centered = F - F.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            return centered @ vt[:2].T
        except np.linalg.LinAlgError:
            return None

    def snapshot_for_api(self) -> Dict[str, Any]:
        F = self._current_matrix
        if F is None:
            return {"agents": [], "factor_names": FACTOR_NAMES}

        pca = self.pca_projection()
        agents_out: List[Dict[str, Any]] = []
        for idx, agent_id in enumerate(self._agent_ids):
            entry: Dict[str, Any] = {
                "agent_id": agent_id,
                "factors": dict(zip(FACTOR_NAMES, [float(x) for x in F[idx].tolist()])),
                "activity": round(float(self._activity_map.get(agent_id, 0.0)), 4),
                "volume_share": round(float(self._volume_share_map.get(agent_id, 0.0)), 6),
            }
            if pca is not None:
                entry["pca"] = {"x": float(pca[idx, 0]), "y": float(pca[idx, 1])}
            agents_out.append(entry)

        return {"agents": agents_out, "factor_names": FACTOR_NAMES}

    def _record_mid(self, tick: int, mid: float) -> None:
        self._mid_by_tick[tick] = float(mid)

    def _record_orders(
        self,
        tick: int,
        mid: float,
        best_bid: Optional[float],
        best_ask: Optional[float],
        orders: List[Order],
    ) -> None:
        for order in orders:
            if order.order_type == OrderType.CANCEL:
                continue
            side_sign = 1.0 if order.side == Side.BID else -1.0
            self._order_events.append(
                {
                    "tick": tick,
                    "agent_id": order.agent_id,
                    "side_sign": side_sign,
                    "qty": float(order.qty),
                    "order_type": order.order_type.value,
                    "aggr": self._order_aggressiveness(order, mid, best_bid, best_ask),
                }
            )

    def _record_fills(self, tick: int, fills: List[Fill]) -> None:
        for fill in fills:
            qty = float(fill.qty)
            self._fill_events.append(
                {
                    "tick": tick,
                    "agent_id": fill.buy_agent_id,
                    "signed_qty": qty,
                    "abs_qty": qty,
                }
            )
            self._fill_events.append(
                {
                    "tick": tick,
                    "agent_id": fill.sell_agent_id,
                    "signed_qty": -qty,
                    "abs_qty": qty,
                }
            )

    def _evict_old(self, tick: int) -> None:
        cutoff = tick - self.window
        while self._order_events and self._order_events[0]["tick"] <= cutoff:
            self._order_events.popleft()
        while self._fill_events and self._fill_events[0]["tick"] <= cutoff:
            self._fill_events.popleft()

        stale_ticks = [t for t in self._mid_by_tick.keys() if t <= cutoff - 1]
        for t in stale_ticks:
            self._mid_by_tick.pop(t, None)

    def _compute_raw_features(self, agent_ids: List[str], tick: int) -> Dict[str, Dict[str, float]]:
        orders_by_agent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        fills_by_agent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for event in self._order_events:
            orders_by_agent[event["agent_id"]].append(event)
        for event in self._fill_events:
            fills_by_agent[event["agent_id"]].append(event)

        vol_by_tick = self._vol_by_tick(tick)
        market_vol = float(np.mean(list(vol_by_tick.values()))) if vol_by_tick else 0.0

        raw: Dict[str, Dict[str, float]] = {}
        for agent_id in agent_ids:
            orders = orders_by_agent.get(agent_id, [])
            fills = fills_by_agent.get(agent_id, [])

            ord_qty = float(sum(e["qty"] for e in orders))
            fill_qty = float(sum(e["abs_qty"] for e in fills))
            ord_signed = float(sum(e["side_sign"] * e["qty"] for e in orders))
            fill_signed = float(sum(e["signed_qty"] for e in fills))

            turnover_raw = fill_qty / max(float(self.window), 1.0)
            directional = ord_signed / max(ord_qty, 1e-9) if ord_qty > 0 else 0.0
            inv_skew = fill_signed / max(fill_qty, 1e-9) if fill_qty > 0 else 0.0
            order_aggr = (
                float(np.average([e["aggr"] for e in orders], weights=[e["qty"] for e in orders]))
                if orders
                else 0.0
            )
            holding = self._avg_holding_period_ratio(fills, tick)
            vol_exposure = self._vol_exposure_ratio(orders, vol_by_tick, market_vol)

            raw[agent_id] = {
                "turnover_raw": turnover_raw,
                "avg_holding_period": holding,
                "directional_bias": float(np.clip(directional, -1.0, 1.0)),
                "volatility_exposure": vol_exposure,
                "inventory_skew": float(np.clip(inv_skew, -1.0, 1.0)),
                "order_aggressiveness": float(np.clip(order_aggr, 0.0, 1.0)),
                "activity_raw": ord_qty + fill_qty,
                "volume_raw": fill_qty,
            }
        return raw

    def _normalize_features(
        self,
        raw: Dict[str, Dict[str, float]],
        agent_ids: List[str],
    ) -> np.ndarray:
        if not agent_ids:
            self._activity_map = {}
            return np.zeros((0, N_FACTORS), dtype=float)

        turnover_vals = np.array([raw[a]["turnover_raw"] for a in agent_ids], dtype=float)
        activity_vals = np.array([raw[a]["activity_raw"] for a in agent_ids], dtype=float)
        volume_vals = np.array([raw[a]["volume_raw"] for a in agent_ids], dtype=float)

        turn_scale = float(np.percentile(turnover_vals, 90)) if np.any(turnover_vals > 0) else 1.0
        act_scale = float(np.percentile(activity_vals, 90)) if np.any(activity_vals > 0) else 1.0
        turn_scale = max(turn_scale, 1e-6)
        act_scale = max(act_scale, 1e-6)

        matrix_rows: List[np.ndarray] = []
        activity_map: Dict[str, float] = {}
        volume_share_map: Dict[str, float] = {}
        vol_total = float(np.sum(volume_vals))
        n_agents = float(max(1, len(agent_ids)))
        for agent_id in agent_ids:
            r = raw[agent_id]
            turnover = float(np.clip(r["turnover_raw"] / turn_scale, 0.0, 1.0))
            holding = float(np.clip(r["avg_holding_period"], 0.0, 1.0))
            directional = float(np.clip(r["directional_bias"], -1.0, 1.0))
            vol_exp = float(np.clip(r["volatility_exposure"], 0.0, 1.0))
            inv_skew = float(np.clip(r["inventory_skew"], -1.0, 1.0))
            aggr = float(np.clip(r["order_aggressiveness"], 0.0, 1.0))

            matrix_rows.append(np.array([turnover, holding, directional, vol_exp, inv_skew, aggr], dtype=float))
            activity_map[agent_id] = float(np.clip(r["activity_raw"] / act_scale, 0.0, 1.0))
            if vol_total > 1e-12:
                volume_share_map[agent_id] = float(max(0.0, r["volume_raw"]) / vol_total)
            else:
                volume_share_map[agent_id] = 1.0 / n_agents

        self._activity_map = activity_map
        self._volume_share_map = volume_share_map
        return np.vstack(matrix_rows)

    def _vol_by_tick(self, tick: int) -> Dict[int, float]:
        out: Dict[int, float] = {}
        start = tick - self.window + 1
        for t in range(max(start, 2), tick + 1):
            p0 = self._mid_by_tick.get(t - 1)
            p1 = self._mid_by_tick.get(t)
            if p0 is None or p1 is None or p0 <= 0:
                continue
            out[t] = abs(float(np.log(p1 / p0)))
        return out

    def _vol_exposure_ratio(
        self,
        orders: List[Dict[str, Any]],
        vol_by_tick: Dict[int, float],
        market_vol: float,
    ) -> float:
        if not orders or not vol_by_tick:
            return 0.0
        act_by_tick: Dict[int, float] = defaultdict(float)
        for e in orders:
            act_by_tick[e["tick"]] += e["qty"]

        numer = 0.0
        denom = 0.0
        for t, act in act_by_tick.items():
            numer += act * vol_by_tick.get(t, 0.0)
            denom += act
        if denom < 1e-9:
            return 0.0
        agent_vol = numer / denom
        if market_vol < 1e-9:
            return 0.0
        return float(np.clip((agent_vol / market_vol) / 2.0, 0.0, 1.0))

    def _avg_holding_period_ratio(self, fills: List[Dict[str, Any]], tick: int) -> float:
        if not fills:
            return 0.0

        lots: Deque[Dict[str, float]] = deque()
        weighted_age = 0.0
        closed_qty = 0.0

        fills_sorted = sorted(fills, key=lambda x: x["tick"])
        for f in fills_sorted:
            cur_tick = int(f["tick"])
            signed = float(f["signed_qty"])
            side = 1.0 if signed > 0 else -1.0
            qty_left = abs(signed)

            while qty_left > 1e-9 and lots and lots[0]["side"] != side:
                lot = lots[0]
                matched = min(qty_left, lot["qty"])
                weighted_age += matched * max(cur_tick - lot["open_tick"], 0.0)
                closed_qty += matched
                lot["qty"] -= matched
                qty_left -= matched
                if lot["qty"] <= 1e-9:
                    lots.popleft()

            if qty_left > 1e-9:
                lots.append({"side": side, "qty": qty_left, "open_tick": float(cur_tick)})

        # Include open inventory age as ongoing holding time.
        for lot in lots:
            weighted_age += lot["qty"] * max(tick - lot["open_tick"], 0.0)
            closed_qty += lot["qty"]

        if closed_qty < 1e-9:
            return 0.0
        avg_ticks = weighted_age / closed_qty
        return float(np.clip(avg_ticks / max(self.window, 1), 0.0, 1.0))

    def _order_aggressiveness(
        self,
        order: Order,
        mid: float,
        best_bid: Optional[float],
        best_ask: Optional[float],
    ) -> float:
        if order.order_type == OrderType.MARKET:
            return 1.0

        bid = best_bid if best_bid is not None else mid
        ask = best_ask if best_ask is not None else mid
        spread = max(ask - bid, 1e-6)

        if order.side == Side.BID:
            if best_ask is not None and order.price >= best_ask:
                return 1.0
            rel = (order.price - bid) / spread
            return float(np.clip(rel, 0.0, 1.0))

        if best_bid is not None and order.price <= best_bid:
            return 1.0
        rel = (ask - order.price) / spread
        return float(np.clip(rel, 0.0, 1.0))

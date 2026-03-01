"""
engine/simulation.py

Top-level simulation loop — ties everything together.

This is the entry point that:
  1. Creates and populates the market with agents
  2. Runs the discrete tick-by-tick event loop
  3. Feeds each tick's state through the full pipeline:
       matching engine → execution → crowding → analytics
  4. Exposes an async interface for the FastAPI WebSocket layer
  5. Can also be run as a standalone script for console debugging

Usage (script mode):
    cd crowdalpha
    python -m engine.simulation --ticks 200 --print-prices

Usage (API mode):
    sim = Simulation(config)
    await sim.run_async(broadcast_fn=ws_broadcaster)
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from engine.core.order import Order, Fill, Side, OrderType
from engine.core.order_book import OrderBook
from engine.core.matching_engine import MatchingEngine
from engine.core.execution import ExecutionEngine, ImpactModel
from engine.agents.base_agent import BaseAgent, MarketState
from engine.agents.registry import AgentRegistry
from engine.crowding.factor_space import FactorSpace
from engine.crowding.crowding_matrix import CrowdingMatrix
from engine.crowding.alpha_decay import AlphaDecay
from engine.analytics.diagnostics import Diagnostics
from engine.analytics.regime_detector import RegimeDetector
from engine.analytics.fragility import FragilityIndex
from engine.events import EventType, make_event


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    # Population
    n_momentum:       int   = 3
    n_mean_reversion: int   = 3
    n_market_makers:  int   = 2
    n_rl:             int   = 0

    # Tick control
    tick_delay_ms:    float = 100.0   # 0 = as fast as possible
    max_ticks:        Optional[int]   = None  # None = run forever

    # Market parameters
    initial_price:    float = 100.0
    initial_cash:     float = 100_000.0
    eta:              float = 0.01    # market impact coefficient
    crowding_kappa:   float = 18.0    # impact amplification sensitivity to agent crowding
    seed:             Optional[int]  = None


# ---------------------------------------------------------------------------
# Simulation state (sent to API / frontend each tick)
# ---------------------------------------------------------------------------

@dataclass
class TickState:
    tick:           int
    timestamp:      int         # ns
    mid_price:      Optional[float]
    spread:         Optional[float]
    vwap:           float
    volatility:     float
    regime:         str
    lfi:            float
    lfi_near_depth_ratio: float
    lfi_alert:      str
    crowding:       float       # scalar intensity
    order_book:     Dict        # depth snapshot
    recent_fills:   List[Dict]
    agent_stats:    List[Dict]
    crowding_data:  Dict        # full crowding snapshot
    factor_data:    Dict        # factor space snapshot
    decay_data:     Dict        # alpha decay snapshot

    def to_ws_event(self) -> Dict:
        return make_event(
            EventType.TICK,
            {
                "tick": self.tick,
                "mid_price": self.mid_price,
                "spread": self.spread,
                "vwap": self.vwap,
                "volatility": self.volatility,
                "regime": self.regime,
                "lfi": self.lfi,
                "lfi_near_depth_ratio": self.lfi_near_depth_ratio,
                "lfi_alert": self.lfi_alert,
                "crowding": self.crowding,
                "order_book": self.order_book,
                "recent_fills": self.recent_fills,
                "agent_stats": self.agent_stats,
            },
            timestamp=self.timestamp,
        )


# ---------------------------------------------------------------------------
# Main Simulation class
# ---------------------------------------------------------------------------

class Simulation:

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.config  = config or SimulationConfig()
        self.sim_id  = f"sim_{int(time.time())}"
        self._tick   = 0
        self._running= False

        # Set random seed
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        # Core engine
        self.book             = OrderBook()
        self.matching_engine  = MatchingEngine(self.book, on_fill=self._on_fill)
        self.execution_engine = ExecutionEngine(ImpactModel(self.config.eta))

        # Agents
        self.registry = AgentRegistry()
        self.agents: List[BaseAgent] = self.registry.create_default_population(
            n_momentum      = self.config.n_momentum,
            n_mean_rev      = self.config.n_mean_reversion,
            n_market_makers = self.config.n_market_makers,
            n_rl            = self.config.n_rl,
        )
        # Register all agent positions with execution engine
        for agent in self.agents:
            self.execution_engine.get_or_create(agent.agent_id, self.config.initial_cash)

        # Crowding pipeline
        self.factor_space     = FactorSpace(window=50)
        self.crowding_matrix  = CrowdingMatrix()
        self.alpha_decay      = AlphaDecay(min_samples=20)

        # Analytics
        self.diagnostics      = Diagnostics()
        self.regime_detector  = RegimeDetector()
        self.fragility        = FragilityIndex()

        # Internal state
        self._recent_fills:  List[Fill]  = []
        self._trade_tape:    List[Dict[str, Any]] = []
        self._price_history: List[float] = []
        self._vwap_num:      float       = 0.0
        self._vwap_den:      float       = 0.0

        # Latest state (for API polling)
        self._last_state:    Optional[TickState] = None
        self._last_events:   List[Dict[str, Any]] = []
        self._last_regime:   Optional[str] = None

        # Callback for WebSocket broadcast
        self._broadcast: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_agent(self, agent: BaseAgent) -> None:
        self.agents.append(agent)
        self.execution_engine.get_or_create(agent.agent_id, self.config.initial_cash)

    def stop(self) -> None:
        self._running = False
        for agent in self.agents:
            close_fn = getattr(agent, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

    def is_running(self) -> bool:
        return self._running

    @property
    def current_state(self) -> Optional[TickState]:
        return self._last_state

    @property
    def latest_events(self) -> List[Dict[str, Any]]:
        return list(self._last_events)

    @property
    def trade_history(self) -> List[Dict[str, Any]]:
        return list(self._trade_tape)

    # ------------------------------------------------------------------
    # Blocking run (script / test mode)
    # ------------------------------------------------------------------

    def run(self, n_ticks: int = 200, print_prices: bool = False) -> List[TickState]:
        """Run synchronously for n_ticks. Returns list of TickStates."""
        self._running = True
        states = []
        for _ in range(n_ticks):
            if not self._running:
                break
            state, _ = self._tick_once()
            states.append(state)
            if print_prices:
                print(
                    f"T={state.tick:4d} | "
                    f"mid={state.mid_price or 0:.4f} | "
                    f"spread={state.spread or 0:.4f} | "
                    f"vol={state.volatility:.5f} | "
                    f"regime={state.regime:<8} | "
                    f"crowd={state.crowding:.3f} | "
                    f"lfi={state.lfi:.3f}"
                )
        self._running = False
        return states

    # ------------------------------------------------------------------
    # Async run (FastAPI / WebSocket mode)
    # ------------------------------------------------------------------

    async def run_async(
        self,
        broadcast_fn: Optional[Callable] = None,
    ) -> None:
        """
        Run indefinitely (or until max_ticks), yielding control after
        each tick. broadcast_fn receives a JSON-serializable dict.
        """
        self._broadcast = broadcast_fn
        self._running   = True
        delay = self.config.tick_delay_ms / 1000.0

        while self._running:
            if self.config.max_ticks and self._tick >= self.config.max_ticks:
                break

            state, events = self._tick_once()

            if self._broadcast:
                try:
                    for event in events:
                        await self._broadcast(event)
                except Exception:
                    pass

            if delay > 0:
                await asyncio.sleep(delay)

        self._running = False

    # ------------------------------------------------------------------
    # Single tick
    # ------------------------------------------------------------------

    def _tick_once(self) -> tuple[TickState, List[Dict[str, Any]]]:
        self._tick += 1
        tick = self._tick
        self._recent_fills.clear()

        # --- 1. Build market state snapshot for agents ---
        mid = self.book.mid_price or (
            self._price_history[-1] if self._price_history else self.config.initial_price
        )
        snap = self.book.depth_snapshot(n_levels=10)
        best_bid_pre = snap["bids"][0][0] if snap["bids"] else None
        best_ask_pre = snap["asks"][0][0] if snap["asks"] else None

        recent_trade_dicts = self._trade_tape[-20:]

        vol = self._rolling_vol()
        impact_buy_mult, impact_sell_mult = self.alpha_decay.current_impact_multipliers()
        agent_impact_mult = self.crowding_matrix.impact_multipliers(self.config.crowding_kappa)
        agent_crowding = self.crowding_matrix.agent_intensity_map
        self.execution_engine.impact.set_side_multipliers(impact_buy_mult, impact_sell_mult)
        self.execution_engine.impact.set_agent_multipliers(agent_impact_mult)

        pre_agent_state: Dict[str, Dict[str, float]] = {
            aid: {
                "cash": float(pos.cash),
                "inventory": float(pos.inventory),
                "pnl": float(pos.mark_to_market(mid)),
            }
            for aid, pos in self.execution_engine.positions.items()
        }

        # --- 2. Collect orders from all agents ---
        all_orders: List[Order] = []
        for agent in self.agents:
            pos = self.execution_engine.positions.get(agent.agent_id)
            mstate = MarketState(
                tick          = tick,
                mid_price     = mid,
                best_bid      = best_bid_pre,
                best_ask      = best_ask_pre,
                spread        = self.book.spread,
                bid_levels    = snap["bids"],
                ask_levels    = snap["asks"],
                recent_trades = recent_trade_dicts,
                inventory     = pos.inventory if pos else 0,
                cash          = pos.cash      if pos else self.config.initial_cash,
                pnl           = pos.mark_to_market(mid) if pos else 0.0,
                volatility    = vol,
                vwap          = self._vwap(mid),
                crowding_intensity = self.crowding_matrix.intensity,
                agent_crowding_intensity = float(agent_crowding.get(agent.agent_id, 0.0)),
                crowding_side_pressure = self.alpha_decay.side_pressure,
                impact_buy_mult = impact_buy_mult,
                impact_sell_mult = impact_sell_mult,
                agent_impact_mult = float(agent_impact_mult.get(agent.agent_id, 1.0)),
            )
            try:
                orders = agent.generate_orders(mstate)
                if orders:
                    all_orders.extend(orders)
            except (TypeError, ValueError):
                # Interface contract violations are fatal in research mode.
                raise
            except Exception:
                pass  # agent errors don't crash the simulation

        # --- 3. Shuffle order submission (no first-mover advantage) ---
        random.shuffle(all_orders)

        # --- 4. Match ---
        fills = self.matching_engine.process(all_orders)
        self._recent_fills = fills
        fill_dicts = [
            {
                "tick": tick,
                "price": f.price,
                "qty": f.qty,
                "buy_agent": f.buy_agent_id[:8],
                "sell_agent": f.sell_agent_id[:8],
                "timestamp": f.timestamp,
            }
            for f in fills
        ]
        if fill_dicts:
            self._trade_tape.extend(fill_dicts)
            if len(self._trade_tape) > 500:
                self._trade_tape = self._trade_tape[-500:]

        # --- 5. Execution accounting ---
        current_mid = self.book.mid_price or mid
        self.execution_engine.apply_fills(fills, current_mid)

        # Legacy cumulative VWAP bookkeeping (rolling VWAP is used in state).
        for fill in fills:
            self._vwap_num += fill.price * fill.qty
            self._vwap_den += fill.qty

        # --- 6. Sync agent positions from execution engine ---
        for agent in self.agents:
            pos = self.execution_engine.positions.get(agent.agent_id)
            if pos:
                agent.sync_position(pos.cash, pos.inventory, pos.mark_to_market(current_mid))

        # --- 7. Price history ---
        if current_mid:
            self._price_history.append(current_mid)
            if len(self._price_history) > 1000:
                self._price_history.pop(0)

        # Post-trade book snapshot for analytics and API state.
        snap_post = self.book.depth_snapshot(n_levels=10)

        # --- 8. Crowding pipeline ---
        agent_ids = [a.agent_id for a in self.agents]
        F = self.factor_space.update(
            agents=self.agents,
            tick=tick,
            mid_price=current_mid,
            best_bid=best_bid_pre,
            best_ask=best_ask_pre,
            orders=all_orders,
            fills=fills,
        )
        activity_weights = self.factor_space.activity_weights(agent_ids)
        intensity = self.crowding_matrix.update(
            F,
            agent_ids,
            activity_weights=activity_weights,
        )
        self.alpha_decay.update(
            crowding_intensity=intensity,
            agents=self.agents,
            agent_activity=self.factor_space.latest_activity_map(),
            crowding_by_agent=self.crowding_matrix.agent_intensity_map,
            order_flow_imbalance=self._order_flow_imbalance(all_orders),
        )

        # --- 9. Analytics ---
        self.diagnostics.update(
            agents=self.agents,
            fills=fills,
            mid_pre=mid,
            mid_post=current_mid,
        )
        book_depth = sum(q for _, q in snap_post["bids"]) + sum(q for _, q in snap_post["asks"])
        lfi = self.fragility.update(
            tick=tick,
            bid_levels=snap_post["bids"],
            ask_levels=snap_post["asks"],
            mid_price=current_mid,
            fills=fills,
        )
        regime = self.regime_detector.update(
            tick=tick,
            mid_price=current_mid,
            spread=self.book.spread,
            depth=float(book_depth),
            lfi=lfi,
            crowding=intensity,
            fill_imbalance=self.fragility.fill_imbalance,
        )
        # --- 10. Assemble TickState ---

        state = TickState(
            tick           = tick,
            timestamp      = time.time_ns(),
            mid_price      = round(current_mid, 4) if current_mid else None,
            spread         = round(self.book.spread, 4) if self.book.spread else None,
            vwap           = round(self._vwap(current_mid), 4),
            volatility     = round(vol, 6),
            regime         = regime.value,
            lfi            = round(lfi, 4),
            lfi_near_depth_ratio = round(self.fragility.near_depth_ratio, 6),
            lfi_alert      = self.fragility.alert_level,
            crowding       = round(intensity, 4),
            order_book     = snap_post,
            recent_fills   = fill_dicts,
            agent_stats    = self.diagnostics.snapshot_all(self.agents),
            crowding_data  = self.crowding_matrix.snapshot_for_api(),
            factor_data    = self.factor_space.snapshot_for_api(),
            decay_data     = self.alpha_decay.snapshot_for_api(),
        )
        self._last_state = state
        events = self._build_tick_events(
            tick=tick,
            timestamp=state.timestamp,
            all_orders=all_orders,
            fills=fills,
            mid_pre=mid,
            mid_post=current_mid,
            spread_pre=snap.get("spread"),
            spread_post=snap_post.get("spread"),
            pre_agent_state=pre_agent_state,
            post_agent_state=self.execution_engine.snapshot(current_mid),
            crowding_snapshot=state.crowding_data,
            regime_value=state.regime,
            tick_event=state.to_ws_event(),
        )
        self._last_events = events
        return state, events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_tick_events(
        self,
        tick: int,
        timestamp: int,
        all_orders: List[Order],
        fills: List[Fill],
        mid_pre: float,
        mid_post: float,
        spread_pre: Optional[float],
        spread_post: Optional[float],
        pre_agent_state: Dict[str, Dict[str, float]],
        post_agent_state: List[Dict[str, Any]],
        crowding_snapshot: Dict[str, Any],
        regime_value: str,
        tick_event: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        for order in all_orders:
            events.append(
                make_event(
                    EventType.ORDER_SUBMITTED,
                    {
                        "tick": tick,
                        "order_id": order.order_id,
                        "agent_id": order.agent_id,
                        "side": order.side.value,
                        "order_type": order.order_type.value,
                        "price": round(float(order.price), 6),
                        "qty": int(order.qty),
                        "cancel_target": order.cancel_target,
                    },
                    timestamp=order.timestamp,
                )
            )

        for fill in fills:
            events.append(
                make_event(
                    EventType.ORDER_FILLED,
                    {
                        "tick": tick,
                        "buy_order_id": fill.buy_order_id,
                        "sell_order_id": fill.sell_order_id,
                        "buy_agent_id": fill.buy_agent_id,
                        "sell_agent_id": fill.sell_agent_id,
                        "price": round(float(fill.price), 6),
                        "qty": int(fill.qty),
                    },
                    timestamp=fill.timestamp,
                )
            )

        if abs(mid_post - mid_pre) > 1e-12 or self._norm_spread(spread_pre) != self._norm_spread(spread_post):
            events.append(
                make_event(
                    EventType.PRICE_UPDATED,
                    {
                        "tick": tick,
                        "prev_mid_price": round(float(mid_pre), 6),
                        "mid_price": round(float(mid_post), 6),
                        "prev_spread": self._norm_spread(spread_pre),
                        "spread": self._norm_spread(spread_post),
                    },
                    timestamp=timestamp,
                )
            )

        changed_agents: List[Dict[str, Any]] = []
        for cur in post_agent_state:
            aid = str(cur["agent_id"])
            prev = pre_agent_state.get(aid)
            cash = float(cur.get("cash", 0.0))
            inv = int(cur.get("inventory", 0))
            pnl = float(cur.get("pnl", 0.0))
            if prev is None:
                changed_agents.append(cur)
                continue
            if (
                abs(cash - prev["cash"]) > 1e-9
                or abs(float(inv) - prev["inventory"]) > 1e-9
                or abs(pnl - prev["pnl"]) > 1e-9
            ):
                changed_agents.append(cur)

        if changed_agents:
            events.append(
                make_event(
                    EventType.AGENT_STATE_CHANGED,
                    {"tick": tick, "agents": changed_agents},
                    timestamp=timestamp,
                )
            )

        events.append(
            make_event(
                EventType.CROWDING_MATRIX_UPDATED,
                {
                    "tick": tick,
                    "crowding_intensity": crowding_snapshot.get("crowding_intensity", 0.0),
                    "agent_ids": crowding_snapshot.get("agent_ids", []),
                    "matrix": crowding_snapshot.get("matrix", []),
                    "agent_crowding_intensity": crowding_snapshot.get("agent_crowding_intensity", {}),
                    "top_crowded_pairs": crowding_snapshot.get("top_crowded_pairs", []),
                },
                timestamp=timestamp,
            )
        )

        if self._last_regime is None or regime_value != self._last_regime:
            events.append(
                make_event(
                    EventType.REGIME_CHANGED,
                    {
                        "tick": tick,
                        "prev_regime": self._last_regime,
                        "regime": regime_value,
                    },
                    timestamp=timestamp,
                )
            )
        self._last_regime = regime_value

        events.append(tick_event)
        return events

    @staticmethod
    def _norm_spread(spread: Optional[float]) -> Optional[float]:
        if spread is None:
            return None
        return round(float(spread), 6)

    def _on_fill(self, fill: Fill) -> None:
        self._recent_fills.append(fill)

    def _rolling_vol(self, window: int = 20) -> float:
        if len(self._price_history) < 2:
            return 0.0
        recent = self._price_history[-window:]
        if len(recent) < 2:
            return 0.0
        log_rets = np.diff(np.log(np.array(recent) + 1e-9))
        return float(np.std(log_rets))

    def _vwap(self, fallback_price: Optional[float] = None, window: int = 200) -> float:
        """
        Rolling VWAP from recent fills.

        Using a rolling window avoids stale fair-value lines when the market
        drifts over long sessions with regime changes.
        """
        tape = self._trade_tape[-max(1, int(window)):]
        if not tape:
            if fallback_price is not None:
                return float(fallback_price)
            return self.config.initial_price

        num = 0.0
        den = 0.0
        for tr in tape:
            price = float(tr.get("price", 0.0))
            qty = float(tr.get("qty", 0.0))
            if price <= 0.0 or qty <= 0.0:
                continue
            num += price * qty
            den += qty

        if den < 1e-9:
            if fallback_price is not None:
                return float(fallback_price)
            return self.config.initial_price
        return num / den

    def _order_flow_imbalance(self, orders: List[Order]) -> float:
        buy = 0.0
        sell = 0.0
        for order in orders:
            if order.order_type == OrderType.CANCEL:
                continue
            if order.side == Side.BID:
                buy += float(order.qty)
            else:
                sell += float(order.qty)
        total = buy + sell
        if total < 1e-9:
            return 0.0
        return float((buy - sell) / total)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CrowdAlpha simulation")
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tick-delay", type=float, default=0.0)
    parser.add_argument("--mom", type=int, default=4, help="number of momentum agents")
    parser.add_argument("--rev", type=int, default=4, help="number of mean-reversion agents")
    parser.add_argument("--mm", type=int, default=2, help="number of market makers")
    parser.add_argument("--rl", type=int, default=0, help="number of RL agents")
    args = parser.parse_args()

    cfg = SimulationConfig(
        n_momentum=args.mom,
        n_mean_reversion=args.rev,
        n_market_makers=args.mm,
        n_rl=args.rl,
        seed=args.seed,
        tick_delay_ms=args.tick_delay,
    )
    sim = Simulation(cfg)
    print("=== CrowdAlpha Simulation ===")
    print(f"Agents: {[a.agent_id for a in sim.agents]}")
    print(f"Running {args.ticks} ticks...\n")
    states = sim.run(n_ticks=args.ticks, print_prices=True)
    if states:
        mid_start = states[0].mid_price or cfg.initial_price
        mid_end = states[-1].mid_price or mid_start
        returns = np.diff(np.log(np.array([s.mid_price for s in states if s.mid_price] or [mid_start])))
        realized_vol = float(np.std(returns)) if len(returns) > 0 else 0.0
        print(
            f"\nSummary: start={mid_start:.4f} end={mid_end:.4f} "
            f"ret={((mid_end / mid_start) - 1.0) * 100:.2f}% "
            f"vol={realized_vol:.6f}"
        )
    print("\nDone.")

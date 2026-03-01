"""
engine/agents/user_strategy_worker.py

Subprocess worker used by SandboxedUserAgent.
Protocol: JSON lines over stdin/stdout.
"""

from __future__ import annotations

import inspect
import json
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from engine.agents.base_agent import BaseAgent, MarketState
from engine.agents.user_strategy_sandbox import validate_user_strategy_source
from engine.core.order import Order, OrderType, Side


def _safe_builtins() -> Dict[str, Any]:
    return {
        "__build_class__": __build_class__,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
        "round": round,
        "int": int,
        "float": float,
        "bool": bool,
        "str": str,
        "super": super,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "print": print,
    }


def _serialize_orders(orders: List[Order]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for order in orders:
        out.append(
            {
                "side": order.side.value,
                "order_type": order.order_type.value,
                "price": float(order.price),
                "qty": int(order.qty),
                "cancel_target": order.cancel_target,
            }
        )
    return out


def _serialize_factor(vec: Any) -> List[float]:
    arr = np.asarray(vec, dtype=float).flatten()
    if arr.size == 0:
        return [0.0] * 6
    return [float(x) for x in arr[:16]]


class _WorkerRuntime:
    def __init__(self) -> None:
        self.agent: Optional[BaseAgent] = None

    def init(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        strategy_name = str(payload["strategy_name"])
        code = str(payload["code"])
        agent_id = str(payload["agent_id"])
        config = payload.get("config") or {}
        initial_cash = float(payload.get("initial_cash", 100_000.0))

        validate_user_strategy_source(code)
        compiled = compile(code, "<user_strategy>", "exec")

        globals_ns: Dict[str, Any] = {
            "__builtins__": _safe_builtins(),
            "__name__": "__user_strategy__",
            "BaseAgent": BaseAgent,
            "MarketState": MarketState,
            "Order": Order,
            "OrderType": OrderType,
            "Side": Side,
            "np": np,
        }
        locals_ns: Dict[str, Any] = {}
        exec(compiled, globals_ns, locals_ns)

        candidates = [
            v
            for v in locals_ns.values()
            if inspect.isclass(v) and issubclass(v, BaseAgent) and v is not BaseAgent
        ]
        if len(candidates) != 1:
            raise ValueError("Strategy code must define exactly one BaseAgent subclass.")
        cls = candidates[0]
        if not hasattr(cls, "on_tick") or not hasattr(cls, "factor_vector"):
            raise ValueError("User strategy must implement on_tick and factor_vector.")

        try:
            self.agent = cls(
                agent_id=agent_id,
                strategy_type=strategy_name,
                initial_cash=initial_cash,
                **config,
            )
        except TypeError:
            self.agent = cls(agent_id=agent_id, initial_cash=initial_cash, **config)

        return {"ok": True}

    def on_tick(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.agent is None:
            raise RuntimeError("Worker not initialized.")

        state = MarketState(
            tick=int(payload["tick"]),
            mid_price=payload.get("mid_price"),
            best_bid=payload.get("best_bid"),
            best_ask=payload.get("best_ask"),
            spread=payload.get("spread"),
            bid_levels=tuple(tuple(x) for x in payload.get("bid_levels", [])),
            ask_levels=tuple(tuple(x) for x in payload.get("ask_levels", [])),
            recent_trades=tuple(payload.get("recent_trades", [])),
            inventory=int(payload.get("inventory", 0)),
            cash=float(payload.get("cash", 0.0)),
            pnl=float(payload.get("pnl", 0.0)),
            volatility=float(payload.get("volatility", 0.0)),
            vwap=float(payload.get("vwap", 0.0)),
            crowding_intensity=float(payload.get("crowding_intensity", 0.0)),
            crowding_side_pressure=float(payload.get("crowding_side_pressure", 0.0)),
            impact_buy_mult=float(payload.get("impact_buy_mult", 1.0)),
            impact_sell_mult=float(payload.get("impact_sell_mult", 1.0)),
        )
        self.agent.sync_position(state.cash, state.inventory, state.pnl)
        orders = self.agent.generate_orders(state)
        factor = self.agent.factor_vector()
        return {"orders": _serialize_orders(orders), "factor_vector": _serialize_factor(factor)}

    def factor_vector(self) -> Dict[str, Any]:
        if self.agent is None:
            raise RuntimeError("Worker not initialized.")
        return {"factor_vector": _serialize_factor(self.agent.factor_vector())}


def _send(resp: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()


def main() -> int:
    runtime = _WorkerRuntime()
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            op = req.get("op")
            payload = req.get("payload") or {}

            if op == "init":
                result = runtime.init(payload)
            elif op == "on_tick":
                result = runtime.on_tick(payload)
            elif op == "factor_vector":
                result = runtime.factor_vector()
            elif op == "shutdown":
                _send({"id": req_id, "ok": True, "result": {"shutdown": True}})
                return 0
            else:
                raise ValueError(f"Unknown op '{op}'")

            _send({"id": req_id, "ok": True, "result": result})
        except Exception as exc:
            _send({"id": req_id, "ok": False, "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
engine/agents/sandboxed_user_agent.py

BaseAgent wrapper that executes user strategy code in a subprocess with
strict per-request timeouts.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from engine.core.order import Order, OrderType, Side

from .base_agent import BaseAgent, MarketState
from .user_strategy_sandbox import validate_user_strategy_source


class SandboxedUserAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        strategy_name: str,
        source_code: str,
        timeout_ms: int = 120,
        startup_timeout_ms: int = 1200,
        max_orders_per_tick: int = 12,
        max_qty: int = 200,
        initial_cash: float = 100_000.0,
        **user_config: Any,
    ) -> None:
        super().__init__(agent_id=agent_id, strategy_type=strategy_name, initial_cash=initial_cash)
        self._source_code = source_code
        self._timeout_s = max(0.02, float(timeout_ms) / 1000.0)
        self._startup_timeout_s = max(0.2, float(startup_timeout_ms) / 1000.0)
        self._max_orders_per_tick = max(1, int(max_orders_per_tick))
        self._max_qty = max(1, int(max_qty))
        self._user_config = dict(user_config)
        self._proc: Optional[subprocess.Popen] = None
        self._req_id = 0
        self._last_factor = np.zeros(6, dtype=float)

        validate_user_strategy_source(source_code)
        self._start_worker()

    def on_tick(self, state: MarketState) -> List[Order]:
        if self._proc is None:
            return []
        payload = {
            "tick": state.tick,
            "mid_price": state.mid_price,
            "best_bid": state.best_bid,
            "best_ask": state.best_ask,
            "spread": state.spread,
            "bid_levels": list(state.bid_levels),
            "ask_levels": list(state.ask_levels),
            "recent_trades": list(state.recent_trades),
            "inventory": state.inventory,
            "cash": state.cash,
            "pnl": state.pnl,
            "volatility": state.volatility,
            "vwap": state.vwap,
            "crowding_intensity": state.crowding_intensity,
            "crowding_side_pressure": state.crowding_side_pressure,
            "impact_buy_mult": state.impact_buy_mult,
            "impact_sell_mult": state.impact_sell_mult,
        }
        try:
            resp = self._rpc("on_tick", payload)
        except RuntimeError:
            return []

        self._last_factor = np.asarray(resp.get("factor_vector", self._last_factor), dtype=float)
        return self._deserialize_orders(resp.get("orders", []))

    def factor_vector(self) -> np.ndarray:
        if self._proc is None:
            return self._last_factor
        try:
            resp = self._rpc("factor_vector", {})
            self._last_factor = np.asarray(resp.get("factor_vector", self._last_factor), dtype=float)
        except RuntimeError:
            pass
        return self._last_factor

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            self._raw_send({"id": self._next_req_id(), "op": "shutdown", "payload": {}})
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=0.5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def __del__(self) -> None:  # pragma: no cover
        self.close()

    def _start_worker(self) -> None:
        self.close()
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "engine.agents.user_strategy_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._rpc(
            "init",
            {
                "strategy_name": self.strategy_type,
                "code": self._source_code,
                "agent_id": self.agent_id,
                "initial_cash": self.initial_cash,
                "config": self._user_config,
            },
            timeout_s=max(self._startup_timeout_s, self._timeout_s),
        )

    def _next_req_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _rpc(self, op: str, payload: Dict[str, Any], timeout_s: Optional[float] = None) -> Dict[str, Any]:
        if self._proc is None:
            raise RuntimeError("Sandbox worker is not running.")
        req_id = self._next_req_id()
        self._raw_send({"id": req_id, "op": op, "payload": payload})
        wait_s = self._timeout_s if timeout_s is None else max(0.01, float(timeout_s))
        resp_line = self._readline_with_timeout(wait_s)
        if resp_line is None:
            self._handle_worker_failure("User strategy timed out.")
            raise RuntimeError("User strategy timed out.")
        try:
            resp = json.loads(resp_line)
        except json.JSONDecodeError:
            self._handle_worker_failure("Malformed response from strategy sandbox.")
            raise RuntimeError("Malformed response from strategy sandbox.")
        if resp.get("id") != req_id:
            self._handle_worker_failure("Mismatched response id from strategy sandbox.")
            raise RuntimeError("Mismatched response id from strategy sandbox.")
        if not resp.get("ok", False):
            msg = str(resp.get("error", "User strategy execution failed."))
            self._handle_worker_failure(msg)
            raise RuntimeError(msg)
        return resp.get("result", {})

    def _raw_send(self, payload: Dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Sandbox worker stdin unavailable.")
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()

    def _readline_with_timeout(self, timeout_s: float) -> Optional[str]:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return None
        q: "queue.Queue[Any]" = queue.Queue(maxsize=1)

        def _reader() -> None:
            try:
                q.put(proc.stdout.readline())
            except Exception as exc:
                q.put(exc)

        th = threading.Thread(target=_reader, daemon=True)
        th.start()
        try:
            item = q.get(timeout=timeout_s)
        except queue.Empty:
            return None
        if isinstance(item, Exception):
            return None
        if item == "":
            return None
        return item.strip()

    def _handle_worker_failure(self, reason: str) -> None:
        self.close()
        # Keep simulation alive; this agent simply goes silent after failure.
        self._last_factor = np.zeros(6, dtype=float)
        self.config["sandbox_error"] = reason

    def _deserialize_orders(self, raw_orders: Any) -> List[Order]:
        if not isinstance(raw_orders, list):
            return []
        out: List[Order] = []
        for item in raw_orders[: self._max_orders_per_tick]:
            if not isinstance(item, dict):
                continue
            try:
                side = Side(str(item.get("side")))
                order_type = OrderType(str(item.get("order_type", "LIMIT")))
                qty = int(item.get("qty", 0))
                qty = max(1, min(qty, self._max_qty))
                price = float(item.get("price", 0.0))
                if order_type == OrderType.MARKET:
                    price = max(0.0001, price)
                else:
                    price = max(0.0001, price)
                order = Order(
                    agent_id=self.agent_id,
                    side=side,
                    order_type=order_type,
                    price=price,
                    qty=qty,
                    cancel_target=item.get("cancel_target"),
                )
                out.append(order)
            except Exception:
                continue
        return out

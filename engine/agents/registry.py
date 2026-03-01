"""
engine/agents/registry.py

Agent factory + runtime registration.

Phase 7:
  User strategies are registered as source code and instantiated via
  SandboxedUserAgent (subprocess + timeout), not direct exec in main process.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Type

from .base_agent import BaseAgent
from .market_maker import MarketMakerAgent
from .mean_reversion import MeanReversionAgent
from .momentum import MomentumAgent
from .rl_agent import RLAgent
from .sandboxed_user_agent import SandboxedUserAgent
from .user_strategy_sandbox import StrategySafetyError, validate_user_strategy_source


BUILTIN_AGENTS: Dict[str, Type[BaseAgent]] = {
    "momentum": MomentumAgent,
    "mean_reversion": MeanReversionAgent,
    "market_maker": MarketMakerAgent,
    "rl_ppo": RLAgent,
}


_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{2,63}$")


class AgentRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Type[BaseAgent]] = dict(BUILTIN_AGENTS)
        self._user_sources: Dict[str, str] = {}

    def list_strategies(self) -> list:
        return sorted(list(self._registry.keys()) + list(self._user_sources.keys()))

    def get_class(self, strategy_type: str) -> Type[BaseAgent]:
        if strategy_type in self._registry:
            return self._registry[strategy_type]
        if strategy_type in self._user_sources:
            return SandboxedUserAgent
        raise ValueError(
            f"Unknown strategy '{strategy_type}'. "
            f"Available: {self.list_strategies()}"
        )

    def create(
        self,
        strategy_type: str,
        agent_id: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseAgent:
        cfg = dict(config or {})
        if strategy_type in self._registry:
            cls = self._registry[strategy_type]
            try:
                return cls(agent_id=agent_id, **cfg)
            except TypeError as exc:
                raise ValueError(f"Bad config for strategy '{strategy_type}': {exc}") from exc

        if strategy_type in self._user_sources:
            try:
                return SandboxedUserAgent(
                    agent_id=agent_id,
                    strategy_name=strategy_type,
                    source_code=self._user_sources[strategy_type],
                    **cfg,
                )
            except Exception as exc:
                raise ValueError(f"Failed to start sandboxed strategy '{strategy_type}': {exc}") from exc

        raise ValueError(
            f"Unknown strategy '{strategy_type}'. "
            f"Available: {self.list_strategies()}"
        )

    def create_default_population(
        self,
        n_momentum: int = 3,
        n_mean_rev: int = 3,
        n_market_makers: int = 2,
        n_rl: int = 1,
    ) -> list:
        agents = []
        for i in range(n_momentum):
            fast = 5 + i
            slow = 15 + i * 3
            agents.append(
                self.create("momentum", f"mom_{i+1}", {"fast_window": fast, "slow_window": slow})
            )

        for i in range(n_mean_rev):
            thresh = 1.2 + i * 0.2
            agents.append(self.create("mean_reversion", f"rev_{i+1}", {"threshold": thresh}))

        for i in range(n_market_makers):
            agents.append(
                self.create(
                    "market_maker",
                    f"mm_{i+1}",
                    {
                        "base_spread": 0.05 + i * 0.02,
                        "min_half_spread": 0.01,
                        "quote_jitter": 0.02 + i * 0.01,
                    },
                )
            )

        for i in range(n_rl):
            agents.append(self.create("rl_ppo", f"rl_{i+1}"))

        return agents

    def register_user_strategy(self, strategy_name: str, code: str) -> Type[BaseAgent]:
        name = str(strategy_name).strip()
        if not _NAME_RE.match(name):
            raise ValueError(
                "strategy_name must match ^[a-zA-Z][a-zA-Z0-9_]{2,63}$"
            )
        if name in BUILTIN_AGENTS:
            raise ValueError(f"'{name}' is reserved for a built-in strategy.")

        try:
            validate_user_strategy_source(code)
        except StrategySafetyError as exc:
            raise ValueError(str(exc)) from exc

        # Probe by starting/stopping a sandbox instance.
        probe = SandboxedUserAgent(
            agent_id=f"probe_{name}",
            strategy_name=name,
            source_code=code,
            timeout_ms=150,
            max_orders_per_tick=3,
        )
        probe.close()

        self._user_sources[name] = code
        return SandboxedUserAgent


default_registry = AgentRegistry()

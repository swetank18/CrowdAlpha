"""
engine/agents/registry.py

Agent factory and plug-in system.

Responsibilities:
  - Map strategy name strings → Agent classes
  - Instantiate agents with validated configs
  - Support runtime registration (for user-submitted strategies)

User-submitted strategies:
  Strategy code is executed in a restricted namespace.
  The class must subclass BaseAgent and implement on_tick + factor_vector.
  The registry validates the interface before allowing registration.
"""

from __future__ import annotations

import inspect
from typing import Dict, Type, Any, Optional

import numpy as np

from .base_agent import BaseAgent, MarketState
from .momentum import MomentumAgent
from .mean_reversion import MeanReversionAgent
from .market_maker import MarketMakerAgent
from .rl_agent import RLAgent


# ---------------------------------------------------------------------------
# Built-in agent registry
# ---------------------------------------------------------------------------

BUILTIN_AGENTS: Dict[str, Type[BaseAgent]] = {
    "momentum":       MomentumAgent,
    "mean_reversion": MeanReversionAgent,
    "market_maker":   MarketMakerAgent,
    "rl_ppo":         RLAgent,
}


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------

class AgentRegistry:

    def __init__(self) -> None:
        self._registry: Dict[str, Type[BaseAgent]] = dict(BUILTIN_AGENTS)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_strategies(self) -> list:
        return list(self._registry.keys())

    def get_class(self, strategy_type: str) -> Type[BaseAgent]:
        if strategy_type not in self._registry:
            raise ValueError(
                f"Unknown strategy '{strategy_type}'. "
                f"Available: {self.list_strategies()}"
            )
        return self._registry[strategy_type]

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def create(
        self,
        strategy_type: str,
        agent_id:      str,
        config:        Optional[Dict[str, Any]] = None,
    ) -> BaseAgent:
        """
        Create an agent instance. Config dict is passed as kwargs to
        the agent class constructor — unknown kwargs are silently ignored
        via **config, so each agent picks what it needs.
        """
        cls = self.get_class(strategy_type)
        cfg = config or {}
        try:
            return cls(agent_id=agent_id, **cfg)
        except TypeError as e:
            raise ValueError(f"Bad config for strategy '{strategy_type}': {e}")

    def create_default_population(
        self,
        n_momentum:      int = 3,
        n_mean_rev:      int = 3,
        n_market_makers: int = 2,
        n_rl:            int = 1,
    ) -> list:
        """
        Create a default mixed population for simulation startup.
        Returns a list of BaseAgent instances.
        """
        agents = []
        for i in range(n_momentum):
            # Vary EMA windows slightly to prevent perfect crowding at start
            fast = 5 + i
            slow = 15 + i * 3
            agents.append(self.create("momentum", f"mom_{i+1}",
                                      {"fast_window": fast, "slow_window": slow}))

        for i in range(n_mean_rev):
            thresh = 1.2 + i * 0.2
            agents.append(self.create("mean_reversion", f"rev_{i+1}",
                                      {"threshold": thresh}))

        for i in range(n_market_makers):
            agents.append(self.create("market_maker", f"mm_{i+1}",
                                      {"base_spread": 0.3 + i * 0.1}))

        for i in range(n_rl):
            agents.append(self.create("rl_ppo", f"rl_{i+1}"))

        return agents

    # ------------------------------------------------------------------
    # Runtime registration (user-submitted strategies)
    # ------------------------------------------------------------------

    def register_user_strategy(
        self,
        strategy_name: str,
        code: str,
    ) -> Type[BaseAgent]:
        """
        Compile and register a user-submitted Python strategy.

        The code must define exactly one class that subclasses BaseAgent.
        IMPORTANT: This is NOT sandboxed — for research/trusted environments only.
        Production deployments must add RestrictedPython or subprocess isolation.
        """
        namespace: Dict[str, Any] = {
            "BaseAgent":   BaseAgent,
            "MarketState": MarketState,
            "np":          np,
        }

        try:
            exec(compile(code, "<user_strategy>", "exec"), namespace)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in strategy code: {e}")

        # Find the user-defined class
        user_classes = [
            v for v in namespace.values()
            if inspect.isclass(v)
            and issubclass(v, BaseAgent)
            and v is not BaseAgent
        ]

        if not user_classes:
            raise ValueError(
                "No BaseAgent subclass found in submitted code. "
                "Your class must inherit from BaseAgent."
            )
        if len(user_classes) > 1:
            raise ValueError(
                "Multiple BaseAgent subclasses found. Submit exactly one."
            )

        cls = user_classes[0]
        self._validate_interface(cls)
        self._registry[strategy_name] = cls
        return cls

    def _validate_interface(self, cls: Type[BaseAgent]) -> None:
        """Verify the class implements the required interface."""
        if not hasattr(cls, "on_tick"):
            raise ValueError("Strategy must implement on_tick(state) -> list[Order]")
        if not hasattr(cls, "factor_vector"):
            raise ValueError("Strategy must implement factor_vector() -> np.ndarray")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

default_registry = AgentRegistry()

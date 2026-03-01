"""
engine/agents/rl_agent.py

Reinforcement Learning agent using PPO (Proximal Policy Optimization)
via stable-baselines3.

Architecture:
  - Observation space: 8-dimensional continuous vector
      [mid_price_normalized, spread, inventory_normalized,
       volatility, momentum_5, momentum_20, bid_depth, ask_depth]
  - Action space: Discrete(5)
      0 = HOLD
      1 = BUY_SMALL  (qty=2)
      2 = BUY_LARGE  (qty=8)
      3 = SELL_SMALL (qty=2)
      4 = SELL_LARGE (qty=8)

Training:
  - The agent is trained OFFLINE via train_rl_agent() standalone script
  - A frozen policy (.zip file) is loaded at runtime
  - If no model file exists, the agent falls back to random actions

Factor vector:
  - Based on the network's softmax output probabilities → reveals
    what the agent "thinks" the market is doing

This is academically interesting because a trained PPO policy can
be compared to hand-coded strategies in the crowding matrix —
does RL converge to momentum or mean-reversion patterns?
"""

from __future__ import annotations

import os
import numpy as np
from typing import List, Optional
from engine.core.order import Order, Side, OrderType
from .base_agent import BaseAgent, MarketState


# Sentinel: stable-baselines3 is optional (training mode only)
try:
    from stable_baselines3 import PPO as SB3_PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class RLAgent(BaseAgent):

    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "../../models/rl_ppo_policy.zip"
    )

    def __init__(
        self,
        agent_id:   str,
        model_path: Optional[str] = None,
        max_inv:    int = 50,
        initial_cash: float = 100_000.0,
    ) -> None:
        super().__init__(agent_id, "rl_ppo", initial_cash)
        self.max_inv     = max_inv
        self.model_path  = model_path or self.DEFAULT_MODEL_PATH
        self._model      = None
        self._last_probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        self._price_history: List[float] = []

        # Try to load frozen policy
        self._load_model()

    # ------------------------------------------------------------------

    def on_tick(self, state: MarketState) -> List[Order]:
        mid = state.mid_or_last
        self._price_history.append(mid)

        obs = self._build_observation(state)
        action = self._select_action(obs)
        return self._action_to_orders(action, mid)

    def factor_vector(self) -> np.ndarray:
        """
        Use the policy's action probabilities as the factor vector.
        If model unavailable, return uniform distribution.

        We project 5-action probs into our 5-dim factor space:
          [0]  momentum_like  = (BUY_LARGE - SELL_LARGE) normalized
          [1]  mean_rev_like  = (BUY_SMALL + SELL_SMALL) relative
          [2]  bid_agg        = BUY_SMALL + BUY_LARGE probs
          [3]  ask_agg        = SELL_SMALL + SELL_LARGE probs
          [4]  turnover_rate  = 1 - HOLD
        """
        p = self._last_probs
        momentum  = np.clip((p[2] - p[4]), -1.0, 1.0)
        mean_rev  = np.clip((p[1] + p[3]) - p[0], -1.0, 1.0)
        bid_agg   = float(p[1] + p[2])
        ask_agg   = float(p[3] + p[4])
        turnover  = float(1.0 - p[0])
        return np.array([momentum, mean_rev, bid_agg, ask_agg, turnover])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_observation(self, state: MarketState) -> np.ndarray:
        """Build normalized 8-dim observation vector."""
        mid = state.mid_or_last
        spread   = (state.spread or 1.0) / mid
        inv_norm = np.clip(self.inventory / self.max_inv, -1.0, 1.0)
        vol      = state.volatility

        # Momentum features
        hist = self._price_history
        mom5  = (mid / hist[-5]  - 1.0) if len(hist) >= 5  else 0.0
        mom20 = (mid / hist[-20] - 1.0) if len(hist) >= 20 else 0.0

        # Depth imbalance
        bid_depth = sum(q for _, q in state.bid_levels[:3]) if state.bid_levels else 0
        ask_depth = sum(q for _, q in state.ask_levels[:3]) if state.ask_levels else 0
        total_depth = bid_depth + ask_depth + 1e-9
        depth_imbalance = (bid_depth - ask_depth) / total_depth

        return np.array([
            np.clip(mid / 100.0, 0.5, 5.0),   # rough price normalization
            spread,
            inv_norm,
            vol,
            np.clip(mom5,  -0.05, 0.05),
            np.clip(mom20, -0.10, 0.10),
            depth_imbalance,
            float(self.pnl / 1000.0),          # PnL in thousands
        ], dtype=np.float32)

    def _select_action(self, obs: np.ndarray) -> int:
        if self._model is not None:
            action, _ = self._model.predict(obs, deterministic=False)
            # Approximate action probabilities from policy
            try:
                import torch
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    dist = self._model.policy.get_distribution(obs_t)
                    self._last_probs = dist.distribution.probs.numpy()[0]
            except Exception:
                pass
            return int(action)
        else:
            # Fallback: random action weighted toward HOLD
            action = np.random.choice(5, p=[0.4, 0.15, 0.1, 0.15, 0.2])
            self._last_probs = np.array([0.4, 0.15, 0.1, 0.15, 0.2])
            return action

    def _action_to_orders(self, action: int, mid: float) -> List[Order]:
        action_map = {
            0: None,           # HOLD
            1: ("BID", 2),     # BUY_SMALL
            2: ("BID", 8),     # BUY_LARGE
            3: ("ASK", 2),     # SELL_SMALL
            4: ("ASK", 8),     # SELL_LARGE
        }
        spec = action_map.get(action)
        if spec is None:
            return []
        side_str, qty = spec
        side = Side.BID if side_str == "BID" else Side.ASK

        # Inventory guard
        if side == Side.BID and self.inventory >= self.max_inv:
            return []
        if side == Side.ASK and self.inventory <= -self.max_inv:
            return []

        # Aggressive limit order (crosses spread by 0.1%)
        aggression = 0.001
        if side == Side.BID:
            price = round(mid * (1 + aggression), 4)
        else:
            price = round(mid * (1 - aggression), 4)

        return [Order(
            agent_id=self.agent_id, side=side,
            order_type=OrderType.LIMIT,
            price=price, qty=qty,
        )]

    def _load_model(self) -> None:
        if not SB3_AVAILABLE:
            return
        if os.path.exists(self.model_path):
            try:
                self._model = SB3_PPO.load(self.model_path)
            except Exception as e:
                print(f"[RLAgent] Could not load model: {e}")

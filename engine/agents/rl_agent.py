"""
engine/agents/rl_agent.py

Online-learning RL agent.

Design:
  - Observation: vectorized MarketState features (price, spread, depth,
    inventory, PnL, volatility, crowding, impact skew).
  - Action: discrete side/size/aggressiveness bundles.
  - Reward: risk-adjusted PnL delta minus estimated impact cost.

If stable-baselines3 (and gymnasium) are installed, PPO is trained online
from the agent's own transition stream during simulation. Otherwise the
agent degrades to stochastic policy sampling with the same action space.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from engine.core.order import Order, OrderType, Side
from .base_agent import BaseAgent, MarketState


try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    SB3_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    SB3_AVAILABLE = False
    gym = None
    spaces = None
    PPO = None
    DummyVecEnv = None


@dataclass(frozen=True)
class _ActionSpec:
    side: Optional[Side]
    qty: int
    aggressiveness: float


@dataclass(frozen=True)
class _Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray


ACTION_SET: Tuple[_ActionSpec, ...] = (
    _ActionSpec(side=None, qty=0, aggressiveness=0.0),          # HOLD
    _ActionSpec(side=Side.BID, qty=2, aggressiveness=0.25),     # BUY small passive
    _ActionSpec(side=Side.BID, qty=5, aggressiveness=0.60),     # BUY medium
    _ActionSpec(side=Side.BID, qty=8, aggressiveness=0.95),     # BUY large aggressive
    _ActionSpec(side=Side.ASK, qty=2, aggressiveness=0.25),     # SELL small passive
    _ActionSpec(side=Side.ASK, qty=5, aggressiveness=0.60),     # SELL medium
    _ActionSpec(side=Side.ASK, qty=8, aggressiveness=0.95),     # SELL large aggressive
)


if SB3_AVAILABLE:

    class _TransitionReplayEnv(gym.Env):
        """Small env that replays in-sim transitions for PPO updates."""

        metadata = {"render_modes": []}

        def __init__(self, obs_dim: int, n_actions: int) -> None:
            super().__init__()
            self.observation_space = spaces.Box(
                low=-5.0,
                high=5.0,
                shape=(obs_dim,),
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(n_actions)
            self._transitions: List[_Transition] = []
            self._idx = 0

        def set_transitions(self, transitions: List[_Transition]) -> None:
            self._transitions = transitions
            self._idx = 0

        def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
            super().reset(seed=seed)
            self._idx = 0
            if not self._transitions:
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}
            return self._transitions[0].obs.astype(np.float32), {}

        def step(self, action: int):
            if not self._transitions:
                return (
                    np.zeros(self.observation_space.shape, dtype=np.float32),
                    0.0,
                    True,
                    False,
                    {},
                )

            idx = min(self._idx, len(self._transitions) - 1)
            transition = self._transitions[idx]
            reward = float(transition.reward)
            # Small mismatch penalty keeps the learner anchored to the
            # behavior-policy transitions gathered from the live market.
            if int(action) != int(transition.action):
                reward -= 0.01

            self._idx += 1
            terminated = self._idx >= len(self._transitions)
            if terminated:
                next_obs = transition.next_obs
            else:
                next_obs = self._transitions[self._idx].obs
            return next_obs.astype(np.float32), reward, terminated, False, {}


class RLAgent(BaseAgent):
    """PPO-based agent with online adaptation inside the simulation loop."""

    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(__file__),
        "../../models/rl_ppo_policy.zip",
    )

    def __init__(
        self,
        agent_id: str,
        model_path: Optional[str] = None,
        max_inv: int = 60,
        initial_cash: float = 100_000.0,
        enable_online_learning: bool = True,
        train_every: int = 40,
        warmup_ticks: int = 80,
        learn_steps: int = 96,
        transition_buffer: int = 1500,
        reward_scale: float = 25.0,
        risk_aversion: float = 2.0,
        impact_penalty: float = 1.0,
        exploration_start: float = 0.35,
        exploration_end: float = 0.05,
        exploration_decay: float = 0.997,
        save_every: int = 400,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(agent_id, "rl_ppo", initial_cash)
        self.max_inv = max(1, int(max_inv))
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.enable_online_learning = bool(enable_online_learning)
        self.train_every = max(10, int(train_every))
        self.warmup_ticks = max(20, int(warmup_ticks))
        self.learn_steps = max(32, int(learn_steps))
        self.reward_scale = max(1e-6, float(reward_scale))
        self.risk_aversion = max(0.0, float(risk_aversion))
        self.impact_penalty = max(0.0, float(impact_penalty))
        self.exploration_prob = float(np.clip(exploration_start, 0.0, 1.0))
        self.exploration_end = float(np.clip(exploration_end, 0.0, 1.0))
        self.exploration_decay = float(np.clip(exploration_decay, 0.9, 1.0))
        self.save_every = max(0, int(save_every))
        self.seed = seed

        self._obs_dim = 14
        self._n_actions = len(ACTION_SET)
        self._tick_count = 0

        self._prev_obs: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None
        self._prev_context: Optional[Dict[str, float]] = None
        self._mid_history: Deque[float] = deque(maxlen=200)
        self._reward_history: Deque[float] = deque(maxlen=200)
        self._transitions: Deque[_Transition] = deque(maxlen=max(200, int(transition_buffer)))

        self._last_probs = np.ones(self._n_actions, dtype=float) / float(self._n_actions)
        self._last_impact_cost = 0.0
        self._last_reward = 0.0
        self._last_risk_penalty = 0.0

        self._model = None
        self._train_env = None
        if SB3_AVAILABLE and self.enable_online_learning:
            self._init_model()

    def on_tick(self, state: MarketState) -> List[Order]:
        self._tick_count += 1
        mid = state.mid_or_last
        self._mid_history.append(mid)

        obs = self._build_observation(state)
        self._ingest_transition_if_ready(obs, state)
        self._maybe_online_train()

        action = self._select_action(obs)
        action_spec = ACTION_SET[action]
        self._prev_obs = obs
        self._prev_action = action
        self._prev_context = {
            "pnl": float(state.pnl),
            "inventory": float(state.inventory),
            "mid": float(mid),
            "spread": float(state.spread or max(0.01, 0.0005 * mid)),
            "impact_buy_mult": float(state.impact_buy_mult),
            "impact_sell_mult": float(state.impact_sell_mult),
            "qty": float(action_spec.qty),
            "aggressiveness": float(action_spec.aggressiveness),
            "is_buy": 1.0 if action_spec.side == Side.BID else 0.0,
            "is_sell": 1.0 if action_spec.side == Side.ASK else 0.0,
        }

        self.exploration_prob = max(
            self.exploration_end,
            self.exploration_prob * self.exploration_decay,
        )
        return self._action_to_orders(action, state)

    def factor_vector(self) -> np.ndarray:
        """
        Agent behavior signature derived from action probabilities and
        realized reward stats.
        """
        p = self._last_probs
        buy_pressure = float(np.sum(p[1:4]))
        sell_pressure = float(np.sum(p[4:7]))
        aggr = float(
            p[1] * ACTION_SET[1].aggressiveness
            + p[2] * ACTION_SET[2].aggressiveness
            + p[3] * ACTION_SET[3].aggressiveness
            + p[4] * ACTION_SET[4].aggressiveness
            + p[5] * ACTION_SET[5].aggressiveness
            + p[6] * ACTION_SET[6].aggressiveness
        )
        inv_norm = float(np.clip(self.inventory / self.max_inv, -1.0, 1.0))
        reward_mean = float(np.mean(self._reward_history)) if self._reward_history else 0.0
        turnover = float(1.0 - p[0])
        return np.array(
            [buy_pressure, sell_pressure, aggr, inv_norm, reward_mean, turnover],
            dtype=float,
        )

    def _build_observation(self, state: MarketState) -> np.ndarray:
        mid = state.mid_or_last
        spread = float(state.spread or max(0.01, 0.0005 * mid))
        vwap = float(state.vwap or mid)
        inv_norm = float(np.clip(state.inventory / self.max_inv, -1.0, 1.0))
        pnl_norm = float(np.clip(state.pnl / max(1.0, self.initial_cash), -2.0, 2.0))

        bid_depth = float(sum(q for _, q in state.bid_levels[:3])) if state.bid_levels else 0.0
        ask_depth = float(sum(q for _, q in state.ask_levels[:3])) if state.ask_levels else 0.0
        total_depth = bid_depth + ask_depth + 1e-9
        depth_imbalance = float((bid_depth - ask_depth) / total_depth)

        trades = state.recent_trades[-20:]
        trade_imbalance = 0.0
        if trades:
            buy_proxy = 0.0
            sell_proxy = 0.0
            for tr in trades:
                qty = float(tr.get("qty", 0.0))
                price = float(tr.get("price", mid))
                if price >= mid:
                    buy_proxy += qty
                else:
                    sell_proxy += qty
            denom = buy_proxy + sell_proxy + 1e-9
            trade_imbalance = (buy_proxy - sell_proxy) / denom

        mom5 = 0.0
        mom20 = 0.0
        hist = list(self._mid_history)
        if len(hist) >= 5 and hist[-5] > 0:
            mom5 = (mid / hist[-5]) - 1.0
        if len(hist) >= 20 and hist[-20] > 0:
            mom20 = (mid / hist[-20]) - 1.0

        impact_skew = float((state.impact_buy_mult / max(state.impact_sell_mult, 1e-9)) - 1.0)

        obs = np.array(
            [
                np.clip((mid / max(vwap, 1e-9)) - 1.0, -0.2, 0.2),
                np.clip(spread / max(mid, 1e-9), 0.0, 0.02),
                inv_norm,
                pnl_norm,
                np.clip(state.volatility * 100.0, 0.0, 1.0),
                np.clip(mom5, -0.1, 0.1),
                np.clip(mom20, -0.2, 0.2),
                np.clip(depth_imbalance, -1.0, 1.0),
                np.clip(trade_imbalance, -1.0, 1.0),
                np.clip(state.crowding_intensity, -1.0, 1.0),
                np.clip(state.crowding_side_pressure, -1.0, 1.0),
                np.clip(impact_skew, -2.0, 2.0),
                np.clip(state.impact_buy_mult - 1.0, 0.0, 4.0),
                np.clip(state.impact_sell_mult - 1.0, 0.0, 4.0),
            ],
            dtype=np.float32,
        )
        return obs

    def _ingest_transition_if_ready(self, obs: np.ndarray, state: MarketState) -> None:
        if self._prev_obs is None or self._prev_action is None or self._prev_context is None:
            return

        reward = self._compute_reward(state, self._prev_context)
        transition = _Transition(
            obs=self._prev_obs.astype(np.float32),
            action=int(self._prev_action),
            reward=float(reward),
            next_obs=obs.astype(np.float32),
        )
        self._transitions.append(transition)
        self._reward_history.append(float(reward))
        self._last_reward = float(reward)

    def _compute_reward(self, state: MarketState, prev_ctx: Dict[str, float]) -> float:
        pnl_delta = float(state.pnl - prev_ctx["pnl"])
        prev_inv_norm = float(np.clip(prev_ctx["inventory"] / self.max_inv, -1.0, 1.0))
        risk_penalty = self.risk_aversion * (prev_inv_norm ** 2) * max(0.5, prev_ctx["mid"] * 0.01)

        spread = max(prev_ctx["spread"], prev_ctx["mid"] * 1e-4)
        qty = prev_ctx["qty"]
        aggr = prev_ctx["aggressiveness"]
        side_mult = 1.0
        if prev_ctx["is_buy"] > 0.5:
            side_mult = prev_ctx["impact_buy_mult"]
        elif prev_ctx["is_sell"] > 0.5:
            side_mult = prev_ctx["impact_sell_mult"]

        impact_cost = self.impact_penalty * qty * spread * (0.2 + aggr) * side_mult

        reward_raw = pnl_delta - risk_penalty - impact_cost
        reward = float(np.clip(reward_raw / self.reward_scale, -5.0, 5.0))
        self._last_impact_cost = float(impact_cost)
        self._last_risk_penalty = float(risk_penalty)
        return reward

    def _select_action(self, obs: np.ndarray) -> int:
        if self._model is not None and np.random.random() > self.exploration_prob:
            action, _ = self._model.predict(obs, deterministic=False)
            self._update_action_probs(obs)
            return int(action)

        # Exploration / fallback behavior.
        probs = np.array([0.36, 0.12, 0.10, 0.08, 0.12, 0.10, 0.12], dtype=float)
        probs /= probs.sum()
        self._last_probs = probs
        return int(np.random.choice(self._n_actions, p=probs))

    def _update_action_probs(self, obs: np.ndarray) -> None:
        if self._model is None:
            return
        try:
            import torch

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                dist = self._model.policy.get_distribution(obs_t)
                probs = dist.distribution.probs.detach().cpu().numpy()[0]
                if probs.shape[0] == self._n_actions:
                    self._last_probs = probs.astype(float)
        except Exception:
            # Keep last distribution unchanged if policy introspection fails.
            pass

    def _action_to_orders(self, action: int, state: MarketState) -> List[Order]:
        spec = ACTION_SET[int(np.clip(action, 0, self._n_actions - 1))]
        if spec.side is None:
            return []

        if spec.side == Side.BID and self.inventory >= self.max_inv:
            return []
        if spec.side == Side.ASK and self.inventory <= -self.max_inv:
            return []

        mid = state.mid_or_last
        spread = max(float(state.spread or 0.0), mid * 1e-4)

        if spec.aggressiveness >= 0.9:
            return [
                Order(
                    agent_id=self.agent_id,
                    side=spec.side,
                    order_type=OrderType.MARKET,
                    price=mid,
                    qty=spec.qty,
                )
            ]

        best_bid = float(state.best_bid) if state.best_bid is not None else (mid - spread / 2.0)
        best_ask = float(state.best_ask) if state.best_ask is not None else (mid + spread / 2.0)

        if spec.side == Side.BID:
            price = best_bid + (spec.aggressiveness * spread)
            if state.best_ask is not None:
                price = min(price, best_ask - 1e-4)
        else:
            price = best_ask - (spec.aggressiveness * spread)
            if state.best_bid is not None:
                price = max(price, best_bid + 1e-4)

        price = round(max(0.0001, float(price)), 4)
        return [
            Order(
                agent_id=self.agent_id,
                side=spec.side,
                order_type=OrderType.LIMIT,
                price=price,
                qty=spec.qty,
            )
        ]

    def _init_model(self) -> None:
        if not SB3_AVAILABLE or not self.enable_online_learning:
            return

        self._train_env = DummyVecEnv([lambda: _TransitionReplayEnv(self._obs_dim, self._n_actions)])
        self._model = PPO(
            "MlpPolicy",
            self._train_env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            seed=self.seed,
            device="cpu",
        )

        if os.path.exists(self.model_path):
            try:
                self._model = PPO.load(self.model_path, env=self._train_env, device="cpu")
            except Exception:
                # Keep a fresh model if loading fails.
                pass

    def _maybe_online_train(self) -> None:
        if self._model is None or self._train_env is None:
            return
        if not self.enable_online_learning:
            return
        if len(self._transitions) < self.warmup_ticks:
            return
        if self._tick_count % self.train_every != 0:
            return

        try:
            replay_env = self._train_env.envs[0]
            replay_env.set_transitions(list(self._transitions))
            self._model.learn(
                total_timesteps=self.learn_steps,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            if self.save_every > 0 and self._tick_count % self.save_every == 0:
                self._persist_model()
        except Exception:
            # If online updates fail, continue simulation with current policy.
            pass

    def _persist_model(self) -> None:
        if self._model is None or not self.model_path:
            return
        directory = os.path.dirname(self.model_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            self._model.save(self.model_path)
        except Exception:
            pass

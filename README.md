# CrowdAlpha

CrowdAlpha is a multi-agent limit order book simulation platform with:
- a research-oriented market engine (price emerges from agent interaction),
- live API and WebSocket streaming,
- a React dashboard,
- crowding and fragility analytics,
- optional online RL agents,
- sandboxed user strategy deployment.

## Current Capabilities

### Engine
- Discrete-time LOB simulation with matching and execution.
- Built-in agent types:
  - `momentum`
  - `mean_reversion`
  - `market_maker`
  - `rl_ppo` (optional dependencies)

### Crowding and Analytics
- 6D behavioral factor space per agent:
  - turnover rate
  - average holding period
  - directional bias
  - volatility exposure
  - inventory skew
  - order aggressiveness
- Pairwise cosine similarity matrix.
- Per-agent crowding intensity (`Phi_i`) using volume-share weights.
- Crowding-driven impact amplification in execution path.
- Alpha decay fit and half-life estimation from Sharpe vs cumulative crowding exposure.
- Regime detection and liquidity fragility metrics.

### Platform Layer
- FastAPI REST endpoints for market state, analytics, and strategies.
- WebSocket event stream for all major simulation events.
- Frontend dashboard (React + TypeScript) with live updates.
- Sandboxed user strategy registration and deployment using subprocess workers + timeout.

## Repository Structure

```text
api/        FastAPI app, routers, websocket manager
engine/     simulation core, agents, analytics, crowding, execution
frontend/   React dashboard
tests/      phase-based test suite
db/         DB scaffolding for persistence layer
```

## Quickstart

## 1) Backend setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Run API + simulation loop:

```bash
uvicorn api.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/status
```

## 2) Frontend setup

```bash
cd frontend
npm install
npm run dev
```

By default frontend expects:
- API: `http://localhost:8000`
- WS: `ws://localhost:8000/ws/market`

Override with:
- `VITE_API_URL`
- `VITE_WS_URL`

## 3) Run tests

```bash
python -m pytest -q
```

## API Overview

Base URL: `http://localhost:8000`

### Health
- `GET /`
- `GET /status`

### Market
- `GET /market/book`
- `GET /market/trades`
- `GET /market/metrics`
- `GET /market/snapshot`

### Analytics
- `GET /analytics/crowding`
- `GET /analytics/factor-space`
- `GET /analytics/decay`
- `GET /analytics/regime`
- `GET /analytics/fragility`
- `GET /analytics/snapshot`

### Strategies
- `GET /strategies`
- `POST /strategies`
- `POST /strategies/user`
- `GET /strategies/{agent_id}/stats`
- `GET /strategies/leaderboard`

### WebSocket
- `GET /ws/schema` (event contract)
- `WS /ws/market` (live stream)

## User Strategy Deployment (Sandboxed)

Register only:

```json
POST /strategies/user
{
  "strategy_name": "my_strategy",
  "code": "...python code...",
  "deploy": false
}
```

Register and deploy immediately:

```json
POST /strategies/user
{
  "strategy_name": "my_strategy",
  "code": "...python code...",
  "deploy": true,
  "agent_id": "user_alpha_1",
  "config": {
    "timeout_ms": 120
  }
}
```

Strategy requirements:
- define exactly one class inheriting `BaseAgent`
- implement `on_tick(state)` and `factor_vector()`
- return valid `Order` objects through the interface contract

Safety model:
- AST safety checks before registration
- strategy execution in subprocess worker
- per-request timeout and process teardown on failure

## Crowding Model (Implemented)

- Agent factor vector: `f_i(t)`
- Similarity matrix:
  - `C_ij(t) = cosine(f_i, f_j)`
- Per-agent crowding intensity:
  - `Phi_i(t) = (1/(N-1)) * sum_{j!=i} C_ij(t) * w_j`
  - `w_j` = recent trading volume share of agent `j`
- Effective impact multiplier:
  - `lambda_i_eff = lambda_0 * (1 + kappa * Phi_i)`
- Alpha decay fit:
  - exponential fit on Sharpe vs cumulative crowding exposure

## RL Agent (Optional)

`rl_ppo` works without extra packages (fallback policy), and uses online PPO when installed.

Optional dependencies in `requirements.txt`:
- `stable-baselines3`
- `gymnasium`
- `torch`

## Notes

- The simulation is in-memory and research-focused by default.
- Production hardening still needed for:
  - auth/rate limits,
  - persistence and migrations,
  - stronger isolation for untrusted code,
  - deployment config and monitoring.

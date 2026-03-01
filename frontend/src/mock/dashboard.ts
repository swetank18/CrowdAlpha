import type {
  AgentStat,
  CrowdingData,
  DecayData,
  FactorSpaceData,
  MarketPoint,
  RegimeTransition,
} from "../store/simulation";

export const MOCK_TIMELINE: MarketPoint[] = Array.from({ length: 120 }, (_, i) => {
  const tick = i + 1;
  const wave = Math.sin(i / 8) * 0.9 + Math.cos(i / 17) * 0.4;
  const mid = 100 + wave;
  const crowding = 0.2 + (Math.sin(i / 13) + 1) * 0.35;
  return {
    tick,
    mid,
    vwap: 99.8 + Math.sin(i / 20) * 0.6,
    crowding: Number(crowding.toFixed(4)),
    volume: Math.round(20 + Math.abs(Math.sin(i / 6)) * 65),
    regime: crowding > 0.8 ? "UNSTABLE" : crowding > 0.65 ? "TRENDING" : "CALM",
  };
});

export const MOCK_REGIME_TRANSITIONS: RegimeTransition[] = [
  { tick: 1, prev: null, next: "CALM" },
  { tick: 42, prev: "CALM", next: "TRENDING" },
  { tick: 78, prev: "TRENDING", next: "UNSTABLE" },
];

export const MOCK_AGENTS: AgentStat[] = [
  {
    agent_id: "mom_1",
    strategy_type: "momentum",
    pnl: 1832.2,
    sharpe: 1.41,
    sortino: 1.98,
    max_drawdown: 452.1,
    turnover: 0.44,
    spread_cost: 302.1,
    market_impact_cost: 210.4,
    kurtosis: 1.4,
    vol_autocorr: 0.22,
    hit_rate: 0.59,
    avg_fill_size: 4.8,
    fill_count: 142,
    inventory: 6,
  },
  {
    agent_id: "mom_2",
    strategy_type: "momentum",
    pnl: 1194.8,
    sharpe: 1.11,
    sortino: 1.64,
    max_drawdown: 503.4,
    turnover: 0.38,
    spread_cost: 286.9,
    market_impact_cost: 226.1,
    kurtosis: 1.8,
    vol_autocorr: 0.25,
    hit_rate: 0.56,
    avg_fill_size: 5.2,
    fill_count: 128,
    inventory: 11,
  },
  {
    agent_id: "rev_1",
    strategy_type: "mean_reversion",
    pnl: 870.3,
    sharpe: 0.92,
    sortino: 1.21,
    max_drawdown: 388.8,
    turnover: 0.29,
    spread_cost: 210.8,
    market_impact_cost: 162.5,
    kurtosis: 0.9,
    vol_autocorr: 0.14,
    hit_rate: 0.54,
    avg_fill_size: 3.9,
    fill_count: 109,
    inventory: -4,
  },
  {
    agent_id: "mm_1",
    strategy_type: "market_maker",
    pnl: 1442.6,
    sharpe: 1.26,
    sortino: 1.44,
    max_drawdown: 615.4,
    turnover: 0.71,
    spread_cost: 121.8,
    market_impact_cost: 337.9,
    kurtosis: 2.1,
    vol_autocorr: 0.31,
    hit_rate: 0.58,
    avg_fill_size: 6.4,
    fill_count: 194,
    inventory: -12,
  },
];

export const MOCK_CROWDING: CrowdingData = {
  agent_ids: MOCK_AGENTS.map((a) => a.agent_id),
  activity_weights: [1.2, 1.1, 0.8, 1.4],
  crowding_intensity: 0.63,
  matrix: [
    [1, 0.82, 0.14, 0.34],
    [0.82, 1, 0.2, 0.38],
    [0.14, 0.2, 1, 0.21],
    [0.34, 0.38, 0.21, 1],
  ],
  top_crowded_pairs: [
    { agent_a: "mom_1", agent_b: "mom_2", similarity: 0.82, pair_activity: 1.32 },
    { agent_a: "mm_1", agent_b: "mom_2", similarity: 0.38, pair_activity: 1.54 },
    { agent_a: "mm_1", agent_b: "mom_1", similarity: 0.34, pair_activity: 1.68 },
  ],
};

export const MOCK_FACTOR_SPACE: FactorSpaceData = {
  factor_names: [
    "turnover_rate",
    "avg_holding_period",
    "directional_bias",
    "volatility_exposure",
    "inventory_skew",
    "order_aggressiveness",
  ],
  agents: [
    { agent_id: "mom_1", pca: { x: 1.2, y: 0.3 }, activity: 0.91, factors: { directional_bias: 0.8 } },
    { agent_id: "mom_2", pca: { x: 1.0, y: 0.6 }, activity: 0.88, factors: { directional_bias: 0.7 } },
    { agent_id: "rev_1", pca: { x: -0.4, y: 0.2 }, activity: 0.62, factors: { directional_bias: -0.3 } },
    { agent_id: "mm_1", pca: { x: -1.1, y: -0.5 }, activity: 1.0, factors: { directional_bias: 0.1 } },
  ],
};

export const MOCK_DECAY: DecayData = {
  crowding_intensity_history: MOCK_TIMELINE.slice(-80).map((p) => p.crowding),
  side_pressure_history: MOCK_TIMELINE.slice(-80).map((_, i) => Math.sin(i / 11) * 0.6),
  impact_multipliers: { buy: 1.62, sell: 1.33, side_pressure: 0.29 },
  agent_decay_params: [
    {
      agent_id: "mom_1",
      alpha_max: 1.82,
      lambda: 1.34,
      half_life: 0.52,
      r_squared: 0.87,
      n_samples: 130,
      curve: Array.from({ length: 25 }, (_, i) => {
        const x = i / 24;
        return { crowding: Number(x.toFixed(4)), predicted_sharpe: Number((1.82 * Math.exp(-1.34 * x)).toFixed(4)) };
      }),
    },
    {
      agent_id: "rev_1",
      alpha_max: 1.26,
      lambda: 0.74,
      half_life: 0.94,
      r_squared: 0.79,
      n_samples: 126,
      curve: Array.from({ length: 25 }, (_, i) => {
        const x = i / 24;
        return { crowding: Number(x.toFixed(4)), predicted_sharpe: Number((1.26 * Math.exp(-0.74 * x)).toFixed(4)) };
      }),
    },
    {
      agent_id: "mm_1",
      alpha_max: 1.41,
      lambda: 0.98,
      half_life: 0.71,
      r_squared: 0.75,
      n_samples: 124,
      curve: Array.from({ length: 25 }, (_, i) => {
        const x = i / 24;
        return { crowding: Number(x.toFixed(4)), predicted_sharpe: Number((1.41 * Math.exp(-0.98 * x)).toFixed(4)) };
      }),
    },
  ],
};


import { create } from "zustand";

export type RegimeName = "CALM" | "TRENDING" | "UNSTABLE" | "CRASH" | "CRASH_PRONE";
export type LfiAlert = "NORMAL" | "WARNING" | "DANGER" | "CRITICAL";

export interface DepthLevel {
  price: number;
  qty: number;
}

export interface FillEvent {
  tick?: number;
  price: number;
  qty: number;
  buy_agent: string;
  sell_agent: string;
  timestamp: number;
}

export interface AgentStat {
  agent_id: string;
  strategy_type: string;
  pnl: number;
  sharpe: number | null;
  sortino: number | null;
  max_drawdown: number;
  turnover: number;
  spread_cost: number;
  market_impact_cost: number;
  kurtosis: number;
  vol_autocorr: number;
  hit_rate: number;
  avg_fill_size: number;
  fill_count: number;
  inventory: number;
  cash?: number;
}

export interface MarketPoint {
  tick: number;
  mid: number | null;
  vwap: number;
  crowding: number;
  volume: number;
  regime: RegimeName | null;
}

export interface RegimeTransition {
  tick: number;
  prev: RegimeName | null;
  next: RegimeName;
}

export interface CrowdingPair {
  agent_a: string;
  agent_b: string;
  similarity: number;
  pair_activity?: number;
}

export interface CrowdingData {
  agent_ids: string[];
  matrix: number[][];
  activity_weights: number[];
  crowding_intensity: number;
  top_crowded_pairs: CrowdingPair[];
}

export interface FactorPoint {
  agent_id: string;
  factors: Record<string, number>;
  pca?: { x: number; y: number };
  activity?: number;
}

export interface FactorSpaceData {
  factor_names: string[];
  agents: FactorPoint[];
}

export interface DecayCurvePoint {
  crowding: number;
  predicted_sharpe: number;
}

export interface DecayParam {
  agent_id: string;
  alpha_max: number;
  lambda: number;
  half_life: number | null;
  r_squared: number;
  n_samples?: number;
  curve: DecayCurvePoint[];
}

export interface DecayData {
  crowding_intensity_history: number[];
  side_pressure_history?: number[];
  impact_multipliers?: {
    buy: number;
    sell: number;
    side_pressure: number;
  };
  agent_decay_params: DecayParam[];
}

export interface SelectedPair {
  a: string;
  b: string;
}

export interface SimState {
  tick: number;
  mid_price: number | null;
  spread: number | null;
  vwap: number;
  volatility: number;
  regime: RegimeName | null;
  lfi: number;
  lfi_alert: LfiAlert;
  crowding: number;
  bids: DepthLevel[];
  asks: DepthLevel[];
  recent_fills: FillEvent[];
  timeline: MarketPoint[];
  regime_transitions: RegimeTransition[];
  agent_stats: AgentStat[];
  crowding_data: CrowdingData | null;
  factor_space: FactorSpaceData | null;
  decay_data: DecayData | null;
  selected_pair: SelectedPair | null;
  connected: boolean;
  setConnected: (connected: boolean) => void;
  setSelectedPair: (pair: SelectedPair | null) => void;
  applyTick: (payload: any) => void;
  applyCrowdingEvent: (payload: any) => void;
  applyRegimeChanged: (payload: any) => void;
  applyAnalyticsSnapshot: (snapshot: any) => void;
}

const MAX_TIMELINE = 500;
const MAX_FILLS = 300;
const MAX_REGIME_TRANSITIONS = 120;

function toDepth(levels: any): DepthLevel[] {
  if (!Array.isArray(levels)) return [];
  return levels
    .map((lv: any) => ({ price: Number(lv?.[0]), qty: Number(lv?.[1]) }))
    .filter((lv) => Number.isFinite(lv.price) && Number.isFinite(lv.qty));
}

function toFills(fills: any): FillEvent[] {
  if (!Array.isArray(fills)) return [];
  return fills
    .map((f: any) => ({
      tick: typeof f?.tick === "number" ? f.tick : undefined,
      price: Number(f?.price),
      qty: Number(f?.qty),
      buy_agent: String(f?.buy_agent ?? ""),
      sell_agent: String(f?.sell_agent ?? ""),
      timestamp: Number(f?.timestamp ?? Date.now()),
    }))
    .filter((f) => Number.isFinite(f.price) && Number.isFinite(f.qty) && Number.isFinite(f.timestamp));
}

export const useSimStore = create<SimState>((set) => ({
  tick: 0,
  mid_price: null,
  spread: null,
  vwap: 0,
  volatility: 0,
  regime: null,
  lfi: 0,
  lfi_alert: "NORMAL",
  crowding: 0,
  bids: [],
  asks: [],
  recent_fills: [],
  timeline: [],
  regime_transitions: [],
  agent_stats: [],
  crowding_data: null,
  factor_space: null,
  decay_data: null,
  selected_pair: null,
  connected: false,
  setConnected: (connected) => set({ connected }),
  setSelectedPair: (pair) => set({ selected_pair: pair }),
  applyTick: (payload) =>
    set((s) => {
      const nextTick = Number(payload?.tick ?? s.tick);
      const nextRegime = (payload?.regime ?? s.regime ?? null) as RegimeName | null;
      const book = payload?.order_book ?? {};
      const fills = toFills(payload?.recent_fills ?? []);
      const volume = fills.reduce((acc, f) => acc + f.qty, 0);

      const timelinePoint: MarketPoint = {
        tick: nextTick,
        mid: payload?.mid_price ?? s.mid_price,
        vwap: Number(payload?.vwap ?? s.vwap ?? 0),
        crowding: Number(payload?.crowding ?? s.crowding ?? 0),
        volume,
        regime: nextRegime,
      };

      const transitions = [...s.regime_transitions];
      if (nextRegime && s.regime && nextRegime !== s.regime) {
        transitions.push({ tick: nextTick, prev: s.regime, next: nextRegime });
      } else if (nextRegime && !s.regime && transitions.length === 0) {
        transitions.push({ tick: nextTick, prev: null, next: nextRegime });
      }

      return {
        tick: nextTick,
        mid_price: payload?.mid_price ?? s.mid_price,
        spread: payload?.spread ?? s.spread,
        vwap: Number(payload?.vwap ?? s.vwap ?? 0),
        volatility: Number(payload?.volatility ?? s.volatility ?? 0),
        regime: nextRegime,
        lfi: Number(payload?.lfi ?? s.lfi ?? 0),
        lfi_alert: (payload?.lfi_alert ?? s.lfi_alert ?? "NORMAL") as LfiAlert,
        crowding: Number(payload?.crowding ?? s.crowding ?? 0),
        bids: toDepth(book?.bids),
        asks: toDepth(book?.asks),
        recent_fills: [...fills, ...s.recent_fills].slice(0, MAX_FILLS),
        timeline: [...s.timeline, timelinePoint].slice(-MAX_TIMELINE),
        regime_transitions: transitions.slice(-MAX_REGIME_TRANSITIONS),
        agent_stats: Array.isArray(payload?.agent_stats) ? payload.agent_stats : s.agent_stats,
      };
    }),
  applyCrowdingEvent: (payload) =>
    set((s) => ({
      crowding_data: {
        agent_ids: Array.isArray(payload?.agent_ids) ? payload.agent_ids : [],
        matrix: Array.isArray(payload?.matrix) ? payload.matrix : [],
        activity_weights: Array.isArray(payload?.activity_weights) ? payload.activity_weights : [],
        crowding_intensity: Number(payload?.crowding_intensity ?? s.crowding),
        top_crowded_pairs: Array.isArray(payload?.top_crowded_pairs) ? payload.top_crowded_pairs : [],
      },
      crowding: Number(payload?.crowding_intensity ?? s.crowding),
    })),
  applyRegimeChanged: (payload) =>
    set((s) => {
      const next = payload?.regime as RegimeName | undefined;
      if (!next) return s;
      const tick = Number(payload?.tick ?? s.tick);
      return {
        regime: next,
        regime_transitions: [...s.regime_transitions, { tick, prev: (payload?.prev_regime ?? s.regime) as RegimeName | null, next }].slice(
          -MAX_REGIME_TRANSITIONS
        ),
      };
    }),
  applyAnalyticsSnapshot: (snapshot) =>
    set((s) => ({
      agent_stats: Array.isArray(snapshot?.agent_diagnostics) ? snapshot.agent_diagnostics : s.agent_stats,
      crowding_data: snapshot?.crowding
        ? {
            agent_ids: Array.isArray(snapshot.crowding.agent_ids) ? snapshot.crowding.agent_ids : [],
            matrix: Array.isArray(snapshot.crowding.matrix) ? snapshot.crowding.matrix : [],
            activity_weights: Array.isArray(snapshot.crowding.activity_weights) ? snapshot.crowding.activity_weights : [],
            crowding_intensity: Number(snapshot.crowding.crowding_intensity ?? s.crowding),
            top_crowded_pairs: Array.isArray(snapshot.crowding.top_crowded_pairs) ? snapshot.crowding.top_crowded_pairs : [],
          }
        : s.crowding_data,
      factor_space: snapshot?.factor_space
        ? {
            factor_names: Array.isArray(snapshot.factor_space.factor_names) ? snapshot.factor_space.factor_names : [],
            agents: Array.isArray(snapshot.factor_space.agents) ? snapshot.factor_space.agents : [],
          }
        : s.factor_space,
      decay_data: snapshot?.alpha_decay
        ? {
            crowding_intensity_history: Array.isArray(snapshot.alpha_decay.crowding_intensity_history)
              ? snapshot.alpha_decay.crowding_intensity_history
              : [],
            side_pressure_history: Array.isArray(snapshot.alpha_decay.side_pressure_history)
              ? snapshot.alpha_decay.side_pressure_history
              : [],
            impact_multipliers: snapshot.alpha_decay.impact_multipliers,
            agent_decay_params: Array.isArray(snapshot.alpha_decay.agent_decay_params)
              ? snapshot.alpha_decay.agent_decay_params
              : [],
          }
        : s.decay_data,
      crowding: Number(snapshot?.market?.crowding ?? s.crowding),
      regime: (snapshot?.market?.regime ?? s.regime) as RegimeName | null,
      lfi: Number(snapshot?.market?.lfi ?? s.lfi),
      spread: Number(snapshot?.market?.spread ?? s.spread ?? 0),
      mid_price: snapshot?.market?.mid_price ?? s.mid_price,
      volatility: Number(snapshot?.market?.volatility ?? s.volatility ?? 0),
    })),
}));

export function applyTickEvent(payload: any) {
  useSimStore.getState().applyTick(payload);
}

export function applyCrowdingMatrixEvent(payload: any) {
  useSimStore.getState().applyCrowdingEvent(payload);
}

export function applyRegimeChangedEvent(payload: any) {
  useSimStore.getState().applyRegimeChanged(payload);
}

export function applyAnalyticsSnapshot(snapshot: any) {
  useSimStore.getState().applyAnalyticsSnapshot(snapshot);
}


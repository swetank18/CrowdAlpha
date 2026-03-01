/**
 * stores/simulation.ts
 *
 * Zustand slice for live simulation state.
 * Updated on every TICK WebSocket event.
 */

import { create } from "zustand";

export interface DepthLevel {
    price: number;
    qty: number;
}

export interface FillEvent {
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
    hit_rate: number;
    fill_count: number;
    inventory: number;
}

export interface SimState {
    tick: number;
    mid_price: number | null;
    spread: number | null;
    vwap: number;
    volatility: number;
    regime: "CALM" | "TRENDING" | "UNSTABLE" | "CRASH" | "CRASH_PRONE" | null;
    lfi: number;
    lfi_alert: "NORMAL" | "WARNING" | "DANGER" | "CRITICAL";
    crowding: number;
    bids: DepthLevel[];
    asks: DepthLevel[];
    recent_fills: FillEvent[];
    price_history: number[];
    agent_stats: AgentStat[];
    connected: boolean;
}

const MAX_PRICE_HISTORY = 300;
const MAX_FILLS = 50;

export const useSimStore = create<SimState>(() => ({
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
    price_history: [],
    agent_stats: [],
    connected: false,
}));

export function applyTickEvent(payload: any) {
    useSimStore.setState((s) => {
        const newFills: FillEvent[] = [
            ...(payload.recent_fills || []).map((f: any) => ({
                price: f.price,
                qty: f.qty,
                buy_agent: f.buy_agent,
                sell_agent: f.sell_agent,
                timestamp: f.timestamp || Date.now(),
            })),
            ...s.recent_fills,
        ].slice(0, MAX_FILLS);

        const newHistory = payload.mid_price
            ? [...s.price_history, payload.mid_price].slice(-MAX_PRICE_HISTORY)
            : s.price_history;

        const book = payload.order_book || {};

        return {
            tick: payload.tick ?? s.tick,
            mid_price: payload.mid_price ?? s.mid_price,
            spread: payload.spread ?? s.spread,
            vwap: payload.vwap ?? s.vwap,
            volatility: payload.volatility ?? s.volatility,
            regime: payload.regime ?? s.regime,
            lfi: payload.lfi ?? s.lfi,
            lfi_alert: payload.lfi_alert ?? s.lfi_alert,
            crowding: payload.crowding ?? s.crowding,
            bids: (book.bids || []).map(([p, q]: [number, number]) => ({ price: p, qty: q })),
            asks: (book.asks || []).map(([p, q]: [number, number]) => ({ price: p, qty: q })),
            recent_fills: newFills,
            price_history: newHistory,
            agent_stats: payload.agent_stats ?? s.agent_stats,
        };
    });
}

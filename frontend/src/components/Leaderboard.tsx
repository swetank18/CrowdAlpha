import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { MOCK_AGENTS, MOCK_CROWDING } from "../mock/dashboard";
import type { AgentStat } from "../store/simulation";
import { useSimStore } from "../store/simulation";
import { useShallow } from "zustand/react/shallow";
import { API_BASE } from "../config";

const SHARPE_WINDOW_TICKS = 50;
const STARTING_CAPITAL = 100_000;
const NEUTRAL_BAND = STARTING_CAPITAL * 0.001;
const REORDER_INTERVAL_MS = 1200;

type RuntimeAgent = {
  agent_id: string;
  strategy_type: string;
  config?: Record<string, unknown>;
};

function fmt(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return value.toFixed(digits);
}

function toTemplate(strategyType: string): "momentum" | "mean_reversion" | "market_maker" {
  if (strategyType === "momentum") return "momentum";
  if (strategyType === "mean_reversion") return "mean_reversion";
  if (strategyType === "market_maker") return "market_maker";
  return "momentum";
}

function toFiniteNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  return null;
}

function pickNumber(config: Record<string, unknown> | undefined, keys: string[], fallback: number): number {
  if (!config) return fallback;
  for (const key of keys) {
    const v = toFiniteNumber(config[key]);
    if (v !== null) return v;
  }
  return fallback;
}

function pnlClass(pnl: number) {
  if (pnl > STARTING_CAPITAL + NEUTRAL_BAND) return "text-emerald-300";
  if (pnl < STARTING_CAPITAL - NEUTRAL_BAND) return "text-rose-300";
  return "text-slate-100";
}

function exposureByAgent(agents: AgentStat[], matrix: number[][], weights: number[]) {
  const exposures = new Map<string, number>();
  if (agents.length === 0 || matrix.length === 0) return exposures;
  for (let i = 0; i < agents.length; i += 1) {
    let num = 0;
    let den = 0;
    for (let j = 0; j < agents.length; j += 1) {
      if (i === j) continue;
      const w = Math.max(0.0001, weights[j] ?? 1);
      const sim = Math.max(0, matrix[i]?.[j] ?? 0);
      num += sim * w;
      den += w;
    }
    exposures.set(agents[i].agent_id, den > 0 ? num / den : 0);
  }
  return exposures;
}

export const Leaderboard = memo(function Leaderboard() {
  const { agentStats, crowdingData, selectedPair, tick } = useSimStore(
    useShallow((s) => ({
      agentStats: s.agent_stats,
      crowdingData: s.crowding_data,
      selectedPair: s.selected_pair,
      tick: s.tick,
    }))
  );

  const stats = agentStats.length > 0 ? agentStats : MOCK_AGENTS;
  const crowding = crowdingData?.agent_ids.length ? crowdingData : MOCK_CROWDING;
  const [runtimeAgents, setRuntimeAgents] = useState<RuntimeAgent[]>([]);

  const liveRanked = useMemo(() => {
    return [...stats].sort((a, b) => (b.sharpe ?? -Infinity) - (a.sharpe ?? -Infinity));
  }, [stats]);

  const [ranked, setRanked] = useState<AgentStat[]>(liveRanked);
  const pendingRankedRef = useRef<AgentStat[]>(liveRanked);
  const lastCommitMsRef = useRef(0);
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    pendingRankedRef.current = liveRanked;
    const now = Date.now();
    const elapsed = now - lastCommitMsRef.current;
    const forceCommit = lastCommitMsRef.current === 0 || liveRanked.length !== ranked.length;
    if (forceCommit || elapsed >= REORDER_INTERVAL_MS) {
      if (flushTimerRef.current) {
        clearTimeout(flushTimerRef.current);
        flushTimerRef.current = null;
      }
      lastCommitMsRef.current = now;
      setRanked(liveRanked);
      return;
    }

    if (flushTimerRef.current) return;
    flushTimerRef.current = setTimeout(() => {
      flushTimerRef.current = null;
      lastCommitMsRef.current = Date.now();
      setRanked(pendingRankedRef.current);
    }, REORDER_INTERVAL_MS - elapsed);
  }, [liveRanked, ranked.length]);

  useEffect(
    () => () => {
      if (flushTimerRef.current) {
        clearTimeout(flushTimerRef.current);
        flushTimerRef.current = null;
      }
    },
    []
  );

  const exposures = useMemo(() => {
    const alignedAgents = crowding.agent_ids
      .map((id) => ranked.find((agent) => agent.agent_id === id))
      .filter((agent): agent is AgentStat => Boolean(agent));
    return exposureByAgent(alignedAgents, crowding.matrix, crowding.activity_weights);
  }, [crowding, ranked]);

  const runtimeById = useMemo(() => {
    const out = new Map<string, RuntimeAgent>();
    for (const row of runtimeAgents) out.set(row.agent_id, row);
    return out;
  }, [runtimeAgents]);

  useEffect(() => {
    let cancelled = false;
    async function loadRuntime() {
      try {
        const res = await fetch(`${API_BASE}/strategies`);
        if (!res.ok) return;
        const payload = await res.json();
        if (cancelled) return;
        const rows = Array.isArray(payload?.agents) ? payload.agents : [];
        setRuntimeAgents(rows as RuntimeAgent[]);
      } catch {
        // Keep operating with in-memory diagnostics data.
      }
    }
    loadRuntime();
    return () => {
      cancelled = true;
    };
  }, []);

  const rowRefs = useRef(new Map<string, HTMLTableRowElement>());
  const prevTop = useRef(new Map<string, number>());
  useLayoutEffect(() => {
    const nextTop = new Map<string, number>();
    for (const agent of ranked) {
      const row = rowRefs.current.get(agent.agent_id);
      if (!row) continue;
      const top = row.getBoundingClientRect().top;
      nextTop.set(agent.agent_id, top);
      const oldTop = prevTop.current.get(agent.agent_id);
      if (oldTop === undefined) continue;
      const delta = oldTop - top;
      if (Math.abs(delta) < 1) continue;
      row.animate([{ transform: `translateY(${delta}px)` }, { transform: "translateY(0)" }], {
        duration: 260,
        easing: "ease-out",
      });
    }
    prevTop.current = nextTop;
  }, [ranked]);

  function onClone(agent: AgentStat) {
    const runtime = runtimeById.get(agent.agent_id);
    const strategyType = runtime?.strategy_type ?? agent.strategy_type;
    const template = toTemplate(strategyType);
    const lookback = Math.round(
      pickNumber(runtime?.config, ["lookback", "window", "slow_window", "fast_window"], 30)
    );
    const aggression = pickNumber(runtime?.config, ["aggression", "entry_threshold", "half_spread"], 0.001);
    const positionSize = Math.round(pickNumber(runtime?.config, ["order_qty", "qty", "size"], 5));

    const q = new URLSearchParams({
      track: "intermediate",
      template,
      clone: agent.agent_id,
      lookback: String(Math.max(5, lookback)),
      aggression: String(Math.max(0, aggression)),
      position_size: String(Math.max(1, positionSize)),
    });
    window.location.assign(`/deploy?${q.toString()}`);
  }

  return (
    <section className="card h-full overflow-hidden">
      <header className="mb-3 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Leaderboard</h3>
        <span className="text-[11px] text-slate-500">
          Rolling {SHARPE_WINDOW_TICKS}-tick Sharpe · Tick {tick} · {ranked.length} strategies competing
        </span>
      </header>

      <div className="overflow-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-y border-surface-3/80 text-[11px] uppercase tracking-wider text-slate-500">
              <th className="px-3 py-2 text-left">#</th>
              <th className="px-3 py-2 text-left">Agent</th>
              <th className="px-3 py-2 text-left">Type</th>
              <th className="px-3 py-2 text-right">Sharpe</th>
              <th className="px-3 py-2 text-right">PnL</th>
              <th className="px-3 py-2 text-right">Crowding Exp</th>
              <th className="px-3 py-2 text-right">Action</th>
            </tr>
          </thead>
          <tbody>
            {ranked.map((agent, idx) => {
              const exposure = exposures.get(agent.agent_id) ?? 0;
              const inPair = selectedPair?.a === agent.agent_id || selectedPair?.b === agent.agent_id;
              return (
                <tr
                  key={agent.agent_id}
                  ref={(el) => {
                    if (el) rowRefs.current.set(agent.agent_id, el);
                    else rowRefs.current.delete(agent.agent_id);
                  }}
                  className={`${inPair ? "bg-sky-900/20" : "bg-transparent"} border-b border-surface-3/50`}
                >
                  <td className="px-3 py-2 text-slate-400">{idx + 1}</td>
                  <td className="px-3 py-2 font-mono text-xs text-slate-200">{agent.agent_id}</td>
                  <td className="px-3 py-2 text-xs text-slate-300">{agent.strategy_type}</td>
                  <td className="px-3 py-2 text-right tabular-nums text-slate-200">{fmt(agent.sharpe, 3)}</td>
                  <td className={`px-3 py-2 text-right tabular-nums ${pnlClass(agent.pnl)}`}>{fmt(agent.pnl, 2)}</td>
                  <td className="px-3 py-2">
                    <div className="ml-auto flex w-28 items-center gap-2">
                      <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-3">
                        <div
                          className="h-full rounded-full bg-rose-400 transition-all duration-200"
                          style={{ width: `${Math.max(0, Math.min(100, exposure * 100))}%` }}
                        />
                      </div>
                      <span className="w-9 text-right text-xs tabular-nums text-slate-300">{fmt(exposure, 2)}</span>
                    </div>
                  </td>
                  <td className="px-3 py-2 text-right">
                    <button
                      type="button"
                      onClick={() => onClone(agent)}
                      className="rounded border border-sky-700/70 bg-sky-900/30 px-2 py-1 text-xs font-medium text-sky-200 transition hover:border-sky-400 hover:bg-sky-900/50"
                    >
                      Clone
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
});

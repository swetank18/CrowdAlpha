import { memo, useMemo, useState } from "react";
import { MOCK_AGENTS, MOCK_CROWDING } from "../mock/dashboard";
import type { AgentStat } from "../store/simulation";
import { useSimStore } from "../store/simulation";
import { useShallow } from "zustand/react/shallow";

function fmt(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return value.toFixed(digits);
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
  const { agentStats, crowdingData, selectedPair } = useSimStore(
    useShallow((s) => ({
      agentStats: s.agent_stats,
      crowdingData: s.crowding_data,
      selectedPair: s.selected_pair,
    }))
  );

  const stats = agentStats.length > 0 ? agentStats : MOCK_AGENTS;
  const crowding = crowdingData?.agent_ids.length ? crowdingData : MOCK_CROWDING;
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const ranked = useMemo(() => {
    return [...stats].sort((a, b) => (b.sharpe ?? -Infinity) - (a.sharpe ?? -Infinity));
  }, [stats]);

  const exposures = useMemo(() => {
    const alignedAgents = crowding.agent_ids
      .map((id) => ranked.find((agent) => agent.agent_id === id))
      .filter((agent): agent is AgentStat => Boolean(agent));
    return exposureByAgent(alignedAgents, crowding.matrix, crowding.activity_weights);
  }, [crowding, ranked]);

  const grouped = useMemo(() => {
    const byType = new Map<string, AgentStat[]>();
    for (const agent of ranked) {
      const key = agent.strategy_type || "unknown";
      const bucket = byType.get(key) ?? [];
      bucket.push(agent);
      byType.set(key, bucket);
    }
    return Array.from(byType.entries());
  }, [ranked]);

  let rank = 1;

  return (
    <section className="card h-full overflow-hidden">
      <header className="mb-3 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Leaderboard</h3>
        <span className="text-[11px] text-slate-500">Ranked by rolling Sharpe</span>
      </header>

      <div className="space-y-2 overflow-auto">
        {grouped.map(([strategyType, agents]) => {
          const open = collapsed[strategyType] ?? true;
          const avgSharpe =
            agents.reduce((acc, agent) => acc + (agent.sharpe ?? 0), 0) / Math.max(agents.length, 1);
          return (
            <div key={strategyType} className="rounded-lg border border-surface-3/80 bg-surface-2/40">
              <button
                type="button"
                className="flex w-full items-center justify-between px-3 py-2 text-left"
                onClick={() =>
                  setCollapsed((prev) => ({
                    ...prev,
                    [strategyType]: !open,
                  }))
                }
              >
                <span className="text-sm font-medium text-slate-200">{strategyType}</span>
                <span className="text-xs text-slate-400">
                  {agents.length} agents | avg Sharpe {fmt(avgSharpe, 3)} | {open ? "Hide" : "Show"}
                </span>
              </button>

              {open && (
                <div className="overflow-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-y border-surface-3/80 text-[11px] uppercase tracking-wider text-slate-500">
                        <th className="px-3 py-2 text-left">#</th>
                        <th className="px-3 py-2 text-left">Agent</th>
                        <th className="px-3 py-2 text-right">Sharpe</th>
                        <th className="px-3 py-2 text-right">PnL</th>
                        <th className="px-3 py-2 text-right">Crowding Exp</th>
                      </tr>
                    </thead>
                    <tbody>
                      {agents.map((agent) => {
                        const currentRank = rank;
                        rank += 1;
                        const exposure = exposures.get(agent.agent_id) ?? 0;
                        const inPair =
                          selectedPair?.a === agent.agent_id || selectedPair?.b === agent.agent_id;
                        return (
                          <tr
                            key={agent.agent_id}
                            className={`${inPair ? "bg-sky-900/20" : "bg-transparent"} border-b border-surface-3/50`}
                          >
                            <td className="px-3 py-2 text-slate-400">{currentRank}</td>
                            <td className="px-3 py-2 font-mono text-xs text-slate-200">{agent.agent_id}</td>
                            <td className="px-3 py-2 text-right tabular-nums text-slate-200">
                              {fmt(agent.sharpe, 3)}
                            </td>
                            <td
                              className={`px-3 py-2 text-right tabular-nums ${
                                agent.pnl >= 0 ? "text-emerald-300" : "text-rose-300"
                              }`}
                            >
                              {fmt(agent.pnl)}
                            </td>
                            <td className="px-3 py-2">
                              <div className="ml-auto flex w-28 items-center gap-2">
                                <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-3">
                                  <div
                                    className="h-full rounded-full bg-rose-400"
                                    style={{ width: `${Math.max(0, Math.min(100, exposure * 100))}%` }}
                                  />
                                </div>
                                <span className="w-9 text-right text-xs tabular-nums text-slate-300">
                                  {fmt(exposure, 2)}
                                </span>
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
});

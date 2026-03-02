import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { useSimStore } from "../store/simulation";

type Props = {
  agentId: string;
};

function fmt(value: number | null | undefined, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function StrategyPage({ agentId }: Props) {
  const { stats, crowding, decay } = useSimStore(
    useShallow((s) => ({
      stats: s.agent_stats,
      crowding: s.crowding_data,
      decay: s.decay_data,
    }))
  );

  const agent = useMemo(() => stats.find((x) => x.agent_id === agentId) ?? null, [stats, agentId]);
  const rank = useMemo(() => {
    if (!agent) return null;
    const ranked = [...stats].sort((a, b) => (b.sharpe ?? -Infinity) - (a.sharpe ?? -Infinity));
    return ranked.findIndex((x) => x.agent_id === agent.agent_id) + 1;
  }, [agent, stats]);

  const topCorrelated = useMemo(() => {
    if (!crowding || !agent) return [];
    const i = crowding.agent_ids.findIndex((id) => id === agent.agent_id);
    if (i < 0) return [];
    return crowding.agent_ids
      .map((id, j) => ({ agent_id: id, similarity: Number(crowding.matrix[i]?.[j] ?? 0) }))
      .filter((row) => row.agent_id !== agent.agent_id)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 3);
  }, [crowding, agent]);

  const agentHalfLife = useMemo(() => {
    if (!decay || !agent) return null;
    const row = decay.agent_decay_params.find((p) => p.agent_id === agent.agent_id);
    return row?.half_life ?? null;
  }, [decay, agent]);

  if (!agent) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-6 md:px-6 md:py-8">
        <div className="card">
          <h1 className="text-lg font-semibold text-slate-100">Strategy Not Found</h1>
          <p className="mt-2 text-sm text-slate-400">
            Agent <span className="font-mono text-slate-200">{agentId}</span> is not active in the current run.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-6 md:px-6 md:py-8">
      <header className="mb-4">
        <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Strategy</p>
        <h1 className="mt-2 text-2xl font-semibold text-slate-100 font-mono">{agent.agent_id}</h1>
        <p className="mt-2 text-sm text-slate-400">
          {agent.strategy_type} | Rank {rank ?? "—"}
        </p>
      </header>

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <div className="card">
          <p className="text-xs uppercase tracking-wider text-slate-500">PnL</p>
          <p className="stat-value">{fmt(agent.pnl, 2)}</p>
        </div>
        <div className="card">
          <p className="text-xs uppercase tracking-wider text-slate-500">Sharpe</p>
          <p className="stat-value">{fmt(agent.sharpe)}</p>
        </div>
        <div className="card">
          <p className="text-xs uppercase tracking-wider text-slate-500">Crowding Exposure</p>
          <p className="stat-value">{fmt(crowding?.agent_crowding_intensity?.[agent.agent_id] ?? null)}</p>
        </div>
        <div className="card">
          <p className="text-xs uppercase tracking-wider text-slate-500">Alpha Half-Life</p>
          <p className="stat-value">{fmt(agentHalfLife, 1)}</p>
        </div>
      </section>

      <section className="mt-4 grid gap-4 lg:grid-cols-2">
        <div className="card">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Top Correlations</h2>
          <div className="mt-3 space-y-2 text-sm text-slate-300">
            {topCorrelated.length === 0 && <p className="text-slate-500">No correlation data yet.</p>}
            {topCorrelated.map((row) => (
              <div key={row.agent_id} className="flex items-center justify-between rounded border border-surface-3 px-3 py-2">
                <span className="font-mono">{row.agent_id}</span>
                <span>{fmt(row.similarity)}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Why Is It Losing?</h2>
          <div className="mt-3 space-y-2 text-sm text-slate-300">
            {agent.sharpe !== null && agent.sharpe >= 0.2 ? (
              <p>Sharpe is currently healthy. Keep monitoring crowding and impact costs.</p>
            ) : (
              <>
                <p>
                  Current Sharpe is <strong>{fmt(agent.sharpe)}</strong>; monitor crowding exposure and impact.
                </p>
                <p>
                  Impact + spread cost:{" "}
                  <strong>{fmt(agent.market_impact_cost + agent.spread_cost, 2)}</strong>
                </p>
                <p>
                  Most correlated peers:{" "}
                  <strong>{topCorrelated.map((row) => row.agent_id).join(", ") || "N/A"}</strong>
                </p>
              </>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

import { memo, useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { MOCK_AGENTS, MOCK_DECAY } from "../mock/dashboard";
import { useSimStore } from "../store/simulation";

const STRATEGY_COLORS: Record<string, string> = {
  momentum: "#0ea5e9",
  mean_reversion: "#22c55e",
  market_maker: "#f59e0b",
  unknown: "#a78bfa",
};

function inferStrategy(agentId: string) {
  if (agentId.startsWith("mom")) return "momentum";
  if (agentId.startsWith("rev")) return "mean_reversion";
  if (agentId.startsWith("mm")) return "market_maker";
  return "unknown";
}

type AggregatedPoint = {
  crowding: number;
  [strategy: string]: number;
};

function fmt(value: number | null | undefined, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return value.toFixed(digits);
}

export const AlphaDecayChart = memo(function AlphaDecayChart() {
  const { decayData, stats } = useSimStore(
    (s) => ({
      decayData: s.decay_data,
      stats: s.agent_stats,
    })
  );

  const decay = decayData?.agent_decay_params.length ? decayData : MOCK_DECAY;
  const agents = stats.length > 0 ? stats : MOCK_AGENTS;

  const strategyByAgent = useMemo(() => {
    const map = new Map<string, string>();
    for (const agent of agents) {
      map.set(agent.agent_id, agent.strategy_type);
    }
    return map;
  }, [agents]);

  const { chartData, strategyKeys } = useMemo(() => {
    const byStrategy = new Map<string, Map<number, number[]>>();
    for (const param of decay.agent_decay_params) {
      const strategy = strategyByAgent.get(param.agent_id) ?? inferStrategy(param.agent_id);
      const bucket = byStrategy.get(strategy) ?? new Map<number, number[]>();
      for (const point of param.curve) {
        const list = bucket.get(point.crowding) ?? [];
        list.push(point.predicted_sharpe);
        bucket.set(point.crowding, list);
      }
      byStrategy.set(strategy, bucket);
    }

    const xValues = Array.from(
      new Set(
        decay.agent_decay_params.flatMap((param) => param.curve.map((point) => point.crowding))
      )
    ).sort((a, b) => a - b);

    const rows: AggregatedPoint[] = xValues.map((x) => ({ crowding: x }));
    const keys = Array.from(byStrategy.keys());
    for (const strategy of keys) {
      const stratMap = byStrategy.get(strategy);
      if (!stratMap) continue;
      for (const row of rows) {
        const list = stratMap.get(row.crowding) ?? [];
        row[strategy] = list.length
          ? list.reduce((acc, value) => acc + value, 0) / list.length
          : NaN;
      }
    }

    return { chartData: rows, strategyKeys: keys };
  }, [decay, strategyByAgent]);

  const impact = decay.impact_multipliers;

  return (
    <section className="card h-full flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Alpha Decay Curves</h3>
        <div className="flex items-center gap-2 text-[11px] text-slate-500">
          <span>Buy Impact {fmt(impact?.buy, 2)}x</span>
          <span>Sell Impact {fmt(impact?.sell, 2)}x</span>
        </div>
      </div>

      <div className="h-[280px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
            <CartesianGrid stroke="#273043" strokeDasharray="3 3" />
            <XAxis
              dataKey="crowding"
              domain={[0, 1]}
              type="number"
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value: number) => value.toFixed(2)}
            />
            <YAxis
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              tickLine={false}
              axisLine={false}
              width={54}
              tickFormatter={(value: number) => value.toFixed(2)}
            />
            <Tooltip
              contentStyle={{
                background: "#121826",
                border: "1px solid #273043",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={(value: number | string | undefined, name: string | undefined) => {
                const num = typeof value === "number" ? value : Number(value ?? 0);
                return [num.toFixed(3), name ?? "value"];
              }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {strategyKeys.map((strategy) => (
              <Line
                key={strategy}
                type="monotone"
                dataKey={strategy}
                name={strategy}
                stroke={STRATEGY_COLORS[strategy] ?? STRATEGY_COLORS.unknown}
                strokeWidth={1.8}
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid gap-2 text-xs text-slate-400 sm:grid-cols-3">
        <div className="rounded-md border border-surface-3 px-2 py-1.5">
          <p className="text-[10px] uppercase tracking-wider text-slate-500">Side Pressure</p>
          <p className="font-mono text-slate-200">{fmt(impact?.side_pressure, 3)}</p>
        </div>
        <div className="rounded-md border border-surface-3 px-2 py-1.5">
          <p className="text-[10px] uppercase tracking-wider text-slate-500">History Points</p>
          <p className="font-mono text-slate-200">{decay.crowding_intensity_history.length}</p>
        </div>
        <div className="rounded-md border border-surface-3 px-2 py-1.5">
          <p className="text-[10px] uppercase tracking-wider text-slate-500">Fitted Agents</p>
          <p className="font-mono text-slate-200">{decay.agent_decay_params.length}</p>
        </div>
      </div>
    </section>
  );
});

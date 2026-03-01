/**
 * components/AlphaDecayChart.tsx
 *
 * Sharpe vs crowding intensity scatter/line for each agent.
 * Shows: observed data points + fitted exponential decay curve.
 */

import React, { useEffect, useState } from "react";
import {
    LineChart, Line, ScatterChart, Scatter,
    XAxis, YAxis, Tooltip, ResponsiveContainer,
    Legend, CartesianGrid,
} from "recharts";

const COLORS = ["#6366f1", "#22c55e", "#f59e0b", "#f43f5e", "#8b5cf6", "#06b6d4"];

interface DecayParams {
    agent_id: string;
    alpha_max: number;
    lambda: number;
    half_life: number | null;
    r_squared: number;
    curve: { crowding: number; predicted_sharpe: number }[];
}

interface DecayData {
    agent_decay_params: DecayParams[];
    crowding_intensity_history: number[];
}

export function AlphaDecayChart() {
    const [data, setData] = useState<DecayData | null>(null);

    useEffect(() => {
        const base = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
        function fetchData() {
            fetch(`${base}/analytics/decay`)
                .then((r) => r.json())
                .then(setData)
                .catch(() => { });
        }
        fetchData();
        const t = setInterval(fetchData, 3000);
        return () => clearInterval(t);
    }, []);

    const params = data?.agent_decay_params ?? [];

    // Crowding intensity over time
    const histData = (data?.crowding_intensity_history ?? []).map((v, i) => ({ i, v }));

    if (params.length === 0) {
        return (
            <div className="card flex items-center justify-center h-48 text-slate-500 text-sm">
                Fitting decay model… (needs ~20 ticks)
            </div>
        );
    }

    return (
        <div className="card flex flex-col gap-4">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Alpha Decay — Sharpe vs Crowding Intensity
            </h3>

            {/* Decay curves */}
            <ResponsiveContainer width="100%" height={200}>
                <LineChart margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#252d3d" />
                    <XAxis
                        dataKey="crowding" type="number" domain={[0, 1]}
                        tick={{ fontSize: 10, fill: "#64748b" }}
                        label={{ value: "Crowding", position: "insideBottom", fontSize: 10, fill: "#64748b" }}
                    />
                    <YAxis
                        tick={{ fontSize: 10, fill: "#64748b" }}
                        label={{ value: "Predicted Sharpe", angle: -90, position: "insideLeft", fontSize: 10, fill: "#64748b" }}
                    />
                    <Tooltip
                        contentStyle={{ background: "#1e2535", border: "1px solid #252d3d", borderRadius: 8, fontSize: 11 }}
                        formatter={(v: number, name: string) => [v.toFixed(4), name]}
                    />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    {params.map((p, idx) => (
                        <Line
                            key={p.agent_id}
                            data={p.curve}
                            dataKey="predicted_sharpe"
                            name={`${p.agent_id.slice(0, 8)} (λ=${p.lambda.toFixed(3)})`}
                            stroke={COLORS[idx % COLORS.length]}
                            strokeWidth={1.5}
                            dot={false}
                            isAnimationActive={false}
                        />
                    ))}
                </LineChart>
            </ResponsiveContainer>

            {/* Crowding history */}
            <div>
                <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
                    Crowding Intensity Over Time
                </p>
                <ResponsiveContainer width="100%" height={80}>
                    <LineChart data={histData} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
                        <Line dataKey="v" stroke="#8b5cf6" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <YAxis domain={[-1, 1]} hide />
                        <Tooltip
                            contentStyle={{ background: "#1e2535", border: "1px solid #252d3d", borderRadius: 8, fontSize: 11 }}
                            formatter={(v: number) => [v.toFixed(3), "Intensity"]}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* Params table */}
            <div className="overflow-auto">
                <table className="w-full text-xs">
                    <thead>
                        <tr className="border-b border-surface-3">
                            <th className="text-left text-[10px] text-slate-500 pb-1 pr-3">Agent</th>
                            <th className="text-right text-[10px] text-slate-500 pb-1 pr-3">α_max</th>
                            <th className="text-right text-[10px] text-slate-500 pb-1 pr-3">λ</th>
                            <th className="text-right text-[10px] text-slate-500 pb-1 pr-3">Half-life</th>
                            <th className="text-right text-[10px] text-slate-500 pb-1">R²</th>
                        </tr>
                    </thead>
                    <tbody>
                        {params.map((p, idx) => (
                            <tr key={p.agent_id}>
                                <td className="py-1 pr-3 font-mono text-[10px]" style={{ color: COLORS[idx % COLORS.length] }}>
                                    {p.agent_id.slice(0, 8)}
                                </td>
                                <td className="py-1 pr-3 text-right tabular-nums">{p.alpha_max.toFixed(3)}</td>
                                <td className="py-1 pr-3 text-right tabular-nums">{p.lambda.toFixed(3)}</td>
                                <td className="py-1 pr-3 text-right tabular-nums">{p.half_life?.toFixed(1) ?? "∞"}</td>
                                <td className="py-1 text-right tabular-nums">{p.r_squared.toFixed(3)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

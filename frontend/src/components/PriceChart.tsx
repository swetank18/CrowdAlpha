/**
 * components/PriceChart.tsx
 *
 * Mid-price + VWAP line chart using Recharts.
 * Consumes rolling price history from Zustand.
 */

import React from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
} from "recharts";
import { useSimStore } from "../store/simulation";

export function PriceChart() {
    const { price_history, vwap, tick } = useSimStore();

    const data = price_history.map((p, i) => ({
        t: tick - price_history.length + i + 1,
        mid: p,
    }));

    const minP = Math.min(...price_history, vwap) * 0.9995;
    const maxP = Math.max(...price_history, vwap) * 1.0005;

    return (
        <div className="card h-full">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                Mid Price
            </h3>
            <ResponsiveContainer width="100%" height="85%">
                <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                    <XAxis
                        dataKey="t"
                        tick={{ fontSize: 10, fill: "#64748b" }}
                        tickLine={false}
                        axisLine={false}
                        interval="preserveStartEnd"
                    />
                    <YAxis
                        domain={[minP, maxP]}
                        tick={{ fontSize: 10, fill: "#64748b" }}
                        tickLine={false}
                        axisLine={false}
                        width={55}
                        tickFormatter={(v) => v.toFixed(2)}
                    />
                    <Tooltip
                        contentStyle={{
                            background: "#1e2535",
                            border: "1px solid #252d3d",
                            borderRadius: 8,
                            fontSize: 12,
                        }}
                        formatter={(v: number) => [v.toFixed(4), "Mid"]}
                        labelFormatter={(l) => `Tick ${l}`}
                    />
                    {vwap > 0 && (
                        <ReferenceLine
                            y={vwap}
                            stroke="#6366f1"
                            strokeDasharray="4 2"
                            strokeWidth={1}
                            label={{ value: "VWAP", fill: "#6366f1", fontSize: 10, position: "insideRight" }}
                        />
                    )}
                    <Line
                        type="monotone"
                        dataKey="mid"
                        stroke="#6366f1"
                        strokeWidth={1.5}
                        dot={false}
                        isAnimationActive={false}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

/**
 * components/CrowdingHeatmap.tsx
 *
 * N×N pairwise cosine similarity heatmap.
 * Fetches from /analytics/crowding every 2s.
 * Color: 0 = dark surface, 1 = purple (crowd color).
 */

import React, { useEffect, useState } from "react";

interface CrowdingData {
    agent_ids: string[];
    matrix: number[][];
    crowding_intensity: number;
    top_crowded_pairs: { agent_a: string; agent_b: string; similarity: number }[];
}

function lerp(t: number, low: [number, number, number], high: [number, number, number]): string {
    const r = Math.round(low[0] + t * (high[0] - low[0]));
    const g = Math.round(low[1] + t * (high[1] - low[1]));
    const b = Math.round(low[2] + t * (high[2] - low[2]));
    return `rgb(${r},${g},${b})`;
}

const COLOR_LOW: [number, number, number] = [15, 17, 23];   // surface
const COLOR_HIGH: [number, number, number] = [139, 92, 246]; // crowd purple

export function CrowdingHeatmap() {
    const [data, setData] = useState<CrowdingData | null>(null);
    const [tooltip, setTooltip] = useState<{ i: number; j: number; v: number } | null>(null);

    useEffect(() => {
        const base = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
        function fetchData() {
            fetch(`${base}/analytics/crowding`)
                .then((r) => r.json())
                .then(setData)
                .catch(() => { });
        }
        fetchData();
        const t = setInterval(fetchData, 2000);
        return () => clearInterval(t);
    }, []);

    if (!data || data.agent_ids.length === 0) {
        return (
            <div className="card flex items-center justify-center h-full text-slate-500 text-sm">
                Waiting for crowding data…
            </div>
        );
    }

    const N = data.agent_ids.length;

    return (
        <div className="card h-full flex flex-col gap-4">
            <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Crowding Heatmap
                </h3>
                <span className="pill bg-crowd/20 text-crowd">
                    Intensity {data.crowding_intensity.toFixed(3)}
                </span>
            </div>

            {/* Matrix */}
            <div className="overflow-auto">
                <div
                    className="grid gap-px"
                    style={{ gridTemplateColumns: `auto repeat(${N}, 1fr)` }}
                >
                    {/* Blank corner */}
                    <div />
                    {data.agent_ids.map((id) => (
                        <div key={id} className="text-[9px] text-slate-500 text-center truncate px-px">
                            {id.slice(0, 6)}
                        </div>
                    ))}
                    {data.matrix.map((row, i) => (
                        <React.Fragment key={i}>
                            <div className="text-[9px] text-slate-500 pr-1 flex items-center">
                                {data.agent_ids[i].slice(0, 6)}
                            </div>
                            {row.map((v, j) => (
                                <div
                                    key={j}
                                    title={`${data.agent_ids[i]} × ${data.agent_ids[j]} = ${v.toFixed(3)}`}
                                    className="aspect-square rounded-sm transition-transform hover:scale-110 cursor-default"
                                    style={{ background: lerp(Math.max(0, v), COLOR_LOW, COLOR_HIGH) }}
                                    onMouseEnter={() => setTooltip({ i, j, v })}
                                    onMouseLeave={() => setTooltip(null)}
                                />
                            ))}
                        </React.Fragment>
                    ))}
                </div>
            </div>

            {/* Top crowded pairs */}
            {data.top_crowded_pairs.length > 0 && (
                <div className="flex flex-col gap-1">
                    <p className="text-[10px] text-slate-500 uppercase tracking-wider">Most Crowded Pairs</p>
                    {data.top_crowded_pairs.slice(0, 3).map((p) => (
                        <div key={`${p.agent_a}-${p.agent_b}`} className="flex items-center justify-between text-xs">
                            <span className="text-slate-400 font-mono">{p.agent_a.slice(0, 6)} × {p.agent_b.slice(0, 6)}</span>
                            <div className="flex-1 mx-2 h-px" style={{ background: lerp(p.similarity, COLOR_LOW, COLOR_HIGH) }} />
                            <span className="text-crowd tabular-nums">{p.similarity.toFixed(3)}</span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

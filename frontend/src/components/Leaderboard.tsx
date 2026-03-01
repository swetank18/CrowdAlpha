/**
 * components/Leaderboard.tsx
 * Sortable agent performance table.
 */

import React, { useState } from "react";
import { useSimStore, AgentStat } from "../store/simulation";
import { ArrowDown, ArrowUp } from "lucide-react";

type SortKey = keyof AgentStat;

function fmt(n: number | null, decimals = 2) {
    if (n === null || n === undefined) return "—";
    return n.toFixed(decimals);
}

export function Leaderboard() {
    const agents = useSimStore((s) => s.agent_stats);
    const [sortKey, setSortKey] = useState<SortKey>("pnl");
    const [asc, setAsc] = useState(false);

    function toggleSort(key: SortKey) {
        if (sortKey === key) setAsc((a) => !a);
        else { setSortKey(key); setAsc(false); }
    }

    const sorted = [...agents].sort((a, b) => {
        const av = (a[sortKey] ?? -Infinity) as number;
        const bv = (b[sortKey] ?? -Infinity) as number;
        return asc ? av - bv : bv - av;
    });

    const cols: { label: string; key: SortKey; render: (a: AgentStat) => React.ReactNode }[] = [
        { label: "Agent", key: "agent_id", render: (a) => <span className="font-mono text-xs">{a.agent_id}</span> },
        { label: "Type", key: "strategy_type", render: (a) => <span className="pill bg-accent/20 text-accent">{a.strategy_type}</span> },
        { label: "PnL", key: "pnl", render: (a) => <span className={a.pnl >= 0 ? "text-bid" : "text-ask"}>{fmt(a.pnl)}</span> },
        { label: "Sharpe", key: "sharpe", render: (a) => fmt(a.sharpe, 3) },
        { label: "Drawdown", key: "max_drawdown", render: (a) => <span className="text-warn">{fmt(a.max_drawdown)}</span> },
        { label: "Fills", key: "fill_count", render: (a) => a.fill_count },
        { label: "Inv", key: "inventory", render: (a) => <span className={a.inventory > 0 ? "text-bid" : a.inventory < 0 ? "text-ask" : ""}>{a.inventory}</span> },
    ];

    return (
        <div className="card overflow-auto">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                Leaderboard
            </h3>
            <table className="w-full text-sm">
                <thead>
                    <tr className="border-b border-surface-3">
                        {cols.map((c) => (
                            <th
                                key={c.key}
                                onClick={() => toggleSort(c.key)}
                                className="text-left text-xs text-slate-500 pb-2 pr-4 cursor-pointer hover:text-slate-300 transition-colors select-none whitespace-nowrap"
                            >
                                <span className="inline-flex items-center gap-1">
                                    {c.label}
                                    {sortKey === c.key && (asc ? <ArrowUp size={10} /> : <ArrowDown size={10} />)}
                                </span>
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {sorted.map((agent, i) => (
                        <tr key={agent.agent_id} className={`border-b border-surface-3/50 transition-colors hover:bg-surface-2 ${i === 0 ? "bg-accent/5" : ""}`}>
                            {cols.map((c) => (
                                <td key={c.key} className="py-2 pr-4 whitespace-nowrap">
                                    {c.render(agent)}
                                </td>
                            ))}
                        </tr>
                    ))}
                    {agents.length === 0 && (
                        <tr><td colSpan={cols.length} className="py-6 text-center text-slate-500 text-sm">Waiting for simulation data…</td></tr>
                    )}
                </tbody>
            </table>
        </div>
    );
}

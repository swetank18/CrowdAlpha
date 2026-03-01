/**
 * components/OrderBook.tsx
 *
 * Animated bid/ask depth visualization using horizontal bars.
 * Updates live from the WebSocket via Zustand.
 */

import React from "react";
import { useSimStore, DepthLevel } from "../store/simulation";

const MAX_LEVELS = 8;

function DepthBar({
    level,
    side,
    maxQty,
}: {
    level: DepthLevel;
    side: "bid" | "ask";
    maxQty: number;
}) {
    const width = maxQty > 0 ? (level.qty / maxQty) * 100 : 0;
    const isBid = side === "bid";

    return (
        <div className="relative flex items-center gap-2 py-[2px] group">
            {/* Background fill bar */}
            <div
                className="absolute inset-y-0 rounded transition-all duration-200"
                style={{
                    [isBid ? "right" : "left"]: 0,
                    width: `${width}%`,
                    background: isBid
                        ? "rgba(34,197,94,0.12)"
                        : "rgba(244,63,94,0.12)",
                }}
            />
            {isBid ? (
                <>
                    <span className="relative text-xs text-slate-400 w-14 text-right tabular-nums">
                        {level.qty}
                    </span>
                    <span className="relative text-xs font-medium text-bid tabular-nums w-20 text-right">
                        {level.price.toFixed(2)}
                    </span>
                </>
            ) : (
                <>
                    <span className="relative text-xs font-medium text-ask tabular-nums w-20">
                        {level.price.toFixed(2)}
                    </span>
                    <span className="relative text-xs text-slate-400 w-14 tabular-nums">
                        {level.qty}
                    </span>
                </>
            )}
        </div>
    );
}

export function OrderBook() {
    const { bids, asks, mid_price, spread } = useSimStore();

    const maxQty = Math.max(
        ...bids.slice(0, MAX_LEVELS).map((b) => b.qty),
        ...asks.slice(0, MAX_LEVELS).map((a) => a.qty),
        1
    );

    return (
        <div className="card flex flex-col h-full">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                Order Book
            </h3>

            {/* Headers */}
            <div className="flex gap-2 mb-1 px-0">
                <span className="text-[10px] text-slate-500 w-14 text-right">Qty</span>
                <span className="text-[10px] text-slate-500 w-20 text-right">Bid</span>
            </div>

            {/* Bids */}
            <div className="flex flex-col-reverse gap-px">
                {bids.slice(0, MAX_LEVELS).map((b) => (
                    <DepthBar key={b.price} level={b} side="bid" maxQty={maxQty} />
                ))}
            </div>

            {/* Spread */}
            <div className="my-2 flex items-center gap-2">
                <div className="flex-1 h-px bg-surface-3" />
                <span className="text-xs text-slate-300 font-semibold tabular-nums">
                    {mid_price?.toFixed(2) ?? "—"}
                </span>
                <span className="text-[10px] text-slate-500">
                    spread {spread?.toFixed(3) ?? "—"}
                </span>
                <div className="flex-1 h-px bg-surface-3" />
            </div>

            {/* Asks */}
            <div className="flex gap-2 mb-1">
                <span className="text-[10px] text-slate-500 w-20">Ask</span>
                <span className="text-[10px] text-slate-500">Qty</span>
            </div>
            <div className="flex flex-col gap-px">
                {asks.slice(0, MAX_LEVELS).map((a) => (
                    <DepthBar key={a.price} level={a} side="ask" maxQty={maxQty} />
                ))}
            </div>
        </div>
    );
}

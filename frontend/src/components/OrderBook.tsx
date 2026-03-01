import { memo, useMemo } from "react";
import { useSimStore, type DepthLevel } from "../store/simulation";
import { useShallow } from "zustand/react/shallow";

const VISIBLE_LEVELS = 24;

interface DepthRowProps {
  level: DepthLevel | null;
  side: "bid" | "ask";
  maxQty: number;
}

const DepthRow = memo(function DepthRow({ level, side, maxQty }: DepthRowProps) {
  if (!level) {
    return <div className="h-5 rounded bg-surface-2/40" />;
  }

  const widthPct = maxQty > 0 ? Math.min(100, (level.qty / maxQty) * 100) : 0;
  const isBid = side === "bid";

  return (
    <div className="relative h-5 overflow-hidden rounded">
      <div
        className="absolute inset-y-0 transition-[width] duration-150"
        style={{
          [isBid ? "right" : "left"]: 0,
          width: `${widthPct}%`,
          background: isBid ? "rgba(34,197,94,0.16)" : "rgba(244,63,94,0.16)",
        }}
      />
      <div className={`relative z-10 grid h-full items-center text-[11px] tabular-nums ${isBid ? "grid-cols-[1fr_auto]" : "grid-cols-[auto_1fr]"}`}>
        {isBid ? (
          <>
            <span className="pr-2 text-right text-slate-400">{level.qty.toFixed(0)}</span>
            <span className="w-16 text-right font-medium text-bid">{level.price.toFixed(2)}</span>
          </>
        ) : (
          <>
            <span className="w-16 font-medium text-ask">{level.price.toFixed(2)}</span>
            <span className="pl-2 text-slate-400">{level.qty.toFixed(0)}</span>
          </>
        )}
      </div>
    </div>
  );
});

function padLevels(levels: DepthLevel[], count: number): Array<DepthLevel | null> {
  if (levels.length >= count) return levels.slice(0, count);
  return [...levels, ...Array.from({ length: count - levels.length }, () => null)];
}

export const OrderBook = memo(function OrderBook() {
  const { bids, asks, midPrice, spread, connected } = useSimStore(
    useShallow((s) => ({
      bids: s.bids,
      asks: s.asks,
      midPrice: s.mid_price,
      spread: s.spread,
      connected: s.connected,
    }))
  );

  const { bidRows, askRows, maxQty } = useMemo(() => {
    const bidSlice = bids.slice(0, VISIBLE_LEVELS);
    const askSlice = asks.slice(0, VISIBLE_LEVELS);
    const max = Math.max(
      1,
      ...bidSlice.map((b) => b.qty),
      ...askSlice.map((a) => a.qty)
    );
    return {
      bidRows: padLevels(bidSlice, VISIBLE_LEVELS),
      askRows: padLevels(askSlice, VISIBLE_LEVELS),
      maxQty: max,
    };
  }, [bids, asks]);

  return (
    <section className="card h-full">
      <header className="mb-3 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Order Book</h3>
        <span className={`pill ${connected ? "bg-emerald-900/40 text-emerald-300" : "bg-slate-700/50 text-slate-300"}`}>
          {connected ? "Live" : "Mock"}
        </span>
      </header>

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <div className="mb-1 grid grid-cols-[1fr_auto] text-[10px] uppercase tracking-wider text-slate-500">
            <span className="text-right pr-2">Qty</span>
            <span className="w-16 text-right">Bid</span>
          </div>
          {bidRows
            .slice()
            .reverse()
            .map((row, idx) => (
              <DepthRow key={`bid-${idx}`} level={row} side="bid" maxQty={maxQty} />
            ))}
        </div>

        <div className="space-y-1">
          <div className="mb-1 grid grid-cols-[auto_1fr] text-[10px] uppercase tracking-wider text-slate-500">
            <span className="w-16">Ask</span>
            <span className="pl-2">Qty</span>
          </div>
          {askRows.map((row, idx) => (
            <DepthRow key={`ask-${idx}`} level={row} side="ask" maxQty={maxQty} />
          ))}
        </div>
      </div>

      <div className="mt-3 flex items-center justify-center gap-3 text-xs">
        <span className="font-semibold tabular-nums text-slate-200">{midPrice?.toFixed(3) ?? "-"}</span>
        <span className="text-slate-500">spread</span>
        <span className="tabular-nums text-slate-300">{spread?.toFixed(4) ?? "-"}</span>
      </div>
    </section>
  );
});

import { memo, useMemo } from "react";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useSimStore } from "../store/simulation";
import { MOCK_REGIME_TRANSITIONS, MOCK_TIMELINE } from "../mock/dashboard";

const REGIME_COLORS: Record<string, string> = {
  CALM: "#22c55e",
  TRENDING: "#0ea5e9",
  UNSTABLE: "#f59e0b",
  CRASH: "#ef4444",
  CRASH_PRONE: "#fb7185",
};

type Point = {
  tick: number;
  mid: number;
  vwap: number;
  crowding: number;
  volume: number;
};

function clamp01(value: number) {
  return Math.min(1, Math.max(0, value));
}

export const PriceChart = memo(function PriceChart() {
  const timeline = useSimStore((s) => s.timeline);
  const transitions = useSimStore((s) => s.regime_transitions);
  const rows = timeline.length > 0 ? timeline : MOCK_TIMELINE;
  const markers = transitions.length > 0 ? transitions : MOCK_REGIME_TRANSITIONS;

  const data = useMemo<Point[]>(() => {
    return rows.reduce<Point[]>((acc, row) => {
      const fallback = acc.length > 0 ? acc[acc.length - 1].mid : 100;
      const mid = row.mid ?? fallback;
      const point: Point = {
        tick: row.tick,
        mid,
        vwap: row.vwap,
        crowding: clamp01(row.crowding),
        volume: row.volume,
      };
      return [...acc, point];
    }, []);
  }, [rows]);

  const [minPrice, maxPrice] = useMemo(() => {
    if (data.length === 0) return [99, 101];
    const prices = data.flatMap((d) => [d.mid, d.vwap]);
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const pad = Math.max(0.01, (max - min) * 0.15);
    return [min - pad, max + pad];
  }, [data]);

  const maxVol = useMemo(() => {
    if (data.length === 0) return 1;
    return Math.max(1, ...data.map((d) => d.volume));
  }, [data]);

  return (
    <section className="card h-full flex flex-col">
      <header className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Price / Volume / Crowding</h3>
        <p className="text-[11px] text-slate-500">Regime transitions marked on timeline</p>
      </header>

      <div className="h-[72%]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 2 }}>
            <CartesianGrid stroke="#273043" strokeDasharray="3 3" />
            <XAxis
              dataKey="tick"
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              tickLine={false}
              axisLine={false}
              minTickGap={28}
            />
            <YAxis
              yAxisId="price"
              domain={[minPrice, maxPrice]}
              width={58}
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => v.toFixed(2)}
            />
            <YAxis
              yAxisId="crowd"
              orientation="right"
              domain={[0, 1]}
              width={40}
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => v.toFixed(2)}
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
                return [num.toFixed(4), name ?? "value"];
              }}
              labelFormatter={(label) => `Tick ${label}`}
            />
            {markers.map((marker) => (
              <ReferenceLine
                key={`${marker.tick}-${marker.next}`}
                x={marker.tick}
                stroke={REGIME_COLORS[marker.next] ?? "#64748b"}
                strokeDasharray="2 3"
                label={{
                  value: marker.next,
                  position: "insideTopRight",
                  fill: REGIME_COLORS[marker.next] ?? "#94a3b8",
                  fontSize: 9,
                }}
              />
            ))}
            <Line
              yAxisId="price"
              dataKey="mid"
              name="Mid"
              stroke="#38bdf8"
              strokeWidth={1.8}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              yAxisId="price"
              dataKey="vwap"
              name="VWAP"
              stroke="#f59e0b"
              strokeDasharray="4 3"
              strokeWidth={1.4}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              yAxisId="crowd"
              dataKey="crowding"
              name="Crowding"
              stroke="#f43f5e"
              strokeWidth={1.6}
              dot={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="h-[28%] border-t border-surface-3 pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 2, right: 8, bottom: 0, left: 0 }}>
            <XAxis dataKey="tick" hide />
            <YAxis domain={[0, maxVol]} hide />
            <Tooltip
              contentStyle={{
                background: "#121826",
                border: "1px solid #273043",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={(value: number | string | undefined) => {
                const num = typeof value === "number" ? value : Number(value ?? 0);
                return [num.toFixed(0), "Volume"];
              }}
              labelFormatter={(label) => `Tick ${label}`}
            />
            <Bar dataKey="volume" name="Volume" fill="#334155" maxBarSize={6} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
});

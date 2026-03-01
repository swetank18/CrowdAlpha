import { memo, useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { MOCK_CROWDING } from "../mock/dashboard";
import { useSimStore } from "../store/simulation";

type HoverCell = {
  i: number;
  j: number;
  x: number;
  y: number;
};

function similarityToColor(similarity: number) {
  const t = Math.min(1, Math.max(0, similarity));
  const start = { r: 34, g: 197, b: 94 };
  const end = { r: 239, g: 68, b: 68 };
  const r = Math.round(start.r + (end.r - start.r) * t);
  const g = Math.round(start.g + (end.g - start.g) * t);
  const b = Math.round(start.b + (end.b - start.b) * t);
  return `rgb(${r},${g},${b})`;
}

export const CrowdingHeatmap = memo(function CrowdingHeatmap() {
  const { crowdingData, selectedPair, setSelectedPair } = useSimStore(
    (s) => ({
      crowdingData: s.crowding_data,
      selectedPair: s.selected_pair,
      setSelectedPair: s.setSelectedPair,
    })
  );

  const data = useMemo(() => {
    if (crowdingData && crowdingData.agent_ids.length > 0) {
      return crowdingData;
    }
    return MOCK_CROWDING;
  }, [crowdingData]);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [size, setSize] = useState(320);
  const [hover, setHover] = useState<HoverCell | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const resize = () => {
      const next = Math.max(220, Math.min(el.clientWidth, el.clientHeight || 400));
      setSize(next);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const n = data.agent_ids.length;
    if (n === 0) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(size * dpr);
    canvas.height = Math.floor(size * dpr);
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    const cell = size / n;
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        const value = data.matrix[i]?.[j] ?? 0;
        ctx.fillStyle = similarityToColor(value);
        ctx.fillRect(j * cell, i * cell, cell, cell);
      }
    }

    const selectedA = selectedPair?.a;
    const selectedB = selectedPair?.b;
    if (selectedA && selectedB) {
      const i = data.agent_ids.indexOf(selectedA);
      const j = data.agent_ids.indexOf(selectedB);
      if (i >= 0 && j >= 0) {
        ctx.strokeStyle = "#f8fafc";
        ctx.lineWidth = 2;
        ctx.strokeRect(j * cell + 1, i * cell + 1, Math.max(cell - 2, 2), Math.max(cell - 2, 2));
        ctx.strokeRect(i * cell + 1, j * cell + 1, Math.max(cell - 2, 2), Math.max(cell - 2, 2));
      }
    }
  }, [data, size, selectedPair]);

  const onPointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const n = data.agent_ids.length;
    if (n === 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const cell = size / n;
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const j = Math.floor(x / cell);
    const i = Math.floor(y / cell);
    if (i < 0 || i >= n || j < 0 || j >= n) {
      setHover(null);
      return;
    }
    setHover({ i, j, x, y });
  };

  const onPointerLeave = () => {
    setHover(null);
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const n = data.agent_ids.length;
    if (n === 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const cell = size / n;
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const j = Math.floor(x / cell);
    const i = Math.floor(y / cell);
    if (i < 0 || i >= n || j < 0 || j >= n || i === j) {
      setSelectedPair(null);
      return;
    }
    setSelectedPair({ a: data.agent_ids[i], b: data.agent_ids[j] });
  };

  const hoveredValue = hover ? data.matrix[hover.i]?.[hover.j] ?? 0 : 0;
  const hoveredLeft = hover ? Math.min(Math.max(hover.x + 10, 8), size - 150) : 0;
  const hoveredTop = hover ? Math.min(Math.max(hover.y + 10, 8), size - 60) : 0;

  return (
    <section className="card h-full flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Crowding Heatmap</h3>
        <span className="pill bg-rose-900/40 text-rose-300">Intensity {data.crowding_intensity.toFixed(3)}</span>
      </div>

      <div ref={containerRef} className="relative flex-1 min-h-0 rounded-lg border border-surface-3 bg-surface/70 p-2">
        <canvas
          ref={canvasRef}
          className="mx-auto block rounded-md"
          onPointerMove={onPointerMove}
          onPointerLeave={onPointerLeave}
          onPointerDown={onPointerDown}
        />
        {hover && (
          <div
            className="pointer-events-none absolute rounded-md border border-surface-3 bg-surface-1/95 px-2 py-1 text-[11px] text-slate-300"
            style={{ left: hoveredLeft, top: hoveredTop }}
          >
            <p className="font-mono">
              {data.agent_ids[hover.i]} × {data.agent_ids[hover.j]}
            </p>
            <p className="text-slate-400">Similarity {hoveredValue.toFixed(3)}</p>
          </div>
        )}
      </div>

      <div className="grid gap-1 text-xs text-slate-400 sm:grid-cols-2">
        <p>Click any off-diagonal cell to link agents across panels.</p>
        <p className="text-right text-slate-500">{data.agent_ids.length} × {data.agent_ids.length} matrix</p>
      </div>
    </section>
  );
});

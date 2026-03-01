import { memo, useEffect, useMemo, useRef } from "react";
import { MOCK_AGENTS, MOCK_FACTOR_SPACE } from "../mock/dashboard";
import { useSimStore } from "../store/simulation";
import { useShallow } from "zustand/react/shallow";

type PlotPoint = {
  agentId: string;
  nx: number;
  ny: number;
  activity: number;
  strategy: string;
};

const STRATEGY_COLORS: Record<string, string> = {
  momentum: "#0ea5e9",
  mean_reversion: "#22c55e",
  market_maker: "#f59e0b",
  unknown: "#94a3b8",
};

function inferStrategy(agentId: string) {
  if (agentId.startsWith("mom")) return "momentum";
  if (agentId.startsWith("rev")) return "mean_reversion";
  if (agentId.startsWith("mm")) return "market_maker";
  return "unknown";
}

function getCanvasDims(el: HTMLDivElement | null) {
  const width = Math.max(280, el?.clientWidth ?? 400);
  const height = Math.max(220, el?.clientHeight ?? 300);
  return { width, height };
}

export const FactorSpace = memo(function FactorSpace() {
  const { factorSpace, agentStats, selectedPair } = useSimStore(
    useShallow((s) => ({
      factorSpace: s.factor_space,
      agentStats: s.agent_stats,
      selectedPair: s.selected_pair,
    }))
  );

  const pointsSource = factorSpace?.agents?.length ? factorSpace.agents : MOCK_FACTOR_SPACE.agents;
  const statsSource = agentStats.length > 0 ? agentStats : MOCK_AGENTS;
  const strategyMap = useMemo(() => {
    const map = new Map<string, string>();
    for (const stat of statsSource) map.set(stat.agent_id, stat.strategy_type);
    return map;
  }, [statsSource]);

  const points = useMemo<PlotPoint[]>(() => {
    const raw = pointsSource.map((agent) => ({
      agentId: agent.agent_id,
      xRaw: agent.pca?.x ?? 0,
      yRaw: agent.pca?.y ?? 0,
      activity: agent.activity ?? 0.5,
      strategy: strategyMap.get(agent.agent_id) ?? inferStrategy(agent.agent_id),
    }));
    if (raw.length === 0) return [];

    const maxAbsX = Math.max(0.001, ...raw.map((p) => Math.abs(p.xRaw)));
    const maxAbsY = Math.max(0.001, ...raw.map((p) => Math.abs(p.yRaw)));

    return raw.map((point) => ({
      agentId: point.agentId,
      // Keep zero-centered PCA points near the center rather than edge-clipping.
      nx: Math.max(0, Math.min(1, (point.xRaw / maxAbsX + 1.0) * 0.5)),
      ny: Math.max(0, Math.min(1, (point.yRaw / maxAbsY + 1.0) * 0.5)),
      activity: point.activity,
      strategy: point.strategy,
    }));
  }, [pointsSource, strategyMap]);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animated = useRef<Map<string, { x: number; y: number }>>(new Map());
  const targets = useRef<Map<string, PlotPoint>>(new Map());
  useEffect(() => {
    const map = new Map<string, PlotPoint>();
    for (const point of points) {
      map.set(point.agentId, point);
      if (!animated.current.has(point.agentId)) {
        animated.current.set(point.agentId, { x: 0, y: 0 });
      }
    }
    targets.current = map;
  }, [points]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      const { width, height } = getCanvasDims(container);
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      ctx.strokeStyle = "#233044";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.moveTo(width / 2, 0);
      ctx.lineTo(width / 2, height);
      ctx.stroke();

      const pad = 28;
      for (const [agentId, target] of targets.current.entries()) {
        const targetX = pad + target.nx * (width - pad * 2);
        const targetY = height - (pad + target.ny * (height - pad * 2));
        const previous = animated.current.get(agentId) ?? { x: targetX, y: targetY };
        const nextX = previous.x + (targetX - previous.x) * 0.16;
        const nextY = previous.y + (targetY - previous.y) * 0.16;
        animated.current.set(agentId, { x: nextX, y: nextY });

        const isSelected = selectedPair?.a === agentId || selectedPair?.b === agentId;
        const color = STRATEGY_COLORS[target.strategy] ?? STRATEGY_COLORS.unknown;
        const radius = 4 + Math.max(0, target.activity) * 5;

        ctx.beginPath();
        ctx.arc(nextX, nextY, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        if (isSelected) {
          ctx.lineWidth = 2.2;
          ctx.strokeStyle = "#f8fafc";
          ctx.stroke();
          ctx.fillStyle = "#e2e8f0";
          ctx.font = "11px 'IBM Plex Mono', monospace";
          ctx.fillText(agentId, nextX + radius + 3, nextY - radius - 2);
        }
      }
      frame = window.requestAnimationFrame(draw);
    };

    let frame = window.requestAnimationFrame(draw);
    const onResize = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(draw);
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", onResize);
    };
  }, [selectedPair]);

  return (
    <section className="card h-full flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Factor Space (PCA)</h3>
        <span className="text-[11px] text-slate-500">{points.length} active agents</span>
      </div>
      <div ref={containerRef} className="flex-1 min-h-0 rounded-lg border border-surface-3 bg-surface/70 p-2">
        <canvas ref={canvasRef} className="block h-full w-full rounded-md" />
      </div>
      <div className="flex flex-wrap gap-2 text-xs">
        {Object.entries(STRATEGY_COLORS).map(([strategy, color]) => (
          <div key={strategy} className="inline-flex items-center gap-1.5 text-slate-400">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
            <span>{strategy}</span>
          </div>
        ))}
      </div>
    </section>
  );
});

import { memo, useEffect, useMemo, useRef, useState } from "react";
import { MOCK_AGENTS, MOCK_FACTOR_SPACE } from "../mock/dashboard";
import { useSimStore } from "../store/simulation";

type PlotPoint = {
  agentId: string;
  x: number;
  y: number;
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

export const FactorSpace = memo(function FactorSpace() {
  const { factorSpace, agentStats, selectedPair } = useSimStore(
    (s) => ({
      factorSpace: s.factor_space,
      agentStats: s.agent_stats,
      selectedPair: s.selected_pair,
    })
  );

  const pointsSource = factorSpace?.agents?.length ? factorSpace.agents : MOCK_FACTOR_SPACE.agents;
  const statsSource = agentStats.length > 0 ? agentStats : MOCK_AGENTS;

  const strategyMap = useMemo(() => {
    const map = new Map<string, string>();
    for (const stat of statsSource) {
      map.set(stat.agent_id, stat.strategy_type);
    }
    return map;
  }, [statsSource]);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [size, setSize] = useState({ width: 400, height: 300 });
  const animated = useRef<Map<string, { x: number; y: number }>>(new Map());
  const targets = useRef<Map<string, PlotPoint>>(new Map());

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const resize = () => {
      setSize({
        width: Math.max(280, el.clientWidth),
        height: Math.max(220, el.clientHeight),
      });
    };
    resize();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", resize);
      return () => window.removeEventListener("resize", resize);
    }
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const normalizedPoints = useMemo(() => {
    const raw = pointsSource.map((agent) => {
      const x = agent.pca?.x ?? 0;
      const y = agent.pca?.y ?? 0;
      const strategy = strategyMap.get(agent.agent_id) ?? inferStrategy(agent.agent_id);
      return {
        agentId: agent.agent_id,
        xRaw: x,
        yRaw: y,
        activity: agent.activity ?? 0.5,
        strategy,
      };
    });

    if (raw.length === 0) return [] as PlotPoint[];

    const minX = Math.min(...raw.map((p) => p.xRaw));
    const maxX = Math.max(...raw.map((p) => p.xRaw));
    const minY = Math.min(...raw.map((p) => p.yRaw));
    const maxY = Math.max(...raw.map((p) => p.yRaw));
    const dx = Math.max(maxX - minX, 0.001);
    const dy = Math.max(maxY - minY, 0.001);

    const pad = 28;
    return raw.map((point) => ({
      agentId: point.agentId,
      x: pad + ((point.xRaw - minX) / dx) * (size.width - pad * 2),
      y: size.height - (pad + ((point.yRaw - minY) / dy) * (size.height - pad * 2)),
      activity: point.activity,
      strategy: point.strategy,
    }));
  }, [pointsSource, size, strategyMap]);

  useEffect(() => {
    const map = new Map<string, PlotPoint>();
    for (const point of normalizedPoints) {
      map.set(point.agentId, point);
      if (!animated.current.has(point.agentId)) {
        animated.current.set(point.agentId, { x: point.x, y: point.y });
      }
    }
    targets.current = map;
  }, [normalizedPoints]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(size.width * dpr);
    canvas.height = Math.floor(size.height * dpr);
    canvas.style.width = `${size.width}px`;
    canvas.style.height = `${size.height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    let frame = 0;
    const draw = () => {
      ctx.clearRect(0, 0, size.width, size.height);

      ctx.strokeStyle = "#233044";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, size.height / 2);
      ctx.lineTo(size.width, size.height / 2);
      ctx.moveTo(size.width / 2, 0);
      ctx.lineTo(size.width / 2, size.height);
      ctx.stroke();

      for (const [agentId, target] of targets.current.entries()) {
        const previous = animated.current.get(agentId) ?? { x: target.x, y: target.y };
        const nextX = previous.x + (target.x - previous.x) * 0.16;
        const nextY = previous.y + (target.y - previous.y) * 0.16;
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

    frame = window.requestAnimationFrame(draw);
    return () => window.cancelAnimationFrame(frame);
  }, [selectedPair, size]);

  return (
    <section className="card h-full flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Factor Space (PCA)</h3>
        <span className="text-[11px] text-slate-500">{normalizedPoints.length} active agents</span>
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

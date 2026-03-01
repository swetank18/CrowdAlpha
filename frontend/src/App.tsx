import { useEffect } from "react";
import "./App.css";
import { useAnalyticsSnapshot } from "./hooks/useAnalyticsSnapshot";
import { useMarketWS } from "./hooks/useMarketWS";
import { AlphaDecayChart } from "./components/AlphaDecayChart";
import { CrowdingHeatmap } from "./components/CrowdingHeatmap";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { FactorSpace } from "./components/FactorSpace";
import { Leaderboard } from "./components/Leaderboard";
import { OrderBook } from "./components/OrderBook";
import { PriceChart } from "./components/PriceChart";
import { useSimStore } from "./store/simulation";
import {
  MOCK_AGENTS,
  MOCK_CROWDING,
  MOCK_DECAY,
  MOCK_FACTOR_SPACE,
  MOCK_REGIME_TRANSITIONS,
  MOCK_TIMELINE,
} from "./mock/dashboard";

function buildMockBook(midPrice: number) {
  const step = 0.02;
  const levels = 24;
  const bids = Array.from({ length: levels }, (_, i) => {
    const depthFactor = Math.exp(-i / 9);
    const seasonal = Math.abs(Math.sin(i * 0.7));
    return {
      price: Number((midPrice - (i + 1) * step).toFixed(2)),
      qty: Math.round(140 * depthFactor + 25 * seasonal + 4),
    };
  });
  const asks = Array.from({ length: levels }, (_, i) => {
    const depthFactor = Math.exp(-i / 9);
    const seasonal = Math.abs(Math.cos(i * 0.65));
    return {
      price: Number((midPrice + (i + 1) * step).toFixed(2)),
      qty: Math.round(130 * depthFactor + 22 * seasonal + 4),
    };
  });
  return { bids, asks };
}

function App() {
  useMarketWS();
  useAnalyticsSnapshot(2500);

  const { connected, tick, midPrice, spread, regime, lfi, crowding, timelineLen, selectedPair } =
    useSimStore((s) => ({
      connected: s.connected,
      tick: s.tick,
      midPrice: s.mid_price,
      spread: s.spread,
      regime: s.regime,
      lfi: s.lfi,
      crowding: s.crowding,
      timelineLen: s.timeline.length,
      selectedPair: s.selected_pair,
    }));

  useEffect(() => {
    if (connected || timelineLen > 0) return;
    const timer = setTimeout(() => {
      const state = useSimStore.getState();
      if (state.connected || state.timeline.length > 0) return;
      const last = MOCK_TIMELINE[MOCK_TIMELINE.length - 1];
      const mid = last.mid ?? 100;
      const { bids, asks } = buildMockBook(mid);
      useSimStore.setState({
        tick: last.tick,
        mid_price: mid,
        spread: 0.04,
        vwap: last.vwap,
        volatility: 0.0012,
        regime: last.regime ?? "CALM",
        lfi: 0.39,
        lfi_alert: "WARNING",
        crowding: MOCK_CROWDING.crowding_intensity,
        bids,
        asks,
        timeline: MOCK_TIMELINE,
        regime_transitions: MOCK_REGIME_TRANSITIONS,
        agent_stats: MOCK_AGENTS,
        crowding_data: MOCK_CROWDING,
        factor_space: MOCK_FACTOR_SPACE,
        decay_data: MOCK_DECAY,
      });
    }, 1000);
    return () => clearTimeout(timer);
  }, [connected, timelineLen]);

  const regimeClass = `regime-${regime ?? "CALM"}`;
  const hasSelection = Boolean(selectedPair);

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,#1f2937_0%,#0f1117_55%,#0a0c12_100%)] text-slate-100">
      <div className="mx-auto max-w-[1500px] px-4 py-5 md:px-6 md:py-7">
        <header className="mb-4 flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-[11px] uppercase tracking-[0.22em] text-slate-500">CrowdAlpha</p>
            <h1 className="text-2xl font-semibold tracking-tight text-slate-100">Market Simulation Dashboard</h1>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`pill ${
                connected ? "bg-emerald-900/40 text-emerald-300" : "bg-amber-900/40 text-amber-300"
              }`}
            >
              {connected ? "WebSocket Live" : "Mock Fallback"}
            </span>
            <span className={`pill ${regimeClass}`}>{regime ?? "CALM"}</span>
            {hasSelection && (
              <span className="pill bg-sky-900/40 text-sky-300">
                Pair {selectedPair?.a} × {selectedPair?.b}
              </span>
            )}
          </div>
        </header>

        <section className="mb-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <div className="card">
            <p className="text-[11px] uppercase tracking-wider text-slate-500">Tick</p>
            <p className="stat-value">{tick}</p>
          </div>
          <div className="card">
            <p className="text-[11px] uppercase tracking-wider text-slate-500">Mid Price</p>
            <p className="stat-value">{midPrice?.toFixed(3) ?? "-"}</p>
          </div>
          <div className="card">
            <p className="text-[11px] uppercase tracking-wider text-slate-500">Spread</p>
            <p className="stat-value">{spread?.toFixed(4) ?? "-"}</p>
          </div>
          <div className="card">
            <p className="text-[11px] uppercase tracking-wider text-slate-500">LFI</p>
            <p className="stat-value">{lfi.toFixed(3)}</p>
          </div>
          <div className="card">
            <p className="text-[11px] uppercase tracking-wider text-slate-500">Crowding</p>
            <p className="stat-value">{crowding.toFixed(3)}</p>
          </div>
        </section>

        <section className="grid grid-cols-1 gap-4 xl:grid-cols-12">
          <div className="xl:col-span-8 h-[350px]">
            <ErrorBoundary compact name="PriceChart">
              <PriceChart />
            </ErrorBoundary>
          </div>
          <div className="xl:col-span-4 h-[350px]">
            <ErrorBoundary compact name="OrderBook">
              <OrderBook />
            </ErrorBoundary>
          </div>
          <div className="xl:col-span-6 h-[420px]">
            <ErrorBoundary compact name="CrowdingHeatmap">
              <CrowdingHeatmap />
            </ErrorBoundary>
          </div>
          <div className="xl:col-span-6 h-[420px]">
            <ErrorBoundary compact name="FactorSpace">
              <FactorSpace />
            </ErrorBoundary>
          </div>
          <div className="xl:col-span-7">
            <ErrorBoundary compact name="Leaderboard">
              <Leaderboard />
            </ErrorBoundary>
          </div>
          <div className="xl:col-span-5">
            <ErrorBoundary compact name="AlphaDecayChart">
              <AlphaDecayChart />
            </ErrorBoundary>
          </div>
        </section>
      </div>
    </main>
  );
}

export default App;

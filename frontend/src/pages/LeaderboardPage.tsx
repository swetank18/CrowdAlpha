import { ErrorBoundary } from "../components/ErrorBoundary";
import { Leaderboard } from "../components/Leaderboard";

export function LeaderboardPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-6 md:px-6 md:py-8">
      <header className="mb-4">
        <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Leaderboard</p>
        <h1 className="mt-2 text-2xl font-semibold text-slate-100">Strategy Rankings</h1>
        <p className="mt-2 text-sm text-slate-400">Live ranking by rolling Sharpe with crowding exposure overlays.</p>
      </header>
      <ErrorBoundary compact name="Leaderboard">
        <Leaderboard />
      </ErrorBoundary>
    </div>
  );
}

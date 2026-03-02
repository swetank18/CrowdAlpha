type Props = {
  go: (path: string) => void;
};

export function LandingPage({ go }: Props) {
  return (
    <div className="mx-auto flex min-h-[calc(100vh-120px)] w-full max-w-6xl items-center px-4 py-12 md:px-6">
      <div className="grid w-full gap-8 md:grid-cols-[1.2fr_0.8fr]">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-slate-500">CrowdAlpha</p>
          <h1 className="mt-3 text-4xl font-semibold tracking-tight text-slate-100 md:text-5xl">
            Multi-Agent Market Lab
          </h1>
          <p className="mt-5 max-w-2xl text-base leading-7 text-slate-300">
            CrowdAlpha is a live limit-order-book simulation where strategies compete and alpha decays under
            crowding pressure.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => go("/dashboard")}
              className="rounded-lg bg-sky-500 px-5 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-sky-400"
            >
              Watch Live
            </button>
            <button
              type="button"
              onClick={() => go("/deploy")}
              className="rounded-lg border border-slate-600 px-5 py-2.5 text-sm font-semibold text-slate-100 transition hover:border-slate-400 hover:bg-slate-800/40"
            >
              Deploy Strategy
            </button>
          </div>
        </div>
        <div className="card">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">How It Works</h2>
          <ol className="mt-4 space-y-3 text-sm text-slate-300">
            <li>1. Observe price, liquidity, crowding, and regime shifts in real time.</li>
            <li>2. Submit strategy configs from templates in the deploy flow.</li>
            <li>3. Track rankings and diagnostics as alpha compresses under cloning pressure.</li>
          </ol>
        </div>
      </div>
    </div>
  );
}

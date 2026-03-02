const EXPERIMENTS = [
  {
    title: "Crash Fragility Under Momentum Crowding",
    summary:
      "Added additional momentum agents and measured spread widening, LFI spike behavior, and cross-agent Sharpe compression.",
    status: "Draft",
  },
  {
    title: "Order Book Resilience by Regime",
    summary:
      "Compared depth decay within 5 ticks during CALM vs CRASH_PRONE periods across repeated seeded runs.",
    status: "Planned",
  },
];

export function ResearchPage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-6 md:px-6 md:py-8">
      <header className="mb-4">
        <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Research</p>
        <h1 className="mt-2 text-2xl font-semibold text-slate-100">Published Experiments</h1>
      </header>
      <div className="grid gap-3">
        {EXPERIMENTS.map((exp) => (
          <article key={exp.title} className="card">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-base font-semibold text-slate-100">{exp.title}</h2>
              <span className="pill bg-slate-700/50 text-slate-200">{exp.status}</span>
            </div>
            <p className="text-sm text-slate-300">{exp.summary}</p>
          </article>
        ))}
      </div>
    </div>
  );
}

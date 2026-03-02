const BASE_AGENT_SNIPPET = `class MyAgent(BaseAgent):
    def __init__(self, agent_id, strategy_type="my_agent", initial_cash=100000.0):
        super().__init__(agent_id=agent_id, strategy_type=strategy_type, initial_cash=initial_cash)

    def on_tick(self, state):
        # return List[Order]
        return []

    def factor_vector(self):
        # return np.array of length 6
        return np.zeros(6)`;

export function DocsPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-6 md:px-6 md:py-8">
      <header className="mb-4">
        <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Docs</p>
        <h1 className="mt-2 text-2xl font-semibold text-slate-100">Base Agent Interface</h1>
      </header>

      <section className="card space-y-3 text-sm text-slate-300">
        <p>Your strategy must implement `on_tick(state)` and `factor_vector()`.</p>
        <p>
          `state` includes only observable fields: best bid/ask, depth, recent trades, own inventory/cash/pnl,
          volatility, crowding context.
        </p>
        <pre className="overflow-auto rounded border border-surface-3 bg-slate-950/60 p-3 text-xs text-slate-200">
          <code>{BASE_AGENT_SNIPPET}</code>
        </pre>
      </section>
    </div>
  );
}

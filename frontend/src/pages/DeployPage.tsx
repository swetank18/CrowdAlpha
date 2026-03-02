import { useMemo, useState, type FormEvent } from "react";
import { API_BASE } from "../config";

type Track = "beginner" | "intermediate" | "advanced";
type Template = "momentum" | "mean_reversion" | "market_maker";

type SubmitState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ok"; submissionId: string }
  | { kind: "error"; message: string };

const TEMPLATE_HELP: Record<Template, string> = {
  momentum: "Follows directional continuation. Works best in trending regimes.",
  mean_reversion: "Fades short-term dislocations around rolling fair value.",
  market_maker: "Provides two-sided liquidity and earns spread with inventory control.",
};

export function DeployPage() {
  const [track, setTrack] = useState<Track>("beginner");
  const [template, setTemplate] = useState<Template>("momentum");
  const [name, setName] = useState("");
  const [contact, setContact] = useState("");
  const [lookback, setLookback] = useState(30);
  const [aggressiveness, setAggressiveness] = useState(0.001);
  const [positionSize, setPositionSize] = useState(5);
  const [cloneSource, setCloneSource] = useState("");
  const [visibility, setVisibility] = useState<"public" | "private">("public");
  const [notes, setNotes] = useState("");
  const [submitState, setSubmitState] = useState<SubmitState>({ kind: "idle" });

  const templateHelp = useMemo(() => TEMPLATE_HELP[template], [template]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setSubmitState({ kind: "loading" });

    try {
      const res = await fetch(`${API_BASE}/strategies/submit-config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          submitter_name: name.trim(),
          contact: contact.trim(),
          mode: track,
          template_strategy: template,
          parameters: {
            lookback,
            aggression: aggressiveness,
            order_qty: positionSize,
          },
          clone_source_agent_id: cloneSource.trim() || null,
          visibility,
          notes: notes.trim() || null,
        }),
      });

      if (!res.ok) {
        let detail = `Request failed (${res.status})`;
        try {
          const payload = await res.json();
          detail = payload?.detail ?? detail;
        } catch {
          // Ignore parse failure and keep fallback detail.
        }
        throw new Error(detail);
      }

      const payload = await res.json();
      setSubmitState({
        kind: "ok",
        submissionId: String(payload?.submission_id ?? "unknown"),
      });
    } catch (err) {
      setSubmitState({
        kind: "error",
        message: err instanceof Error ? err.message : "Submission failed",
      });
    }
  }

  return (
    <div className="mx-auto w-full max-w-4xl px-4 py-6 md:px-6 md:py-8">
      <header className="mb-5">
        <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Deploy</p>
        <h1 className="mt-2 text-2xl font-semibold text-slate-100">Submit Strategy Configuration</h1>
        <p className="mt-2 text-sm text-slate-400">
          v1 flow is config-only. You submit parameters, then an admin approves and activates the strategy.
        </p>
      </header>

      <form onSubmit={onSubmit} className="card space-y-5">
        <div className="grid gap-2">
          <label className="text-xs uppercase tracking-wider text-slate-400">Track</label>
          <div className="flex flex-wrap gap-2">
            {(["beginner", "intermediate", "advanced"] as Track[]).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setTrack(mode)}
                className={`rounded-md border px-3 py-1.5 text-xs font-medium ${
                  track === mode
                    ? "border-sky-400 bg-sky-900/30 text-sky-200"
                    : "border-slate-600 text-slate-300 hover:border-slate-400"
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <label className="grid gap-1.5 text-sm text-slate-300">
            <span className="text-xs uppercase tracking-wider text-slate-500">Your Name</span>
            <input
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="rounded-md border border-slate-600 bg-slate-900/50 px-3 py-2 text-sm outline-none ring-sky-400 focus:ring"
            />
          </label>
          <label className="grid gap-1.5 text-sm text-slate-300">
            <span className="text-xs uppercase tracking-wider text-slate-500">Contact (email/discord)</span>
            <input
              required
              value={contact}
              onChange={(e) => setContact(e.target.value)}
              className="rounded-md border border-slate-600 bg-slate-900/50 px-3 py-2 text-sm outline-none ring-sky-400 focus:ring"
            />
          </label>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <label className="grid gap-1.5 text-sm text-slate-300">
            <span className="text-xs uppercase tracking-wider text-slate-500">Template</span>
            <select
              value={template}
              onChange={(e) => setTemplate(e.target.value as Template)}
              className="rounded-md border border-slate-600 bg-slate-900/50 px-3 py-2 text-sm outline-none ring-sky-400 focus:ring"
            >
              <option value="momentum">momentum</option>
              <option value="mean_reversion">mean_reversion</option>
              <option value="market_maker">market_maker</option>
            </select>
            <span className="text-xs text-slate-500">{templateHelp}</span>
          </label>

          <label className="grid gap-1.5 text-sm text-slate-300">
            <span className="text-xs uppercase tracking-wider text-slate-500">Clone Source (optional)</span>
            <input
              value={cloneSource}
              onChange={(e) => setCloneSource(e.target.value)}
              placeholder="agent_id from leaderboard"
              className="rounded-md border border-slate-600 bg-slate-900/50 px-3 py-2 text-sm outline-none ring-sky-400 focus:ring"
            />
          </label>
        </div>

        <div className="grid gap-4 md:grid-cols-3">
          <label className="grid gap-1.5 text-sm text-slate-300">
            <span className="text-xs uppercase tracking-wider text-slate-500">Lookback</span>
            <input
              type="number"
              min={5}
              max={300}
              value={lookback}
              onChange={(e) => setLookback(Number(e.target.value))}
              className="rounded-md border border-slate-600 bg-slate-900/50 px-3 py-2 text-sm outline-none ring-sky-400 focus:ring"
            />
          </label>

          <label className="grid gap-1.5 text-sm text-slate-300">
            <span className="text-xs uppercase tracking-wider text-slate-500">Aggressiveness</span>
            <input
              type="number"
              min={0}
              step={0.0001}
              value={aggressiveness}
              onChange={(e) => setAggressiveness(Number(e.target.value))}
              className="rounded-md border border-slate-600 bg-slate-900/50 px-3 py-2 text-sm outline-none ring-sky-400 focus:ring"
            />
          </label>

          <label className="grid gap-1.5 text-sm text-slate-300">
            <span className="text-xs uppercase tracking-wider text-slate-500">Position Size</span>
            <input
              type="number"
              min={1}
              max={1000}
              value={positionSize}
              onChange={(e) => setPositionSize(Number(e.target.value))}
              className="rounded-md border border-slate-600 bg-slate-900/50 px-3 py-2 text-sm outline-none ring-sky-400 focus:ring"
            />
          </label>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <label className="grid gap-1.5 text-sm text-slate-300">
            <span className="text-xs uppercase tracking-wider text-slate-500">Visibility</span>
            <select
              value={visibility}
              onChange={(e) => setVisibility(e.target.value as "public" | "private")}
              className="rounded-md border border-slate-600 bg-slate-900/50 px-3 py-2 text-sm outline-none ring-sky-400 focus:ring"
            >
              <option value="public">public</option>
              <option value="private">private</option>
            </select>
          </label>

          <label className="grid gap-1.5 text-sm text-slate-300">
            <span className="text-xs uppercase tracking-wider text-slate-500">Notes</span>
            <input
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="optional notes about your hypothesis"
              className="rounded-md border border-slate-600 bg-slate-900/50 px-3 py-2 text-sm outline-none ring-sky-400 focus:ring"
            />
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="submit"
            disabled={submitState.kind === "loading"}
            className="rounded-lg bg-sky-500 px-5 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:bg-slate-600"
          >
            {submitState.kind === "loading" ? "Submitting..." : "Submit Config"}
          </button>

          {submitState.kind === "ok" && (
            <span className="text-sm text-emerald-300">
              Submitted as <span className="font-mono">{submitState.submissionId}</span>
            </span>
          )}
          {submitState.kind === "error" && <span className="text-sm text-rose-300">{submitState.message}</span>}
        </div>
      </form>
    </div>
  );
}

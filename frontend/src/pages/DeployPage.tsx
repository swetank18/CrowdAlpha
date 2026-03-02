import { useEffect, useMemo, useState, type FormEvent } from "react";
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

const BEGINNER_DEFAULTS: Record<Template, { lookback: number; aggression: number; order_qty: number }> = {
  momentum: { lookback: 30, aggression: 0.0015, order_qty: 6 },
  mean_reversion: { lookback: 40, aggression: 0.001, order_qty: 5 },
  market_maker: { lookback: 20, aggression: 0.0008, order_qty: 8 },
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

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const qTrack = q.get("track");
    const qTemplate = q.get("template");
    const qClone = q.get("clone");
    const qLookback = q.get("lookback");
    const qAgg = q.get("aggression");
    const qPos = q.get("position_size");

    if (qTrack === "beginner" || qTrack === "intermediate" || qTrack === "advanced") {
      setTrack(qTrack);
    }
    if (qTemplate === "momentum" || qTemplate === "mean_reversion" || qTemplate === "market_maker") {
      setTemplate(qTemplate);
    }
    if (qClone) setCloneSource(qClone);
    if (qLookback) {
      const v = Number(qLookback);
      if (Number.isFinite(v)) setLookback(Math.max(5, Math.round(v)));
    }
    if (qAgg) {
      const v = Number(qAgg);
      if (Number.isFinite(v)) setAggressiveness(Math.max(0, v));
    }
    if (qPos) {
      const v = Number(qPos);
      if (Number.isFinite(v)) setPositionSize(Math.max(1, Math.round(v)));
    }
  }, []);

  function buildParameters() {
    if (track === "beginner") {
      return BEGINNER_DEFAULTS[template];
    }
    return {
      lookback,
      aggression: aggressiveness,
      order_qty: positionSize,
    };
  }

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
          parameters: buildParameters(),
          clone_source_agent_id: track === "beginner" ? null : cloneSource.trim() || null,
          visibility: track === "beginner" ? "public" : visibility,
          notes: track === "beginner" ? null : notes.trim() || null,
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

  if (submitState.kind === "ok") {
    return (
      <div className="mx-auto w-full max-w-4xl px-4 py-6 md:px-6 md:py-8">
        <div className="card space-y-4">
          <p className="text-xs uppercase tracking-[0.24em] text-emerald-400">Submission Received</p>
          <h1 className="text-2xl font-semibold text-slate-100">Your strategy is queued for activation.</h1>
          <div className="rounded-md border border-slate-700 bg-slate-900/40 px-3 py-2 text-sm text-slate-300">
            <p>
              Reference ID: <span className="font-mono text-slate-100">{submitState.submissionId}</span>
            </p>
            <p>
              Track: <span className="font-mono text-slate-100">{track}</span> | Template: {template}
            </p>
          </div>
          <ul className="space-y-1 text-sm text-slate-300">
            <li>1. Strategies are reviewed and activated within 2 hours.</li>
            <li>2. You will receive activation confirmation at your contact address with your agent reference.</li>
            <li>3. Watch market conditions and regime shifts while waiting for activation.</li>
          </ul>
          <div className="flex flex-wrap gap-3">
            <a
              href="/dashboard"
              className="rounded-lg bg-sky-500 px-5 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-sky-400"
            >
              Watch Dashboard
            </a>
            <button
              type="button"
              onClick={() => setSubmitState({ kind: "idle" })}
              className="rounded-lg border border-slate-600 px-5 py-2.5 text-sm font-semibold text-slate-100 transition hover:border-slate-400 hover:bg-slate-800/40"
            >
              Submit Another
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-4xl px-4 py-6 md:px-6 md:py-8">
      <header className="mb-5">
        <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Deploy</p>
        <h1 className="mt-2 text-2xl font-semibold text-slate-100">Submit Strategy Configuration</h1>
        <p className="mt-2 text-sm text-slate-400">
          Strategies are reviewed and activated within 2 hours. You will get a confirmation message with your
          reference ID.
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

          {track !== "beginner" && (
            <label className="grid gap-1.5 text-sm text-slate-300">
              <span className="text-xs uppercase tracking-wider text-slate-500">Clone Source (optional)</span>
              <input
                value={cloneSource}
                onChange={(e) => setCloneSource(e.target.value)}
                placeholder="agent_id from leaderboard"
                className="rounded-md border border-slate-600 bg-slate-900/50 px-3 py-2 text-sm outline-none ring-sky-400 focus:ring"
              />
            </label>
          )}
        </div>

        {track !== "beginner" && (
          <>
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
                <span className="text-xs text-slate-500">
                  Number of ticks used to compute signal. Higher = slower, smoother. Range: 10-100.
                </span>
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
                <span className="text-xs text-slate-500">
                  Order size relative to best quote. 0.001 = conservative, 0.01 = aggressive.
                </span>
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
                <span className="text-xs text-slate-500">
                  Max units held at any time. Larger = more market impact.
                </span>
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
                <span className="text-xs text-slate-500">
                  Public strategies can be discovered and cloned. Private strategies still run but are not cloneable.
                </span>
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
          </>
        )}

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="submit"
            disabled={submitState.kind === "loading"}
            className="rounded-lg bg-sky-500 px-5 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:bg-slate-600"
          >
            {submitState.kind === "loading" ? "Submitting..." : "Submit Config"}
          </button>

          {submitState.kind === "error" && <span className="text-sm text-rose-300">{submitState.message}</span>}
        </div>
      </form>
    </div>
  );
}

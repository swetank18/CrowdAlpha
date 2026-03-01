import { useEffect } from "react";
import { applyAnalyticsSnapshot } from "../store/simulation";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export function useAnalyticsSnapshot(pollMs = 2500) {
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | null = null;

    async function load() {
      try {
        const res = await fetch(`${API_BASE}/analytics/snapshot`);
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) applyAnalyticsSnapshot(data);
      } catch {
        // Keep retrying on next interval.
      }
    }

    load();
    timer = setInterval(load, pollMs);

    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [pollMs]);
}


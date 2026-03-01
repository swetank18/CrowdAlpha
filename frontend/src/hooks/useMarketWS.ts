import { useEffect, useRef } from "react";
import {
  applyCrowdingMatrixEvent,
  applyRegimeChangedEvent,
  applyTickEvent,
  useSimStore,
} from "../store/simulation";

const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws/market";
const RECONNECT_DELAY_MS = 2000;

interface WsMessage {
  schema_version?: number;
  type?: string;
  timestamp?: number;
  payload?: unknown;
}

export function useMarketWS() {
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const setConnected = useSimStore((s) => s.setConnected);

  function cleanupPing() {
    if (pingRef.current) {
      clearInterval(pingRef.current);
      pingRef.current = null;
    }
  }

  function connect() {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      cleanupPing();
      pingRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send("ping");
      }, 20_000);
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(String(evt.data)) as WsMessage;
        const type = msg.type;
        const payload = msg.payload;
        if (!type) return;

        switch (type) {
          case "TICK":
            applyTickEvent(payload);
            break;
          case "CROWDING_MATRIX_UPDATED":
            applyCrowdingMatrixEvent(payload);
            break;
          case "REGIME_CHANGED":
            applyRegimeChangedEvent(payload);
            break;
          case "HEARTBEAT":
          case "ORDER_SUBMITTED":
          case "ORDER_FILLED":
          case "PRICE_UPDATED":
          case "AGENT_STATE_CHANGED":
          default:
            break;
        }
      } catch {
        // Ignore malformed messages.
      }
    };

    ws.onclose = () => {
      cleanupPing();
      setConnected(false);
      retryRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
    };

    ws.onerror = () => {
      ws.close();
    };
  }

  useEffect(() => {
    connect();
    return () => {
      cleanupPing();
      if (retryRef.current) clearTimeout(retryRef.current);
      wsRef.current?.close();
    };
    // Intentionally run once for socket lifecycle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}

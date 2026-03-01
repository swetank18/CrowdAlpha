/**
 * hooks/useMarketWS.ts
 *
 * WebSocket hook — manages connection lifecycle, reconnection,
 * and dispatches incoming events to the Zustand store.
 */

import { useEffect, useRef } from "react";
import { useSimStore, applyTickEvent } from "../store/simulation";

const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws/market";
const RECONNECT_DELAY_MS = 2000;

export function useMarketWS() {
    const wsRef = useRef<WebSocket | null>(null);
    const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    function connect() {
        if (wsRef.current?.readyState === WebSocket.OPEN) return;

        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
            useSimStore.setState({ connected: true });
            // Heartbeat ping every 20s
            const ping = setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) ws.send("ping");
                else clearInterval(ping);
            }, 20_000);
        };

        ws.onmessage = (evt) => {
            try {
                const msg = JSON.parse(evt.data as string);
                if (msg.type === "TICK") {
                    applyTickEvent(msg.payload);
                }
                // Future: handle REGIME_CHANGE, FRAGILITY_WARNING, etc.
            } catch {
                // ignore malformed messages
            }
        };

        ws.onclose = () => {
            useSimStore.setState({ connected: false });
            // Reconnect after delay
            retryRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
        };

        ws.onerror = () => ws.close();
    }

    useEffect(() => {
        connect();
        return () => {
            if (retryRef.current) clearTimeout(retryRef.current);
            wsRef.current?.close();
        };
    }, []);
}

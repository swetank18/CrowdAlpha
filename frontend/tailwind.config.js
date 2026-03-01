/** @type {import('tailwindcss').Config} */
export default {
    content: ["./index.html", "./src/**/*.{ts,tsx}"],
    theme: {
        extend: {
            colors: {
                surface: { DEFAULT: "#0f1117", 1: "#161b27", 2: "#1e2535", 3: "#252d3d" },
                accent: { DEFAULT: "#6366f1", dim: "#4338ca" },
                bid: "#22c55e",
                ask: "#f43f5e",
                warn: "#f59e0b",
                crowd: "#8b5cf6",
            },
            fontFamily: {
                sans: ["Space Grotesk", "ui-sans-serif", "sans-serif"],
                mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
            },
        },
    },
    plugins: [],
};

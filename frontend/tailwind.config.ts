import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}", "./app/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#0f172a",
        card: "#1e293b",
        accent: "#22d3ee",
        success: "#22c55e",
        warning: "#facc15",
        danger: "#ef4444"
      }
    }
  },
  plugins: []
};

export default config;


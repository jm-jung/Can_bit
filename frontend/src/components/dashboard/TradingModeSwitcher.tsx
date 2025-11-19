"use client";

import { useSetTradeMode, useTradeMode } from "@/hooks/useBackend";

export default function TradingModeSwitcher() {
  const { data, isLoading } = useTradeMode();
  const mutation = useSetTradeMode();
  const mode = data?.mode ?? "SIM";

  const setMode = (next: "SIM" | "REAL") => {
    if (next === mode) return;
    mutation.mutate(next);
  };

  const buttonClass = (target: "SIM" | "REAL") => {
    const base = "flex-1 rounded px-3 py-2 text-sm font-semibold";
    if (mode === target) {
      if (target === "REAL") return `${base} bg-orange-500 text-black`;
      return `${base} bg-sky-600 text-white`;
    }
    return `${base} bg-slate-700 text-slate-300 hover:bg-slate-600`;
  };

  return (
    <div className="space-y-2">
      <p className="text-sm text-slate-400">Trading Mode</p>
      <div className="flex gap-2">
        <button
          disabled={isLoading || mutation.isPending}
          className={buttonClass("SIM")}
          onClick={() => setMode("SIM")}
        >
          SIM
        </button>
        <button
          disabled={isLoading || mutation.isPending}
          className={buttonClass("REAL")}
          onClick={() => setMode("REAL")}
        >
          REAL
        </button>
      </div>
    </div>
  );
}


"use client";

import { Signal } from "@/lib/types";

const colors: Record<Signal, string> = {
  LONG: "bg-success",
  SHORT: "bg-danger",
  HOLD: "bg-slate-600"
};

export default function SignalCard({ signal }: { signal: Signal }) {
  return (
    <div className="rounded-lg bg-card p-4">
      <p className="text-sm text-slate-400">Strategy Signal</p>
      <div
        className={`mt-2 inline-flex items-center rounded px-3 py-1 text-lg font-semibold ${colors[signal]}`}
      >
        {signal}
      </div>
    </div>
  );
}


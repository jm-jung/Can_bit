"use client";

import { LogEntry } from "@/lib/types";

interface Props {
  trades: LogEntry[];
  isLoading?: boolean;
}

export default function TradesTable({ trades, isLoading }: Props) {
  if (isLoading) {
    return <div className="h-48 animate-pulse rounded bg-card" />;
  }

  if (!trades.length) {
    return <p className="text-sm text-slate-400">No trade logs.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead className="bg-slate-800 text-left text-xs uppercase text-slate-400">
          <tr>
            <th className="px-3 py-2">Timestamp</th>
            <th className="px-3 py-2">Type</th>
            <th className="px-3 py-2">Side</th>
            <th className="px-3 py-2">Price</th>
            <th className="px-3 py-2">Amount</th>
            <th className="px-3 py-2">PnL %</th>
            <th className="px-3 py-2">Mode</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((trade) => (
            <tr key={trade.timestamp} className="odd:bg-slate-900">
              <td className="px-3 py-2">{trade.timestamp}</td>
              <td className="px-3 py-2">{trade.type}</td>
              <td className="px-3 py-2">{trade.side ?? "—"}</td>
              <td className="px-3 py-2">
                {(trade.exit_price ?? trade.entry_price ?? 0).toFixed?.(2) ??
                  trade.exit_price ??
                  trade.entry_price ??
                  "—"}
              </td>
              <td className="px-3 py-2">
                {trade.amount !== undefined ? trade.amount : "—"}
              </td>
              <td className="px-3 py-2">
                {trade.pnl_pct !== undefined
                  ? ((trade.pnl_pct ?? 0) * 100).toFixed(2) + "%"
                  : "—"}
              </td>
              <td className="px-3 py-2">{trade.mode ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}


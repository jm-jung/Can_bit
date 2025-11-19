"use client";

import { LogEntry } from "@/lib/types";

interface Props {
  logs: LogEntry[];
  isLoading?: boolean;
}

export default function LogsTable({ logs, isLoading }: Props) {
  if (isLoading) {
    return <div className="h-48 animate-pulse rounded bg-card" />;
  }

  if (!logs.length) {
    return <p className="text-sm text-slate-400">No logs available.</p>;
  }

  const keys = Array.from(
    new Set(logs.flatMap((entry) => Object.keys(entry))).add("timestamp")
  );

  return (
    <div className="overflow-x-auto text-sm">
      <table className="min-w-full">
        <thead className="bg-slate-800 text-left text-xs uppercase text-slate-400">
          <tr>
            {keys.map((key) => (
              <th key={key} className="px-3 py-2">
                {key}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {logs.map((entry, idx) => (
            <tr key={`${entry.timestamp}-${idx}`} className="odd:bg-slate-900">
              {keys.map((key) => (
                <td key={key} className="px-3 py-2">
                  {String(entry[key] ?? "—")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}


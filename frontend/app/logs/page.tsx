"use client";

import { useMemo, useState } from "react";
import Section from "@/components/common/Section";
import LogsTable from "@/components/common/LogsTable";
import { useErrorLogs, useTrades } from "@/hooks/useBackend";

const tabs = [
  { key: "errors", label: "Errors" },
  { key: "trades", label: "Trades" }
] as const;

export default function LogsPage() {
  const [tab, setTab] = useState<(typeof tabs)[number]["key"]>("errors");
  const errors = useErrorLogs(200);
  const trades = useTrades(200);

  const active = useMemo(() => {
    return tab === "errors" ? errors : trades;
  }, [errors, trades, tab]);

  return (
    <Section title="Logs">
      <div className="mb-4 flex gap-3">
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`rounded px-4 py-2 text-sm font-semibold ${
              tab === t.key ? "bg-accent text-black" : "bg-slate-700 text-slate-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <LogsTable logs={active.data ?? []} isLoading={active.isLoading} />
    </Section>
  );
}


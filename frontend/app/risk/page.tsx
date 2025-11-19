"use client";

import Section from "@/components/common/Section";
import LogsTable from "@/components/common/LogsTable";
import EquityChart from "@/components/dashboard/EquityChart";
import { useDailyReport, useRiskLogs } from "@/hooks/useBackend";

export default function RiskPage() {
  const { data: report } = useDailyReport();
  const { data: logs, isLoading } = useRiskLogs(200);

  return (
    <div className="space-y-6">
      <Section title="Daily Report">
        <div className="grid gap-4 md:grid-cols-3">
          <div className="rounded-lg bg-card p-4">
            <p className="text-sm text-slate-400">Trades</p>
            <p className="text-2xl font-semibold">
              {report?.num_trades ?? 0}
            </p>
          </div>
          <div className="rounded-lg bg-card p-4">
            <p className="text-sm text-slate-400">Win Rate</p>
            <p className="text-2xl font-semibold">
              {report ? (report.win_rate * 100).toFixed(1) : "0.0"}%
            </p>
          </div>
          <div className="rounded-lg bg-card p-4">
            <p className="text-sm text-slate-400">Total PnL</p>
            <p className="text-2xl font-semibold">
              {report ? (report.total_pnl_pct * 100).toFixed(2) : "0.00"}%
            </p>
          </div>
        </div>
      </Section>

      <Section title="Equity Curve">
        <EquityChart />
      </Section>

      <Section title="Risk Logs">
        <LogsTable logs={logs ?? []} isLoading={isLoading} />
      </Section>
    </div>
  );
}


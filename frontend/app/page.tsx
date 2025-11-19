"use client";

import MetricCard from "@/components/dashboard/MetricCard";
import SignalCard from "@/components/dashboard/SignalCard";
import RiskCard from "@/components/dashboard/RiskCard";
import TradingModeSwitcher from "@/components/dashboard/TradingModeSwitcher";
import LiveModeToggle from "@/components/dashboard/LiveModeToggle";
import EquityChart from "@/components/dashboard/EquityChart";
import Section from "@/components/common/Section";
import { useMonitor, useRealtimeLast, useRiskStatus } from "@/hooks/useBackend";
import TradesTable from "@/components/common/TradesTable";
import { useTrades } from "@/hooks/useBackend";

export default function DashboardPage() {
  const { data: realtime } = useRealtimeLast();
  const { data: monitor } = useMonitor();
  const { data: risk } = useRiskStatus();
  const { data: trades, isLoading: tradesLoading } = useTrades(10);

  return (
    <div className="space-y-6">
      <Section title="Realtime Dashboard">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            title="BTC Price"
            value={realtime?.latest_candle.close?.toLocaleString() ?? "—"}
            subtitle={realtime?.latest_candle.timestamp ?? ""}
            variant="primary"
          />
          <SignalCard signal={monitor?.last_signal ?? "HOLD"} />
          <RiskCard risk={monitor?.risk} />
          <div className="rounded-lg bg-card p-4 space-y-4">
            <TradingModeSwitcher />
            <LiveModeToggle />
          </div>
        </div>
      </Section>

      <Section title="Equity Curve">
        <EquityChart />
      </Section>

      <Section title="Recent Trades">
        <TradesTable trades={trades ?? []} isLoading={tradesLoading} />
      </Section>
    </div>
  );
}


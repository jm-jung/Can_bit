"use client";

import { useEquityCurve } from "@/hooks/useBackend";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

export default function EquityChart() {
  const { data, isLoading } = useEquityCurve();

  if (isLoading) {
    return <div className="h-64 animate-pulse rounded-lg bg-card" />;
  }

  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data ?? []}>
          <defs>
            <linearGradient id="equity" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis dataKey="timestamp" hide />
          <YAxis domain={["auto", "auto"]} tick={{ fill: "#94a3b8" }} />
          <Tooltip
            contentStyle={{ background: "#0f172a", border: "none" }}
            labelClassName="text-slate-300"
          />
          <Area
            type="monotone"
            dataKey="equity"
            stroke="#22d3ee"
            fill="url(#equity)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}


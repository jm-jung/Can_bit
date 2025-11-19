"use client";

interface RiskCardProps {
  risk?:
    | {
        equity?: number;
        drawdown_today?: number;
        trading_disabled?: boolean;
        reason?: string | null;
      }
    | undefined;
}

export default function RiskCard({ risk }: RiskCardProps) {
  if (!risk) {
    return (
      <div className="rounded-lg bg-card p-4">
        <p className="text-sm text-slate-400">Risk Status</p>
        <p className="mt-2 text-sm text-slate-500">Loading...</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg bg-card p-4 space-y-2 text-sm">
      <p className="text-slate-400">Risk Status</p>
      <div className="text-xl font-semibold">
        Equity: {(risk.equity ?? 0).toFixed(3)}
      </div>
      <div>Drawdown: {((risk.drawdown_today ?? 0) * 100).toFixed(2)}%</div>
      <div>
        Status:{" "}
        {risk.trading_disabled ? (
          <span className="text-danger font-semibold">Disabled</span>
        ) : (
          <span className="text-success font-semibold">Active</span>
        )}
      </div>
      {risk.reason && (
        <div className="text-xs text-warning">Reason: {risk.reason}</div>
      )}
    </div>
  );
}


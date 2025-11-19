"use client";

interface MetricCardProps {
  title: string;
  value: number | string;
  subtitle?: string;
  variant?: "primary" | "neutral";
}

export default function MetricCard({
  title,
  value,
  subtitle,
  variant = "neutral"
}: MetricCardProps) {
  const color = variant === "primary" ? "text-accent" : "text-white";

  return (
    <div className="rounded-lg bg-card p-4">
      <p className="text-sm text-slate-400">{title}</p>
      <p className={`mt-1 text-2xl font-semibold ${color}`}>{value}</p>
      {subtitle && (
        <p className="mt-1 text-xs text-slate-500 truncate">{subtitle}</p>
      )}
    </div>
  );
}


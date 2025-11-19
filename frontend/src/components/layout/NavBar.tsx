"use client";

import Link from "next/link";
import { useLiveMode, useTradeMode } from "@/hooks/useBackend";

const links = [
  { href: "/", label: "Dashboard" },
  { href: "/trades", label: "Trades" },
  { href: "/risk", label: "Risk" },
  { href: "/logs", label: "Logs" }
];

export default function NavBar() {
  const { data: mode } = useTradeMode();
  const { data: live } = useLiveMode();

  const pillClass = () => {
    if (mode?.mode === "REAL" && live?.live_mode) return "bg-danger";
    if (mode?.mode === "REAL") return "bg-orange-500";
    return "bg-sky-600";
  };

  return (
    <header className="bg-card px-6 py-3 shadow flex items-center justify-between">
      <div className="flex items-center gap-4">
        <span className="text-xl font-semibold text-accent">BTC Trader</span>
        <nav className="hidden md:flex items-center gap-3 text-sm text-slate-300">
          {links.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="hover:text-white transition"
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
      <div className="flex items-center gap-3 text-xs font-semibold uppercase tracking-wide">
        <span className={`${pillClass()} rounded-full px-3 py-1 text-black`}>
          {mode?.mode ?? "…"}
        </span>
        <span
          className={`rounded-full px-3 py-1 ${
            live?.live_mode ? "bg-danger text-white" : "bg-slate-600 text-slate-100"
          }`}
        >
          {live?.live_mode ? "LIVE ON" : "LIVE OFF"}
        </span>
      </div>
    </header>
  );
}


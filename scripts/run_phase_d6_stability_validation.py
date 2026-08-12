#!/usr/bin/env python3
"""
Phase D6: Multi-period stability validation. Run BEST candidate (p0575_e130) on 180d / 365d / 720d.
Same pinned end_date. Collect metrics, trade logs, stability verdict.
"""
from __future__ import annotations

import csv as csv_module
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG = PROJECT_ROOT / "data" / "diagnostics"
REPORTS = PROJECT_ROOT / "data" / "reports"
BACKTESTS = PROJECT_ROOT / "data" / "backtests"
PREFIX = "tcn_candidate_validation_BTCUSDT_5m_h15_t0p004"

END_DATE = "2026-03-03"
# 365d start for pinning (end - 366 days)
START_365 = "2025-03-02"

# Stability thresholds
COST_ON_FLOOR = -0.08   # PASS: cost_on >= this in all periods
MDD_CEILING = 0.15      # PASS: max_drawdown <= this
TRADES_FLOOR = 100      # PASS: trades >= this in all periods

RUN_ID = "phase_d6"
DAYS_LIST = [180, 365, 720]

COMMON = [
    "--id", "h15_t0p004", "--symbol", "BTCUSDT", "--timeframe", "5m",
    "--min-max-proba", "0.575", "--max-entropy", "1.30",
    "--min-hold", "36", "--cooldown", "12",
    "--commission", "0.0009", "--slippage", "0.0001",
    "--regime-filter", "off", "--position-scaling", "off",
    "--time-stop", "on", "--time-stop-bars", "72",
    "--early-exit", "on", "--early-exit-lookback", "12",
    "--early-exit-p-floor", "0.55", "--early-exit-bad-k", "8",
    "--partial-tp", "off", "--break-even-stop", "off",
    "--end-date", END_DATE, "--start-date-365", START_365,
    "--days-list", ",".join(str(d) for d in DAYS_LIST),
    "--run-id", RUN_ID,
    "--emit-trade-log", "on",
    "--trade-log-path", str(BACKTESTS),
]


def _env():
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    return env


def run_validation() -> int:
    cmd = [
        sys.executable, "-X", "faulthandler", "-m", "scripts.run_tcn_candidate_validation",
        *COMMON,
    ]
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=_env(), timeout=7200)
    return r.returncode


def load_json(run_id: str) -> dict | None:
    for f in sorted(DIAG.glob(f"{PREFIX}_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        if run_id in f.stem:
            try:
                with open(f, encoding="utf-8") as fp:
                    d = json.load(fp)
                if d.get("meta", {}).get("run_id") == run_id:
                    return d
            except Exception as e:
                print(f"Load error {f}: {e}", file=sys.stderr)
    return None


def get_row(results: list, days: int) -> dict | None:
    for r in results:
        if r.get("days") == days:
            return r
    return None


def mean_median_hold(csv_path: Path) -> tuple[float | None, float | None]:
    if not csv_path.exists():
        return None, None
    holds: list[float] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            h = row.get("holding_bars", "")
            try:
                holds.append(float(h))
            except (ValueError, TypeError):
                pass
    if not holds:
        return None, None
    n = len(holds)
    mean_h = sum(holds) / n
    sorted_h = sorted(holds)
    median_h = sorted_h[n // 2] if n else None
    return mean_h, median_h


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    BACKTESTS.mkdir(parents=True, exist_ok=True)

    # 1) Run backtests for 180, 365, 720
    print("  Running validation for 180d, 365d, 720d...")
    if run_validation() != 0:
        print("ERROR: Validation failed.", file=sys.stderr)
        return 1

    # 2) Load results
    d = load_json(RUN_ID)
    if not d:
        print("ERROR: JSON not found for run_id=", RUN_ID, file=sys.stderr)
        return 1
    results = d.get("results", [])

    # 3) Build rows with mean_hold / median_hold from trade logs
    rows_by_period: dict[int, dict] = {}
    for days in DAYS_LIST:
        row = get_row(results, days)
        if not row:
            print(f"WARNING: No result for {days}d", file=sys.stderr)
            continue
        csv_raw = BACKTESTS / f"{RUN_ID}_{days}d.csv"
        mean_hold, median_hold = mean_median_hold(csv_raw)
        rows_by_period[days] = {
            "period": days,
            "trades": row.get("trades"),
            "cost_on": row.get("cost_on_return"),
            "cost_off": row.get("cost_off_return"),
            "max_drawdown": row.get("max_drawdown"),
            "win_rate_on": row.get("win_rate_on"),
            "win_rate_off": row.get("win_rate_off"),
            "mean_hold": mean_hold,
            "median_hold": median_hold,
        }

    # 4) Export trade logs with required columns
    for days in DAYS_LIST:
        src = BACKTESTS / f"{RUN_ID}_{days}d.csv"
        dst = BACKTESTS / f"phase_d6_{days}d_trades.csv"
        if src.exists():
            with open(src, encoding="utf-8") as f:
                reader = csv_module.DictReader(f)
                rows_csv = list(reader)
            if rows_csv:
                with open(dst, "w", encoding="utf-8", newline="") as f:
                    w = csv_module.DictWriter(f, fieldnames=["entry_time", "exit_time", "pnl", "exit_reason", "holding_bars"])
                    w.writeheader()
                    for r in rows_csv:
                        w.writerow({
                            "entry_time": r.get("entry_ts", r.get("entry_time", "")),
                            "exit_time": r.get("exit_ts", r.get("exit_time", "")),
                            "pnl": r.get("net_return", r.get("pnl", "")),
                            "exit_reason": r.get("exit_reason", ""),
                            "holding_bars": r.get("holding_bars", ""),
                        })
                print(f"  Exported {dst}")

    # 5) Stability evaluation
    periods_ok = list(rows_by_period.keys())
    cost_on_ok = all(rows_by_period[d].get("cost_on") is not None and rows_by_period[d]["cost_on"] >= COST_ON_FLOOR for d in periods_ok)
    mdd_ok = all(rows_by_period[d].get("max_drawdown") is not None and rows_by_period[d]["max_drawdown"] <= MDD_CEILING for d in periods_ok)
    trades_ok = all(rows_by_period[d].get("trades") is not None and rows_by_period[d]["trades"] >= TRADES_FLOOR for d in periods_ok)
    stability_pass = cost_on_ok and mdd_ok and trades_ok

    # Sign consistency: all cost_on same sign or mixed
    cost_ons = [rows_by_period[d].get("cost_on") for d in periods_ok if rows_by_period[d].get("cost_on") is not None]
    sign_consistent = (all(c >= 0 for c in cost_ons) or all(c < 0 for c in cost_ons)) if cost_ons else True
    # Trade density: trades per 100 days roughly stable
    densities = []
    for d in periods_ok:
        t = rows_by_period[d].get("trades")
        if t is not None and d and d > 0:
            densities.append(t / (d / 100.0))
    density_stable = (max(densities) - min(densities)) <= 50 if len(densities) >= 2 else True

    verdict = "PASS" if stability_pass else "FAIL"
    recommendation = "OPS candidate" if (stability_pass and sign_consistent) else "further tuning"

    # 6) Report
    out_md = REPORTS / "phase_d6_stability_summary.md"
    lines = [
        "# Phase D6 multi-period stability summary",
        "",
        f"- candidate: min_max_proba=0.575, max_entropy=1.30, end_date={END_DATE}",
        f"- PASS rules: cost_on >= {COST_ON_FLOOR} all, max_drawdown <= {MDD_CEILING}, trades >= {TRADES_FLOOR} all",
        "",
        "| period | trades | cost_on | cost_off | max_drawdown | win_rate | mean_hold |",
        "|--------|--------|---------|----------|--------------|----------|-----------|",
    ]
    for days in DAYS_LIST:
        r = rows_by_period.get(days, {})
        tr = r.get("trades") if r.get("trades") is not None else "N/A"
        co = f"{r['cost_on']:.4f}" if r.get("cost_on") is not None else "N/A"
        cf = f"{r['cost_off']:.4f}" if r.get("cost_off") is not None else "N/A"
        mdd = f"{r['max_drawdown']:.4f}" if r.get("max_drawdown") is not None else "N/A"
        wr = f"{r['win_rate_on']:.4f}" if r.get("win_rate_on") is not None else "N/A"
        mh = f"{r['mean_hold']:.1f}" if r.get("mean_hold") is not None else "N/A"
        lines.append(f"| {days}d | {tr} | {co} | {cf} | {mdd} | {wr} | {mh} |")
    lines.extend([
        "",
        "## Stability",
        f"- cost_on >= {COST_ON_FLOOR} (all): **{'OK' if cost_on_ok else 'FAIL'}**",
        f"- max_drawdown <= {MDD_CEILING}: **{'OK' if mdd_ok else 'FAIL'}**",
        f"- trades >= {TRADES_FLOOR} (all): **{'OK' if trades_ok else 'FAIL'}**",
        f"- cost_on sign consistency: **{'OK' if sign_consistent else 'mixed'}**",
        f"- trade density stability: **{'OK' if density_stable else 'FLAG'}**",
        "",
        f"## Verdict: **{verdict}**",
        f"## Recommendation: **{recommendation}**",
        "",
    ])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {out_md}")

    # 7) Final output
    def metrics_str(days: int) -> str:
        r = rows_by_period.get(days, {})
        return (
            f"trades={r.get('trades')} cost_on={r.get('cost_on')} cost_off={r.get('cost_off')} "
            f"mdd={r.get('max_drawdown')} win_rate={r.get('win_rate_on')} mean_hold={r.get('mean_hold')}"
        )

    print("")
    print("=" * 60)
    print("Phase D6 completed")
    print("=" * 60)
    print(f"1) report file path: {out_md}")
    print(f"2) metrics 180d: {metrics_str(180)}")
    print(f"3) metrics 365d: {metrics_str(365)}")
    print(f"4) metrics 720d: {metrics_str(720)}")
    print(f"5) stability verdict: {verdict}")
    print(f"6) recommendation: {recommendation}")
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())

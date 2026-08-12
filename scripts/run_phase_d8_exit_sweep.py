#!/usr/bin/env python3
"""
Phase D8: Exit parameter sweep. Sweep time_stop_bars × early_exit_bad_k on 180/365/720d.
Entry fixed: min_max_proba=0.575, max_entropy=1.30. Baseline: time_stop=72, bad_k=8.
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
START_365 = "2025-03-02"
DAYS_LIST = [180, 365, 720]

# Baseline reference
BASELINE_180_COST = 0.0125
BASELINE_365_COST = -0.0570
BASELINE_720_COST = -0.1120
BASELINE_720_MDD = 0.1523
BASELINE_720_TRADES = 1491
BASELINE_MISMATCH_PCT = 0.05

# Verdict
COST_720_IMPROVE_MIN = 0.02       # ADOPT: 720d cost_on >= baseline + this
MDD_720_IMPROVE_MIN = 0.01        # ADOPT: 720d MDD <= baseline - this
TRADES_720_TOLERANCE = 0.15       # ADOPT: trades within ±15% of baseline
COST_180_FLOOR = 0.0              # ADOPT/REJECT: 180d cost_on >= 0
COST_720_REJECT_WORSE = 0.01      # REJECT: 720d cost_on worse than baseline by this
SPOT_STABLE_720_RANGE = 0.005     # STABLE if 720d cost_on range <= this

TIME_STOP_VALS = [48, 72, 96, 120]
EARLY_EXIT_BAD_K_VALS = [6, 8, 10]

# (run_id, time_stop_bars, early_exit_bad_k). Baseline = (72, 8)
RUNS = [
    (f"phase_d8_ts{ts}_bk{bk}", ts, bk)
    for ts in TIME_STOP_VALS for bk in EARLY_EXIT_BAD_K_VALS
]

COMMON_BASE = [
    "--id", "h15_t0p004", "--symbol", "BTCUSDT", "--timeframe", "5m",
    "--min-max-proba", "0.575", "--max-entropy", "1.30",
    "--min-hold", "36", "--cooldown", "12",
    "--commission", "0.0009", "--slippage", "0.0001",
    "--regime-filter", "off", "--position-scaling", "off",
    "--time-stop", "on",
    "--early-exit", "on", "--early-exit-lookback", "12",
    "--early-exit-p-floor", "0.55",
    "--partial-tp", "off", "--break-even-stop", "off",
    "--end-date", END_DATE, "--start-date-365", START_365,
    "--days-list", ",".join(str(d) for d in DAYS_LIST),
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


def run_validation(
    run_id: str,
    time_stop_bars: int,
    early_exit_bad_k: int,
) -> int:
    cmd = [
        sys.executable, "-X", "faulthandler", "-m", "scripts.run_tcn_candidate_validation",
        *COMMON_BASE,
        "--time-stop-bars", str(time_stop_bars),
        "--early-exit-bad-k", str(early_exit_bad_k),
        "--run-id", run_id,
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


def find_json_path(run_id: str) -> Path | None:
    for f in DIAG.glob(f"{PREFIX}_*.json"):
        if run_id in f.stem:
            try:
                with open(f, encoding="utf-8") as fp:
                    d = json.load(fp)
                if d.get("meta", {}).get("run_id") == run_id:
                    return f
            except Exception:
                pass
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

    # 1) Run all 12 exit combinations
    for run_id, ts, bk in RUNS:
        print(f"  Run {run_id} (time_stop={ts}, early_exit_bad_k={bk})...")
        if run_validation(run_id, ts, bk) != 0:
            print(f"ERROR: {run_id} failed.", file=sys.stderr)
            return 1

    # 2) Load results
    all_rows: list[dict] = []
    baseline_row = None
    for run_id, ts, bk in RUNS:
        d = load_json(run_id)
        if not d:
            print(f"WARNING: JSON not found for {run_id}", file=sys.stderr)
            continue
        res = d.get("results", [])
        r180 = get_row(res, 180)
        r365 = get_row(res, 365)
        r720 = get_row(res, 720)
        mean_hold_720 = mean_median_hold(BACKTESTS / f"{run_id}_720d.csv")[0]
        row = {
            "run_id": run_id,
            "time_stop": ts,
            "early_exit_bad_k": bk,
            "c180": r180.get("cost_on_return") if r180 else None,
            "c365": r365.get("cost_on_return") if r365 else None,
            "c720": r720.get("cost_on_return") if r720 else None,
            "mdd720": r720.get("max_drawdown") if r720 else None,
            "t720": r720.get("trades") if r720 else None,
            "mean_hold_720": mean_hold_720,
            "exit_reason_counts_720": r720.get("exit_reason_counts") if r720 else None,
            "time_stop_exit_count_720": r720.get("time_stop_exit_count") if r720 else None,
            "early_exit_count_720": r720.get("early_exit_count") if r720 else None,
        }
        all_rows.append(row)
        if ts == 72 and bk == 8:
            baseline_row = row
            if r180 and r365 and r720:
                c180, c365, c720 = row["c180"], row["c365"], row["c720"]
                if c180 is None or c365 is None or c720 is None:
                    print("ERROR: Baseline missing cost_on.", file=sys.stderr)
                    return 1
                if abs(c180 - BASELINE_180_COST) / max(abs(BASELINE_180_COST), 1e-6) > BASELINE_MISMATCH_PCT:
                    print(f"ERROR: Baseline 180d cost_on drift >5%: got {c180}, expected ~{BASELINE_180_COST}. STOP.", file=sys.stderr)
                    return 1
                if abs(c365 - BASELINE_365_COST) / max(abs(BASELINE_365_COST), 1e-6) > BASELINE_MISMATCH_PCT:
                    print(f"ERROR: Baseline 365d cost_on drift >5%: got {c365}, expected ~{BASELINE_365_COST}. STOP.", file=sys.stderr)
                    return 1
                if abs(c720 - BASELINE_720_COST) / max(abs(BASELINE_720_COST), 1e-6) > BASELINE_MISMATCH_PCT:
                    print(f"ERROR: Baseline 720d cost_on drift >5%: got {c720}, expected ~{BASELINE_720_COST}. STOP.", file=sys.stderr)
                    return 1
                print(f"  Baseline (ts=72, bk=8) OK: 180d={c180:.4f}, 365d={c365:.4f}, 720d={c720:.4f}")

    if not baseline_row:
        baseline_row = next((r for r in all_rows if r.get("time_stop") == 72 and r.get("early_exit_bad_k") == 8), None)
    c720_base = baseline_row["c720"] if baseline_row else BASELINE_720_COST
    mdd720_base = baseline_row["mdd720"] if baseline_row else BASELINE_720_MDD
    t720_base = baseline_row["t720"] if baseline_row else BASELINE_720_TRADES

    # 3) Verdict
    def verdict_for(r):
        c180 = r.get("c180")
        c720 = r.get("c720")
        mdd720 = r.get("mdd720")
        t720 = r.get("t720")
        if c180 is None or c720 is None:
            return "REJECT"
        if c180 < COST_180_FLOOR:
            return "REJECT"
        if c720 <= c720_base - COST_720_REJECT_WORSE:
            return "REJECT"
        if c720 >= c720_base + COST_720_IMPROVE_MIN and (mdd720 is None or mdd720 <= mdd720_base - MDD_720_IMPROVE_MIN):
            if t720 is not None and t720_base and abs(t720 - t720_base) / t720_base <= TRADES_720_TOLERANCE:
                return "ADOPT_CANDIDATE"
        return "NO_IMPROVE"

    non_reject = [r for r in all_rows if verdict_for(r) != "REJECT"]
    if non_reject:
        best_row = max(non_reject, key=lambda x: (x["c720"] or -1e9))
    else:
        best_row = max([r for r in all_rows if r.get("c720") is not None], key=lambda x: x["c720"])
    best_run_id = best_row["run_id"]
    best_ts, best_bk = best_row["time_stop"], best_row["early_exit_bad_k"]
    verdict = verdict_for(best_row)

    # 4) Spotcheck BEST
    print(f"  Spotcheck BEST {best_run_id} (2 runs)...")
    if run_validation("phase_d8_best_spot1", best_ts, best_bk) != 0:
        print("ERROR: phase_d8_best_spot1 failed.", file=sys.stderr)
        return 1
    if run_validation("phase_d8_best_spot2", best_ts, best_bk) != 0:
        print("ERROR: phase_d8_best_spot2 failed.", file=sys.stderr)
        return 1
    d0 = load_json(best_run_id)
    s1 = load_json("phase_d8_best_spot1")
    s2 = load_json("phase_d8_best_spot2")
    def get_c(res, days):
        r = get_row(res, days) if res else None
        return r.get("cost_on_return") if r else None
    c720_vals = [get_c(d0.get("results", []), 720), get_c(s1.get("results", []), 720), get_c(s2.get("results", []), 720)]
    c720_vals = [x for x in c720_vals if x is not None]
    c720_range = (max(c720_vals) - min(c720_vals)) if len(c720_vals) >= 2 else 0.0
    spot_stable = "STABLE" if c720_range <= SPOT_STABLE_720_RANGE else "FLAG"

    # 5) Report
    out_md = REPORTS / "phase_d8_exit_sweep_summary.md"
    lines = [
        "# Phase D8 exit sweep summary",
        "",
        "## A) Baseline (time_stop=72, early_exit_bad_k=8)",
        f"- 180d cost_on≈{baseline_row['c180']:.4f}, 365d≈{baseline_row['c365']:.4f}, 720d≈{baseline_row['c720']:.4f}" if baseline_row else "",
        f"- 720d MDD≈{baseline_row['mdd720']:.4f}, 720d trades≈{baseline_row['t720']}" if baseline_row else "",
        "",
        "## B) Full results table",
        "",
        "| run_id | time_stop | early_exit_bad_k | 180d cost_on | 365d cost_on | 720d cost_on | 720d MDD | 720d trades | mean_hold |",
        "|--------|-----------|------------------|--------------|--------------|--------------|----------|-------------|-----------|",
    ]
    for r in all_rows:
        c180 = f"{r['c180']:.4f}" if r.get("c180") is not None else "N/A"
        c365 = f"{r['c365']:.4f}" if r.get("c365") is not None else "N/A"
        c720 = f"{r['c720']:.4f}" if r.get("c720") is not None else "N/A"
        mdd = f"{r['mdd720']:.4f}" if r.get("mdd720") is not None else "N/A"
        tr = str(int(r["t720"])) if r.get("t720") is not None else "N/A"
        mh = f"{r['mean_hold_720']:.1f}" if r.get("mean_hold_720") is not None else "N/A"
        lines.append(f"| {r['run_id']} | {r['time_stop']} | {r['early_exit_bad_k']} | {c180} | {c365} | {c720} | {mdd} | {tr} | {mh} |")
    sorted_by_720 = sorted([r for r in all_rows if r.get("c720") is not None], key=lambda x: x["c720"], reverse=True)[:3]
    lines.extend([
        "",
        "## C) Top3 by 720d cost_on",
        "",
    ])
    for i, r in enumerate(sorted_by_720, 1):
        lines.append(f"{i}. **{r['run_id']}** (time_stop={r['time_stop']}, bad_k={r['early_exit_bad_k']}) 720d cost_on={r['c720']:.4f}")
    lines.extend([
        "",
        "## BEST & verdict",
        f"- BEST run_id: **{best_run_id}**",
        f"- spot stability: **{spot_stable}**",
        f"- final verdict: **{verdict}**",
        "",
    ])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {out_md}")

    # 6) Export BEST JSON
    src_json = find_json_path(best_run_id)
    if src_json and src_json.exists():
        import shutil
        shutil.copy(src_json, BACKTESTS / "phase_d8_best_run.json")
        print(f"  Exported {BACKTESTS / 'phase_d8_best_run.json'}")

    # 7) Export BEST trade logs
    for days in DAYS_LIST:
        src = BACKTESTS / f"{best_run_id}_{days}d.csv"
        dst = BACKTESTS / f"phase_d8_best_trades_{days}d.csv"
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

    # 8) Final output
    print("")
    print("=" * 60)
    print("Phase D8 completed")
    print("=" * 60)
    print(f"1) report path: {out_md}")
    print(f"2) BEST run_id: {best_run_id}")
    print(f"3) BEST parameters: time_stop={best_ts}, early_exit_bad_k={best_bk}")
    print(f"4) 180d cost_on: {best_row['c180']:.4f}" if best_row.get("c180") is not None else "4) 180d cost_on: N/A")
    print(f"5) 365d cost_on: {best_row['c365']:.4f}" if best_row.get("c365") is not None else "5) 365d cost_on: N/A")
    print(f"6) 720d cost_on: {best_row['c720']:.4f}" if best_row.get("c720") is not None else "6) 720d cost_on: N/A")
    print(f"7) 720d MDD: {best_row['mdd720']:.4f}" if best_row.get("mdd720") is not None else "7) 720d MDD: N/A")
    print(f"8) 720d trades: {best_row['t720']}" if best_row.get("t720") is not None else "8) 720d trades: N/A")
    print(f"9) spot stability: {spot_stable}")
    print(f"10) final verdict: {verdict}")
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())

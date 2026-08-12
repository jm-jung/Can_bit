#!/usr/bin/env python3
"""
Phase D5: Entry micro-tuning. Sweep min_max_proba × max_entropy around anchor (0.57, 1.30).
4 × 3 = 12 runs. Fixed: regime off, position_scaling off, same dates.
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

BASELINE_COST_ON = -0.0651
BASELINE_MDD = 0.1054
BASELINE_TRADES = 707
BASELINE_MISMATCH_PCT = 0.05
COST_ON_IMPROVE_MIN = 0.003   # ADOPT if cost_on >= baseline + this
MDD_TOLERANCE = 0.01          # ADOPT if MDD <= baseline + this
TRADES_FLOOR_ADOPT = 0.80     # ADOPT if trades >= baseline * this
TRADES_FLOOR_REJECT = 0.60    # REJECT if trades < baseline * this
NO_IMPROVE_DELTA = 0.003      # NO_IMPROVE if |cost_on - baseline| < this
SPOT_STABLE_COST_RANGE = 0.003

END_DATE = "2026-03-03"
START_30 = "2026-02-01"
START_365 = "2025-03-02"

# Common args; min_max_proba and max_entropy overridden per run
COMMON_BASE = [
    "--id", "h15_t0p004", "--symbol", "BTCUSDT", "--timeframe", "5m",
    "--min-hold", "36", "--cooldown", "12",
    "--commission", "0.0009", "--slippage", "0.0001",
    "--regime-filter", "off", "--position-scaling", "off",
    "--time-stop", "on", "--time-stop-bars", "72",
    "--early-exit", "on", "--early-exit-lookback", "12",
    "--early-exit-p-floor", "0.55", "--early-exit-bad-k", "8",
    "--partial-tp", "off", "--break-even-stop", "off",
    "--end-date", END_DATE, "--start-date-30", START_30, "--start-date-365", START_365,
    "--days-list", "30,365",
]

MIN_MAX_PROBA = [0.565, 0.570, 0.575, 0.580]
MAX_ENTROPY = [1.20, 1.25, 1.30]

# (run_id, min_max_proba, max_entropy)
RUNS = [
    (f"phase_d5_p{int(p * 1000):04d}_e{int(e * 100):03d}", p, e)
    for p in MIN_MAX_PROBA for e in MAX_ENTROPY
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
    min_max_proba: float,
    max_entropy: float,
    emit_trade_log: bool = False,
    trade_log_path: str | None = None,
) -> int:
    cmd = [
        sys.executable, "-X", "faulthandler", "-m", "scripts.run_tcn_candidate_validation",
        *COMMON_BASE,
        "--min-max-proba", str(min_max_proba),
        "--max-entropy", str(max_entropy),
        "--run-id", run_id,
    ]
    if emit_trade_log:
        cmd.extend(["--emit-trade-log", "on"])
    if trade_log_path:
        cmd.extend(["--trade-log-path", trade_log_path])
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=_env(), timeout=3600)
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


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    BACKTESTS.mkdir(parents=True, exist_ok=True)

    # 1) Run all 12 combinations
    for run_id, p, e in RUNS:
        print(f"  Run {run_id} (min_max_proba={p}, max_entropy={e})...")
        if run_validation(run_id, p, e) != 0:
            print(f"ERROR: {run_id} failed.", file=sys.stderr)
            return 1

    # 2) Load results; validate anchor (0.57, 1.30) = phase_d5_p0570_e130
    anchor_run_id = "phase_d5_p0570_e130"
    all_rows = []
    for run_id, p, e in RUNS:
        d = load_json(run_id)
        if not d:
            print(f"WARNING: JSON not found for {run_id}", file=sys.stderr)
            continue
        res = d.get("results", [])
        r30 = get_row(res, 30)
        r365 = get_row(res, 365)
        row = {
            "run_id": run_id,
            "min_max_proba": p,
            "max_entropy": e,
            "c30": r30.get("cost_on_return") if r30 else None,
            "c365": r365.get("cost_on_return") if r365 else None,
            "mdd365": r365.get("max_drawdown") if r365 else None,
            "trades365": r365.get("trades") if r365 else None,
        }
        all_rows.append(row)
        if run_id == anchor_run_id:
            cost_b = row["c365"]
            trades_b = row["trades365"]
            if cost_b is None or trades_b is None:
                print("ERROR: Anchor run missing cost_on or trades.", file=sys.stderr)
                return 1
            if abs(cost_b - BASELINE_COST_ON) / abs(BASELINE_COST_ON) > BASELINE_MISMATCH_PCT:
                print(f"ERROR: Anchor cost_on drift > 5%: got {cost_b}, expected ~{BASELINE_COST_ON}. STOP.", file=sys.stderr)
                return 1
            if abs(trades_b - BASELINE_TRADES) / BASELINE_TRADES > BASELINE_MISMATCH_PCT:
                print(f"ERROR: Anchor trades drift > 5%: got {trades_b}, expected ~{BASELINE_TRADES}. STOP.", file=sys.stderr)
                return 1
            print(f"  Anchor OK: cost_on={cost_b:.4f}, MDD={row['mdd365']}, trades={trades_b}")

    baseline_row = next((r for r in all_rows if r["run_id"] == anchor_run_id), None)
    cost_baseline = baseline_row["c365"] if baseline_row else BASELINE_COST_ON
    mdd_baseline = baseline_row["mdd365"] if baseline_row else BASELINE_MDD
    trades_baseline = baseline_row["trades365"] if baseline_row else BASELINE_TRADES

    # 3) Verdict: REJECT / NO_IMPROVE / ADOPT_CANDIDATE; BEST = max cost_on among non-REJECT
    def verdict_for(r):
        c = r.get("c365")
        mdd = r.get("mdd365")
        tr = r.get("trades365")
        if c is None or tr is None:
            return "REJECT"
        if c <= cost_baseline - COST_ON_IMPROVE_MIN or tr < trades_baseline * TRADES_FLOOR_REJECT:
            return "REJECT"
        if abs(c - cost_baseline) < NO_IMPROVE_DELTA:
            return "NO_IMPROVE"
        if mdd is not None and mdd > mdd_baseline + MDD_TOLERANCE:
            return "REJECT"
        if c >= cost_baseline + COST_ON_IMPROVE_MIN and tr >= trades_baseline * TRADES_FLOOR_ADOPT:
            return "ADOPT_CANDIDATE"
        return "NO_IMPROVE"

    non_reject = [r for r in all_rows if verdict_for(r) != "REJECT"]
    if non_reject:
        best_row = max(non_reject, key=lambda x: (x["c365"] or -1e9))
    else:
        best_row = max([r for r in all_rows if r.get("c365") is not None], key=lambda x: x["c365"])
    best_run_id = best_row["run_id"]
    best_p, best_e = best_row["min_max_proba"], best_row["max_entropy"]
    verdict = verdict_for(best_row)

    # 4) Spotcheck BEST (2 runs)
    print(f"  Spotcheck BEST {best_run_id} (2 runs)...")
    if run_validation("phase_d5_best_spot1", best_p, best_e) != 0:
        print("ERROR: phase_d5_best_spot1 failed.", file=sys.stderr)
        return 1
    if run_validation("phase_d5_best_spot2", best_p, best_e) != 0:
        print("ERROR: phase_d5_best_spot2 failed.", file=sys.stderr)
        return 1
    d0 = load_json(best_run_id)
    spot1 = load_json("phase_d5_best_spot1")
    spot2 = load_json("phase_d5_best_spot2")
    r0 = get_row(d0.get("results", []), 365) if d0 else None
    r1 = get_row(spot1.get("results", []), 365) if spot1 else None
    r2 = get_row(spot2.get("results", []), 365) if spot2 else None
    c_vals = [r0.get("cost_on_return"), r1.get("cost_on_return") if r1 else None, r2.get("cost_on_return") if r2 else None]
    c_vals = [x for x in c_vals if x is not None]
    mdd_vals = [r0.get("max_drawdown"), r1.get("max_drawdown") if r1 else None, r2.get("max_drawdown") if r2 else None]
    mdd_vals = [x for x in mdd_vals if x is not None]
    tr_vals = [r0.get("trades"), r1.get("trades") if r1 else None, r2.get("trades") if r2 else None]
    tr_vals = [x for x in tr_vals if x is not None]
    cost_on_range = (max(c_vals) - min(c_vals)) if len(c_vals) >= 2 else 0.0
    mdd_range = (max(mdd_vals) - min(mdd_vals)) if len(mdd_vals) >= 2 else 0.0
    trades_range = (max(tr_vals) - min(tr_vals)) if len(tr_vals) >= 2 else 0
    spot_stable = "STABLE" if cost_on_range <= SPOT_STABLE_COST_RANGE else "FLAG"

    # 5) Report
    out_md = REPORTS / "phase_d5_entry_micro_tuning_summary.md"
    lines = [
        "# Phase D5 entry micro-tuning summary",
        "",
        f"- anchor (0.57, 1.30): cost_on≈{cost_baseline:.4f}, MDD≈{mdd_baseline:.4f}, trades≈{trades_baseline}",
        f"- ADOPT: cost_on >= baseline+{COST_ON_IMPROVE_MIN}, trades >= baseline*{TRADES_FLOOR_ADOPT}, MDD <= baseline+{MDD_TOLERANCE}",
        "",
        "| run_id | min_max_proba | max_entropy | 30d cost_on | 365d cost_on | 365d MDD | trades |",
        "|--------|---------------|-------------|------------|-------------|----------|-------|",
    ]
    for r in all_rows:
        c30 = f"{r['c30']:.4f}" if r.get("c30") is not None else "N/A"
        c365 = f"{r['c365']:.4f}" if r.get("c365") is not None else "N/A"
        mdd = f"{r['mdd365']:.4f}" if r.get("mdd365") is not None else "N/A"
        tr = str(int(r["trades365"])) if r.get("trades365") is not None else "N/A"
        lines.append(f"| {r['run_id']} | {r['min_max_proba']:.3f} | {r['max_entropy']:.2f} | {c30} | {c365} | {mdd} | {tr} |")
    lines.extend([
        "",
        "## BEST spotcheck",
        f"- cost_on_range: {cost_on_range:.4f}",
        f"- MDD_range: {mdd_range:.4f}",
        f"- trades_range: {trades_range}",
        f"- spot stability: **{spot_stable}**",
        "",
        "## BEST & verdict",
        f"- BEST run_id: **{best_run_id}**",
        f"- BEST min_max_proba: **{best_p}**, max_entropy: **{best_e}**",
        f"- final verdict: **{verdict}**",
        "",
    ])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {out_md}")

    # 6) Export BEST JSON
    src_json = find_json_path(best_run_id)
    if src_json and src_json.exists():
        import shutil
        shutil.copy(src_json, BACKTESTS / "phase_d5_best_run.json")
        print(f"  Exported {BACKTESTS / 'phase_d5_best_run.json'}")

    # 7) Export BEST trade log
    print("  Exporting BEST trade log...")
    if run_validation("phase_d5_best_export", best_p, best_e, emit_trade_log=True, trade_log_path=str(BACKTESTS)) != 0:
        print("WARNING: Trade log export run failed.", file=sys.stderr)
    else:
        csv_src = BACKTESTS / "phase_d5_best_export_365d.csv"
        if csv_src.exists():
            with open(csv_src, encoding="utf-8") as f:
                reader = csv_module.DictReader(f)
                rows_csv = list(reader)
            if rows_csv:
                out_csv = BACKTESTS / "phase_d5_best_trades.csv"
                with open(out_csv, "w", encoding="utf-8", newline="") as f:
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
                print(f"  Exported {out_csv}")

    # 8) Final output
    print("")
    print("=" * 60)
    print("Phase D5 completed")
    print("=" * 60)
    print(f"1) report file path: {out_md}")
    print(f"2) BEST run_id: {best_run_id}")
    print(f"3) BEST parameters: min_max_proba={best_p}, max_entropy={best_e}")
    print(f"4) 365d cost_on: {best_row['c365']:.4f}" if best_row.get("c365") is not None else "4) 365d cost_on: N/A")
    print(f"5) 365d MDD: {best_row['mdd365']:.4f}" if best_row.get("mdd365") is not None else "5) 365d MDD: N/A")
    print(f"6) trades: {best_row['trades365']}" if best_row.get("trades365") is not None else "6) trades: N/A")
    print(f"7) spotcheck stability: {spot_stable}")
    print(f"8) final verdict: {verdict}")
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Phase D2: Narrow sweep around D1_v2 BEST (0.58, 1.25) to recover trades (target 620±50)
while keeping cost_on improvement. Baseline check, 9 runs, spotcheck BEST, verdict, export.
"""
from __future__ import annotations

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

# Baseline reference (Phase D1_v2)
BASELINE_COST_ON = -0.10259064874110835
BASELINE_TRADES = 883
BASELINE_MISMATCH_PCT = 0.05

END_DATE = "2026-03-03"
START_30 = "2026-02-01"
START_365 = "2025-03-02"

COMMON = [
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


def _env():
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    return env


def run_validation(
    min_max_proba: float,
    max_entropy: float,
    run_id: str,
    emit_trade_log: bool = False,
    trade_log_path: str | None = None,
) -> int:
    cmd = [
        sys.executable, "-X", "faulthandler", "-m", "scripts.run_tcn_candidate_validation",
        *COMMON,
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


def get_row(results: list, days: int) -> dict | None:
    for r in results:
        if r.get("days") == days:
            return r
    return None


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    BACKTESTS.mkdir(parents=True, exist_ok=True)

    # 1) Baseline run and validate
    print("Phase D2: Baseline run (phase_d2_baseline)...")
    if run_validation(0.55, 1.35, "phase_d2_baseline") != 0:
        print("ERROR: Baseline validation run failed.", file=sys.stderr)
        return 1
    d_base = load_json("phase_d2_baseline")
    if not d_base:
        print("ERROR: Baseline JSON not found.", file=sys.stderr)
        return 1
    row_365 = get_row(d_base.get("results", []), 365)
    if not row_365:
        print("ERROR: Baseline has no 365d result.", file=sys.stderr)
        return 1
    cost_base = row_365.get("cost_on_return")
    trades_base = row_365.get("trades")
    if cost_base is None or trades_base is None:
        print("ERROR: Baseline missing cost_on or trades.", file=sys.stderr)
        return 1
    if abs(cost_base - BASELINE_COST_ON) / abs(BASELINE_COST_ON) > BASELINE_MISMATCH_PCT:
        print(f"ERROR: Baseline cost_on drift > 5%: got {cost_base}, expected ~{BASELINE_COST_ON}. Possible data drift. STOP.", file=sys.stderr)
        return 1
    if abs(trades_base - BASELINE_TRADES) / BASELINE_TRADES > BASELINE_MISMATCH_PCT:
        print(f"ERROR: Baseline trades drift > 5%: got {trades_base}, expected ~{BASELINE_TRADES}. Possible data drift. STOP.", file=sys.stderr)
        return 1
    print(f"  Baseline OK: cost_on={cost_base:.4f}, trades={trades_base}")

    # 2) Grid 3×3
    probas = [0.56, 0.57, 0.58]
    entropies = [1.20, 1.25, 1.30]
    grid_run_ids = []
    for p in probas:
        for e in entropies:
            run_id = f"phase_d2_p{int(p*100):03d}_e{int(e*100):03d}"
            grid_run_ids.append((run_id, p, e))
    for run_id, p, e in grid_run_ids:
        print(f"  Run {run_id} (min_max_proba={p}, max_entropy={e})...")
        if run_validation(p, e, run_id) != 0:
            print(f"ERROR: {run_id} failed.", file=sys.stderr)
            return 1

    # 3) Collect metrics, build table, Top3, BEST
    trades_min = max(0, int(trades_base * 0.60))  # -40% threshold
    rows = []
    for run_id, p, e in grid_run_ids:
        d = load_json(run_id)
        if not d:
            print(f"WARNING: JSON not found for {run_id}", file=sys.stderr)
            continue
        res = d.get("results", [])
        r30 = get_row(res, 30)
        r365 = get_row(res, 365)
        rows.append({
            "run_id": run_id,
            "p": p, "e": e,
            "c30": r30.get("cost_on_return") if r30 else None,
            "c365": r365.get("cost_on_return") if r365 else None,
            "mdd365": r365.get("max_drawdown") if r365 else None,
            "trades365": r365.get("trades") if r365 else None,
        })

    # Top3 by 365d cost_on (descending = less negative first)
    rows_sorted = sorted([r for r in rows if r.get("c365") is not None], key=lambda x: x["c365"], reverse=True)
    top3 = rows_sorted[:3]
    # BEST = top by cost_on that meets trades >= baseline*0.60
    best_row = None
    for r in rows_sorted:
        if (r.get("trades365") or 0) >= trades_min:
            best_row = r
            break
    if not best_row:
        best_row = rows_sorted[0] if rows_sorted else None
    best_run_id = best_row["run_id"] if best_row else "phase_d2_baseline"
    best_p = best_row["p"] if best_row else 0.58
    best_e = best_row["e"] if best_row else 1.25

    # 4) Sweep summary MD
    sweep_md = REPORTS / "phase_d2_sweep_summary.md"
    lines = [
        "# Phase D2 sweep summary",
        "",
        f"- pinned end_date: {END_DATE}",
        f"- baseline: phase_d2_baseline (365d cost_on={cost_base:.4f}, trades={trades_base})",
        "",
        "| run_id | 30d cost_on | 365d cost_on | 365d MDD | trades |",
        "|--------|------------|-------------|----------|-------|",
    ]
    for r in rows:
        c30 = f"{r['c30']:.4f}" if r.get("c30") is not None else "N/A"
        c365 = f"{r['c365']:.4f}" if r.get("c365") is not None else "N/A"
        mdd = f"{r['mdd365']:.4f}" if r.get("mdd365") is not None else "N/A"
        tr = str(int(r["trades365"])) if r.get("trades365") is not None else "N/A"
        lines.append(f"| {r['run_id']} | {c30} | {c365} | {mdd} | {tr} |")
    lines.extend(["", "## Top3 (365d cost_on)", ""])
    for i, r in enumerate(top3, 1):
        lines.append(f"{i}. {r['run_id']} — cost_on={r['c365']:.4f}, MDD={r['mdd365']}, trades={r['trades365']}")
    lines.append("")
    sweep_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {sweep_md}")

    # 5) Spotcheck: run BEST 2 more times (total 3)
    print(f"  Spotcheck BEST {best_run_id} (2 more runs)...")
    if run_validation(best_p, best_e, "phase_d2_best_spot1") != 0:
        print("ERROR: phase_d2_best_spot1 failed.", file=sys.stderr)
        return 1
    if run_validation(best_p, best_e, "phase_d2_best_spot2") != 0:
        print("ERROR: phase_d2_best_spot2 failed.", file=sys.stderr)
        return 1

    # Collect 3 runs: best (from grid) + spot1 + spot2
    spot_ids = [best_run_id, "phase_d2_best_spot1", "phase_d2_best_spot2"]
    spot_rows = []
    for rid in spot_ids:
        d = load_json(rid)
        if not d:
            continue
        r365 = get_row(d.get("results", []), 365)
        if r365:
            spot_rows.append({
                "run_id": rid,
                "c365": r365.get("cost_on_return"),
                "mdd": r365.get("max_drawdown"),
                "trades": r365.get("trades"),
            })

    cost_365_vals = [x["c365"] for x in spot_rows if x.get("c365") is not None]
    mdd_vals = [x["mdd"] for x in spot_rows if x.get("mdd") is not None]
    trades_vals = [x["trades"] for x in spot_rows if x.get("trades") is not None]
    cost_on_range = (max(cost_365_vals) - min(cost_365_vals)) if len(cost_365_vals) >= 2 else 0.0
    mdd_range = (max(mdd_vals) - min(mdd_vals)) if len(mdd_vals) >= 2 else 0.0
    trades_range = (max(trades_vals) - min(trades_vals)) if len(trades_vals) >= 2 else 0

    # 6) Spotcheck summary MD
    spot_md = REPORTS / "phase_d2_spotcheck_summary.md"
    spot_lines = [
        "# Phase D2 spotcheck summary",
        "",
        f"BEST run_id: **{best_run_id}** (min_max_proba={best_p}, max_entropy={best_e})",
        "",
        "| run_id | 365d cost_on | 365d MDD | trades |",
        "|--------|-------------|----------|-------|",
    ]
    for r in spot_rows:
        c = f"{r['c365']:.4f}" if r.get("c365") is not None else "N/A"
        m = f"{r['mdd']:.4f}" if r.get("mdd") is not None else "N/A"
        t = str(int(r["trades"])) if r.get("trades") is not None else "N/A"
        spot_lines.append(f"| {r['run_id']} | {c} | {m} | {t} |")
    spot_lines.extend([
        "",
        "## Ranges (3 runs)",
        f"- cost_on_range: {cost_on_range:.4f}",
        f"- MDD_range: {mdd_range:.4f}",
        f"- trades_range: {trades_range}",
        "",
    ])
    spot_md.write_text("\n".join(spot_lines), encoding="utf-8")
    print(f"  Wrote {spot_md}")

    # 7) Verdict
    best_c365 = best_row["c365"] if best_row else None
    best_trades = best_row["trades365"] if best_row else None
    cost_improved = (best_c365 is not None and best_c365 > cost_base)
    trades_ok = (best_trades is not None and best_trades >= trades_min)
    range_ok = cost_on_range <= 0.002
    if cost_improved and trades_ok and range_ok:
        verdict = "ADOPT"
    else:
        verdict = "REJECT"

    # 8) Export BEST JSON and trade log
    best_src = None
    for f in DIAG.glob(f"{PREFIX}_{best_run_id}.json"):
        best_src = f
        break
    if best_src and best_src.exists():
        import shutil
        shutil.copy(best_src, BACKTESTS / "phase_d2_best_run.json")
        print(f"  Exported {BACKTESTS / 'phase_d2_best_run.json'}")

    # Trade log: run BEST once with emit_trade_log to get CSV
    print("  Exporting BEST trade log...")
    if run_validation(best_p, best_e, "phase_d2_best_export", emit_trade_log=True, trade_log_path=str(BACKTESTS)) != 0:
        print("WARNING: Trade log export run failed.", file=sys.stderr)
    else:
        # Find phase_d2_best_export_365d.csv in BACKTESTS (written per span: 30d.csv, 365d.csv)
        csv_src = BACKTESTS / "phase_d2_best_export_365d.csv"
        if not csv_src.exists():
            csv_src = BACKTESTS / "phase_d2_best_export_365.csv"
        if csv_src.exists():
            import csv as csv_module
            from collections import Counter
            with open(csv_src, encoding="utf-8") as f:
                reader = csv_module.DictReader(f)
                rows_csv = list(reader)
            if rows_csv:
                out_csv = BACKTESTS / "phase_d2_best_trades.csv"
                with open(out_csv, "w", encoding="utf-8", newline="") as f:
                    w = csv_module.DictWriter(f, fieldnames=["entry_time", "exit_time", "pnl", "exit_reason", "holding_bars"])
                    w.writeheader()
                    for r in rows_csv:
                        w.writerow({
                            "entry_time": r.get("entry_ts", ""),
                            "exit_time": r.get("exit_ts", ""),
                            "pnl": r.get("net_return", ""),
                            "exit_reason": r.get("exit_reason", ""),
                            "holding_bars": r.get("holding_bars", ""),
                        })
                print(f"  Exported {out_csv}")
                # Optional: distribution stats for spotcheck summary
                holds = []
                for r in rows_csv:
                    h = r.get("holding_bars")
                    if h != "" and h is not None:
                        try:
                            holds.append(int(float(h)))
                        except (ValueError, TypeError):
                            pass
                if holds:
                    mean_hold = sum(holds) / len(holds)
                    holds_sorted = sorted(holds)
                    median_hold = holds_sorted[len(holds_sorted) // 2] if holds_sorted else 0
                    reasons = Counter(r.get("exit_reason", "") for r in rows_csv)
                    dist_lines = [
                        "", "## BEST run distribution (365d trade log)",
                        f"- mean_hold: {mean_hold:.1f}",
                        f"- median_hold: {median_hold}",
                        "- exit_reason breakdown:",
                    ]
                    for reason, count in reasons.most_common():
                        dist_lines.append(f"  - {reason}: {count}")
                    spot_md.write_text(spot_md.read_text(encoding="utf-8") + "\n".join(dist_lines) + "\n", encoding="utf-8")

    # 9) Final console output
    print("")
    print("=" * 60)
    print("Phase D2 completed")
    print("=" * 60)
    print(f"BEST run_id:        {best_run_id}")
    print(f"365d cost_on:      {best_c365:.4f}" if best_c365 is not None else "365d cost_on:      N/A")
    print(f"365d MDD:          {best_row['mdd365']:.4f}" if best_row and best_row.get("mdd365") is not None else "365d MDD:          N/A")
    print(f"trades:            {best_trades}" if best_trades is not None else "trades:            N/A")
    print(f"spotcheck cost_on_range: {cost_on_range:.4f} (<=0.002 required: {'OK' if range_ok else 'FAIL'})")
    print(f"final verdict:     {verdict}")
    print("")
    print("Summary files generated:")
    print(f"  {sweep_md}")
    print(f"  {spot_md}")
    print("Best configuration candidate identified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

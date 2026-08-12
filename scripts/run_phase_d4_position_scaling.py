#!/usr/bin/env python3
"""
Phase D4: Position scaling experiment. Anchor = Phase D2/D3 (p057_e130, regime off, scaling off).
Test: baseline (off) + 4 linear scaling variants (floor/full/size_min/size_max).
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
COST_ON_IMPROVE_MIN = 0.003   # adopt if cost_on >= baseline + this
MDD_TOLERANCE = 0.005         # adopt if MDD <= baseline + this
TRADES_FLOOR_RATIO = 0.95    # adopt if trades >= baseline * this
SPOT_STABLE_COST_RANGE = 0.003  # STABLE if cost_on_range <= this

END_DATE = "2026-03-03"
START_30 = "2026-02-01"
START_365 = "2025-03-02"

COMMON_BASE = [
    "--id", "h15_t0p004", "--symbol", "BTCUSDT", "--timeframe", "5m",
    "--min-max-proba", "0.57", "--max-entropy", "1.30",
    "--min-hold", "36", "--cooldown", "12",
    "--commission", "0.0009", "--slippage", "0.0001",
    "--regime-filter", "off",
    "--time-stop", "on", "--time-stop-bars", "72",
    "--early-exit", "on", "--early-exit-lookback", "12",
    "--early-exit-p-floor", "0.55", "--early-exit-bad-k", "8",
    "--partial-tp", "off", "--break-even-stop", "off",
    "--end-date", END_DATE, "--start-date-30", START_30, "--start-date-365", START_365,
    "--days-list", "30,365",
]

# (run_id, position_scaling, p_floor, p_full, size_min, size_max)
RUNS = [
    ("phase_d4_baseline", "off", None, None, None, None),
    ("phase_d4_A", "linear", 0.57, 0.67, 0.30, 1.00),
    ("phase_d4_B", "linear", 0.57, 0.65, 0.25, 1.00),
    ("phase_d4_C", "linear", 0.58, 0.65, 0.25, 1.00),
    ("phase_d4_D", "linear", 0.58, 0.63, 0.20, 1.00),
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
    position_scaling: str,
    p_floor: float | None = None,
    p_full: float | None = None,
    size_min: float | None = None,
    size_max: float | None = None,
    emit_trade_log: bool = False,
    trade_log_path: str | None = None,
) -> int:
    cmd = [
        sys.executable, "-X", "faulthandler", "-m", "scripts.run_tcn_candidate_validation",
        *COMMON_BASE,
        "--position-scaling", position_scaling,
        "--run-id", run_id,
    ]
    if position_scaling != "off":
        if p_floor is not None:
            cmd.extend(["--position-p-floor", str(p_floor)])
        if p_full is not None:
            cmd.extend(["--position-p-full", str(p_full)])
        if size_min is not None:
            cmd.extend(["--position-size-min", str(size_min)])
        if size_max is not None:
            cmd.extend(["--position-size-max", str(size_max)])
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

    # 1) Run baseline + 4 variants
    for run_id, scaling, p_floor, p_full, size_min, size_max in RUNS:
        print(f"  Run {run_id} (position_scaling={scaling}" + (f" floor={p_floor} full={p_full} min={size_min} max={size_max}" if scaling != "off" else "") + ")...")
        if run_validation(run_id, scaling, p_floor, p_full, size_min, size_max) != 0:
            print(f"ERROR: {run_id} failed.", file=sys.stderr)
            return 1

    # 2) Load results and validate baseline
    all_rows = []
    for run_id, scaling, _pf, _pfull, _smin, _smax in RUNS:
        d = load_json(run_id)
        if not d:
            print(f"WARNING: JSON not found for {run_id}", file=sys.stderr)
            continue
        res = d.get("results", [])
        r30 = get_row(res, 30)
        r365 = get_row(res, 365)
        mean_pos = r365.get("scale_mean") if r365 else None
        applied = r365.get("entries_scaled_applied_count") if r365 else None
        trades365 = r365.get("trades") if r365 else None
        scaled_ratio = (applied / trades365) if (trades365 and applied is not None and trades365 > 0) else None
        row = {
            "run_id": run_id,
            "c30": r30.get("cost_on_return") if r30 else None,
            "c365": r365.get("cost_on_return") if r365 else None,
            "mdd365": r365.get("max_drawdown") if r365 else None,
            "trades365": trades365,
            "mean_position_size": mean_pos,
            "scaled_trade_ratio": scaled_ratio,
            "scale_bins": r365.get("scale_bins") if r365 else None,
        }
        all_rows.append(row)
        if run_id == "phase_d4_baseline":
            cost_b = row["c365"]
            trades_b = row["trades365"]
            if cost_b is None or trades_b is None:
                print("ERROR: Baseline missing cost_on or trades.", file=sys.stderr)
                return 1
            if abs(cost_b - BASELINE_COST_ON) / abs(BASELINE_COST_ON) > BASELINE_MISMATCH_PCT:
                print(f"ERROR: Baseline cost_on drift > 5%: got {cost_b}, expected ~{BASELINE_COST_ON}. STOP.", file=sys.stderr)
                return 1
            if abs(trades_b - BASELINE_TRADES) / BASELINE_TRADES > BASELINE_MISMATCH_PCT:
                print(f"ERROR: Baseline trades drift > 5%: got {trades_b}, expected ~{BASELINE_TRADES}. STOP.", file=sys.stderr)
                return 1
            print(f"  Baseline OK: cost_on={cost_b:.4f}, MDD={row['mdd365']}, trades={trades_b}")

    baseline_row = next((r for r in all_rows if r["run_id"] == "phase_d4_baseline"), None)
    cost_baseline = baseline_row["c365"] if baseline_row else BASELINE_COST_ON
    mdd_baseline = baseline_row["mdd365"] if baseline_row else BASELINE_MDD
    trades_baseline = baseline_row["trades365"] if baseline_row else BASELINE_TRADES

    # 3) Verdict: REJECT / NO_IMPROVE / ADOPT_CANDIDATE; BEST = max cost_on among non-REJECT
    def verdict_for(r):
        c = r.get("c365")
        mdd = r.get("mdd365")
        tr = r.get("trades365")
        if c is None or mdd is None or tr is None:
            return "REJECT"
        if c <= cost_baseline - COST_ON_IMPROVE_MIN or mdd >= mdd_baseline + MDD_TOLERANCE:
            return "REJECT"
        if tr < trades_baseline * TRADES_FLOOR_RATIO:
            return "REJECT"
        if c >= cost_baseline + COST_ON_IMPROVE_MIN:
            return "ADOPT_CANDIDATE"
        return "NO_IMPROVE"

    non_reject = [r for r in all_rows if verdict_for(r) != "REJECT"]
    if non_reject:
        best_row = max(non_reject, key=lambda x: (x["c365"] or -1e9))
    else:
        best_row = max([r for r in all_rows if r.get("c365") is not None], key=lambda x: x["c365"])
    best_run_id = best_row["run_id"]
    verdict = verdict_for(best_row)
    run_params = next(r for r in RUNS if r[0] == best_run_id)
    best_scaling, best_pf, best_pfull, best_smin, best_smax = run_params[1], run_params[2], run_params[3], run_params[4], run_params[5]

    # 4) Spotcheck BEST (2 runs)
    print(f"  Spotcheck BEST {best_run_id} (2 runs)...")
    if run_validation("phase_d4_best_spot1", best_scaling, best_pf, best_pfull, best_smin, best_smax) != 0:
        print("ERROR: phase_d4_best_spot1 failed.", file=sys.stderr)
        return 1
    if run_validation("phase_d4_best_spot2", best_scaling, best_pf, best_pfull, best_smin, best_smax) != 0:
        print("ERROR: phase_d4_best_spot2 failed.", file=sys.stderr)
        return 1
    d0 = load_json(best_run_id)
    spot1 = load_json("phase_d4_best_spot1")
    spot2 = load_json("phase_d4_best_spot2")
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
    out_md = REPORTS / "phase_d4_position_scaling_summary.md"
    lines = [
        "# Phase D4 position scaling summary",
        "",
        f"- anchor: p057_e130, regime off. baseline cost_on≈{cost_baseline:.4f}, MDD≈{mdd_baseline:.4f}, trades≈{trades_baseline}",
        f"- adopt: cost_on >= baseline+{COST_ON_IMPROVE_MIN}, MDD <= baseline+{MDD_TOLERANCE}, trades >= baseline*{TRADES_FLOOR_RATIO}",
        "",
        "| run_id | 30d cost_on | 365d cost_on | 365d MDD | trades | mean_position_size | scaled_trade_ratio |",
        "|--------|------------|-------------|----------|-------|-------------------|-------------------|",
    ]
    for r in all_rows:
        c30 = f"{r['c30']:.4f}" if r.get("c30") is not None else "N/A"
        c365 = f"{r['c365']:.4f}" if r.get("c365") is not None else "N/A"
        mdd = f"{r['mdd365']:.4f}" if r.get("mdd365") is not None else "N/A"
        tr = str(int(r["trades365"])) if r.get("trades365") is not None else "N/A"
        mps = f"{r['mean_position_size']:.4f}" if r.get("mean_position_size") is not None else "N/A"
        str_ = f"{r['scaled_trade_ratio']:.4f}" if r.get("scaled_trade_ratio") is not None else "N/A"
        lines.append(f"| {r['run_id']} | {c30} | {c365} | {mdd} | {tr} | {mps} | {str_} |")
    lines.append("")
    lines.append("## Scaling stats (detail)")
    lines.append("- median_position_size: N/A (engine does not expose scale_median in validation result)")
    lines.append("- size_bin_counts (365d, scaling runs only):")
    for r in all_rows:
        if r.get("scale_bins"):
            lines.append(f"  - {r['run_id']}: {r['scale_bins']}")
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
        f"- final verdict: **{verdict}**",
        "",
    ])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {out_md}")

    # 6) Export BEST JSON
    src_json = find_json_path(best_run_id)
    if src_json and src_json.exists():
        import shutil
        shutil.copy(src_json, BACKTESTS / "phase_d4_best_run.json")
        print(f"  Exported {BACKTESTS / 'phase_d4_best_run.json'}")

    # 7) Export BEST trade log (with position_size)
    print("  Exporting BEST trade log...")
    if run_validation("phase_d4_best_export", best_scaling, best_pf, best_pfull, best_smin, best_smax, emit_trade_log=True, trade_log_path=str(BACKTESTS)) != 0:
        print("WARNING: Trade log export run failed.", file=sys.stderr)
    else:
        csv_src = BACKTESTS / "phase_d4_best_export_365d.csv"
        if csv_src.exists():
            with open(csv_src, encoding="utf-8") as f:
                reader = csv_module.DictReader(f)
                rows_csv = list(reader)
            if rows_csv:
                out_csv = BACKTESTS / "phase_d4_best_trades.csv"
                with open(out_csv, "w", encoding="utf-8", newline="") as f:
                    w = csv_module.DictWriter(f, fieldnames=["entry_time", "exit_time", "position_size", "pnl", "exit_reason", "holding_bars"])
                    w.writeheader()
                    for r in rows_csv:
                        w.writerow({
                            "entry_time": r.get("entry_ts", r.get("entry_time", "")),
                            "exit_time": r.get("exit_ts", r.get("exit_time", "")),
                            "position_size": r.get("position_scale", r.get("position_size", "")),
                            "pnl": r.get("net_return", r.get("pnl", "")),
                            "exit_reason": r.get("exit_reason", ""),
                            "holding_bars": r.get("holding_bars", ""),
                        })
                print(f"  Exported {out_csv}")

    # 8) Final report
    mean_pos_best = best_row.get("mean_position_size")
    str_best = best_row.get("scaled_trade_ratio")
    print("")
    print("=" * 60)
    print("Phase D4 completed")
    print("=" * 60)
    print("1) Files created: run_phase_d4_position_scaling.py (script), phase_d4_position_scaling_summary.md, phase_d4_best_run.json, phase_d4_best_trades.csv")
    print(f"2) Summary path: {out_md}")
    print(f"3) BEST run_id: {best_run_id}")
    print(f"4) BEST 365d cost_on: {best_row['c365']:.4f}" if best_row.get("c365") is not None else "4) BEST 365d cost_on: N/A")
    print(f"5) BEST 365d MDD: {best_row['mdd365']:.4f}" if best_row.get("mdd365") is not None else "5) BEST 365d MDD: N/A")
    print(f"6) trades: {best_row['trades365']}" if best_row.get("trades365") is not None else "6) trades: N/A")
    print(f"7) mean_position_size: {mean_pos_best:.4f}" if mean_pos_best is not None else "7) mean_position_size: N/A")
    print(f"8) scaled_trade_ratio: {str_best:.4f}" if str_best is not None else "8) scaled_trade_ratio: N/A")
    print(f"9) Spotcheck stability: {spot_stable}")
    print(f"10) Final verdict: {verdict}")
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())

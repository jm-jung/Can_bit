#!/usr/bin/env python3
"""
Phase D2.1: Stability test for three nearby configs (p056_e125, p056_e130, p057_e130).
Each config: original + spot1 + spot2 (9 runs total). Compute ranges, verdict, export BEST.
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

BASELINE_COST_ON = -0.1026
BASELINE_TRADES = 883
BASELINE_MISMATCH_PCT = 0.05
TRADES_MIN = 530  # baseline * 0.60
COST_ON_RANGE_MAX = 0.01  # stability threshold

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

# Three configs: (run_id_prefix, min_max_proba, max_entropy)
CONFIGS = [
    ("phase_d21_p056_e125", 0.56, 1.25),  # A
    ("phase_d21_p056_e130", 0.56, 1.30),  # B
    ("phase_d21_p057_e130", 0.57, 1.30),  # C
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

    # Baseline: use existing phase_d2_baseline or run once
    print("Phase D2.1: Baseline check...")
    d_base = load_json("phase_d2_baseline")
    if not d_base:
        print("  Running phase_d2_baseline...")
        if run_validation(0.55, 1.35, "phase_d2_baseline") != 0:
            print("ERROR: Baseline run failed.", file=sys.stderr)
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
        print(f"ERROR: Baseline cost_on drift > 5%: got {cost_base}, expected ~{BASELINE_COST_ON}. STOP.", file=sys.stderr)
        return 1
    if abs(trades_base - BASELINE_TRADES) / BASELINE_TRADES > BASELINE_MISMATCH_PCT:
        print(f"ERROR: Baseline trades drift > 5%: got {trades_base}, expected ~{BASELINE_TRADES}. STOP.", file=sys.stderr)
        return 1
    print(f"  Baseline OK: cost_on={cost_base:.4f}, trades={trades_base}")

    # Run 3 configs × (original, spot1, spot2) = 9 runs
    all_runs = []
    for prefix, p, e in CONFIGS:
        for suffix in ["", "_spot1", "_spot2"]:
            run_id = prefix + suffix
            print(f"  Run {run_id} (p={p}, e={e})...")
            if run_validation(p, e, run_id) != 0:
                print(f"ERROR: {run_id} failed.", file=sys.stderr)
                return 1
            d = load_json(run_id)
            if not d:
                print(f"WARNING: JSON not found for {run_id}", file=sys.stderr)
                continue
            res = d.get("results", [])
            r30 = get_row(res, 30)
            r365 = get_row(res, 365)
            all_runs.append({
                "run_id": run_id,
                "config": prefix,
                "p": p, "e": e,
                "c30": r30.get("cost_on_return") if r30 else None,
                "c365": r365.get("cost_on_return") if r365 else None,
                "mdd365": r365.get("max_drawdown") if r365 else None,
                "trades365": r365.get("trades") if r365 else None,
            })

    # Per-config stability (original + spot1 + spot2)
    config_summary = []
    for prefix, p, e in CONFIGS:
        runs = [r for r in all_runs if r["config"] == prefix]
        c_vals = [x["c365"] for x in runs if x.get("c365") is not None]
        mdd_vals = [x["mdd365"] for x in runs if x.get("mdd365") is not None]
        t_vals = [x["trades365"] for x in runs if x.get("trades365") is not None]
        cost_on_range = (max(c_vals) - min(c_vals)) if len(c_vals) >= 2 else 0.0
        mdd_range = (max(mdd_vals) - min(mdd_vals)) if len(mdd_vals) >= 2 else 0.0
        trades_range = (max(t_vals) - min(t_vals)) if len(t_vals) >= 2 else 0
        mean_c = sum(c_vals) / len(c_vals) if c_vals else None
        mean_t = sum(t_vals) / len(t_vals) if t_vals else None
        config_summary.append({
            "config": prefix,
            "p": p, "e": e,
            "runs": runs,
            "cost_on_range": cost_on_range,
            "mdd_range": mdd_range,
            "trades_range": trades_range,
            "mean_c365": mean_c,
            "mean_trades": int(mean_t) if mean_t is not None else None,
            "cost_improved": mean_c is not None and mean_c > cost_base,
            "trades_ok": mean_t is not None and mean_t >= TRADES_MIN,
            "range_ok": cost_on_range <= COST_ON_RANGE_MAX,
        })

    # Verdict per config: ADOPT_CANDIDATE if cost_improved and trades_ok and range_ok
    for cs in config_summary:
        cs["verdict"] = "ADOPT_CANDIDATE" if (cs["cost_improved"] and cs["trades_ok"] and cs["range_ok"]) else "REJECT"

    # BEST = ADOPT_CANDIDATE with max mean_c365; else max mean_c365 overall
    candidates = [c for c in config_summary if c["verdict"] == "ADOPT_CANDIDATE"]
    if candidates:
        best_config = max(candidates, key=lambda x: x["mean_c365"] or -1e9)
    else:
        best_config = max(config_summary, key=lambda x: x["mean_c365"] or -1e9)
    best_run_id = best_config["config"]  # original run_id for this config
    best_p, best_e = best_config["p"], best_config["e"]

    # Build report MD
    out_md = REPORTS / "phase_d2_1_stability_summary.md"
    lines = [
        "# Phase D2.1 stability summary",
        "",
        f"- baseline: cost_on={cost_base:.4f}, trades={trades_base}",
        f"- TRADES_MIN={TRADES_MIN}, cost_on_range threshold={COST_ON_RANGE_MAX}",
        "",
        "## All runs (run_id | cost_on | MDD | trades)",
        "",
        "| run_id | 30d cost_on | 365d cost_on | 365d MDD | trades |",
        "|--------|------------|-------------|----------|-------|",
    ]
    for r in all_runs:
        c30 = f"{r['c30']:.4f}" if r.get("c30") is not None else "N/A"
        c365 = f"{r['c365']:.4f}" if r.get("c365") is not None else "N/A"
        mdd = f"{r['mdd365']:.4f}" if r.get("mdd365") is not None else "N/A"
        tr = str(int(r["trades365"])) if r.get("trades365") is not None else "N/A"
        lines.append(f"| {r['run_id']} | {c30} | {c365} | {mdd} | {tr} |")
    lines.append("")
    lines.append("## Per-config stability (3 runs each)")
    lines.append("")
    for cs in config_summary:
        lines.append(f"### {cs['config']} (p={cs['p']}, e={cs['e']})")
        lines.append(f"- cost_on_range: {cs['cost_on_range']:.4f}")
        lines.append(f"- MDD_range: {cs['mdd_range']:.4f}")
        lines.append(f"- trades_range: {cs['trades_range']}")
        lines.append(f"- mean 365d cost_on: {cs['mean_c365']:.4f}" if cs["mean_c365"] is not None else "- mean 365d cost_on: N/A")
        lines.append(f"- verdict: {cs['verdict']}")
        lines.append("")
    lines.append("## BEST")
    lines.append(f"- BEST run_id: **{best_run_id}**")
    lines.append(f"- final verdict: **{best_config['verdict']}**")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {out_md}")

    # Export BEST run JSON (use original run of best config)
    src_json = find_json_path(best_run_id)
    if src_json and src_json.exists():
        import shutil
        shutil.copy(src_json, BACKTESTS / "phase_d2_1_best_run.json")
        print(f"  Exported {BACKTESTS / 'phase_d2_1_best_run.json'}")

    # Export BEST trade log: run once with emit_trade_log
    print("  Exporting BEST trade log...")
    if run_validation(best_p, best_e, "phase_d21_best_export", emit_trade_log=True, trade_log_path=str(BACKTESTS)) != 0:
        print("WARNING: Trade log export run failed.", file=sys.stderr)
    else:
        csv_src = BACKTESTS / "phase_d21_best_export_365d.csv"
        if csv_src.exists():
            with open(csv_src, encoding="utf-8") as f:
                reader = csv_module.DictReader(f)
                rows_csv = list(reader)
            if rows_csv:
                out_csv = BACKTESTS / "phase_d2_1_best_trades.csv"
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

    # Console output
    mean_c = best_config["mean_c365"]
    cost_on_range = best_config["cost_on_range"]
    print("")
    print("=" * 60)
    print("Phase D2.1 completed")
    print("=" * 60)
    print(f"BEST run_id:        {best_run_id}")
    print(f"365d cost_on:      {mean_c:.4f}" if mean_c is not None else "365d cost_on:      N/A")
    print(f"365d MDD:          (see report; 3-run mean)")
    print(f"trades:            {best_config['mean_trades']}" if best_config.get("mean_trades") is not None else "trades:            N/A")
    print(f"cost_on_range:     {cost_on_range:.4f} (<=0.01 required: {'OK' if best_config['range_ok'] else 'FAIL'})")
    print(f"final verdict:     {best_config['verdict']}")
    print("")
    print("Summary: " + str(out_md))
    print("BEST configuration: " + best_run_id + f" (min_max_proba={best_p}, max_entropy={best_e})")
    print("Next recommended step: " + ("Adopt this config for ops if approved." if best_config["verdict"] == "ADOPT_CANDIDATE" else "Consider further tuning or keep D1_v2 BEST."))
    return 0


if __name__ == "__main__":
    sys.exit(main())

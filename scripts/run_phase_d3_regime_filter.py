#!/usr/bin/env python3
"""
Phase D3: Regime filter experiment. Anchor = Phase D2.1 BEST (p057_e130).
Test: off, ema200, ema200_slope, vol_filter_0008, vol_filter_0010.

Run: python scripts/run_phase_d3_regime_filter.py
(If segfault in sandbox, run with full permissions; each validation run ~25-30 min.)
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

# Anchor from Phase D2.1
BASELINE_COST_ON = -0.0651
BASELINE_MDD = 0.105
BASELINE_TRADES = 707
BASELINE_MISMATCH_PCT = 0.05
COST_ON_FLOOR = BASELINE_COST_ON - 0.01   # adopt if cost_on >= this
MDD_CEILING = BASELINE_MDD * 0.85          # adopt if MDD <= this (15% reduction)

END_DATE = "2026-03-03"
START_30 = "2026-02-01"
START_365 = "2025-03-02"

# Anchor params (regime_filter overridden per run)
COMMON_BASE = [
    "--id", "h15_t0p004", "--symbol", "BTCUSDT", "--timeframe", "5m",
    "--min-max-proba", "0.57", "--max-entropy", "1.30",
    "--min-hold", "36", "--cooldown", "12",
    "--commission", "0.0009", "--slippage", "0.0001",
    "--position-scaling", "off",
    "--time-stop", "on", "--time-stop-bars", "72",
    "--early-exit", "on", "--early-exit-lookback", "12",
    "--early-exit-p-floor", "0.55", "--early-exit-bad-k", "8",
    "--partial-tp", "off", "--break-even-stop", "off",
    "--end-date", END_DATE, "--start-date-30", START_30, "--start-date-365", START_365,
    "--days-list", "30,365",
]

# (run_id, regime_filter, vol_threshold or None)
RUNS = [
    ("phase_d3_baseline", "off", None),
    ("phase_d3_ema200", "ema_only", None),
    ("phase_d3_ema200_slope", "ema_plus_slope", None),
    ("phase_d3_vol_filter_0008", "vol_compress", 0.0008),
    ("phase_d3_vol_filter_0010", "vol_compress", 0.0010),
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
    regime_filter: str,
    vol_threshold: float | None = None,
    emit_trade_log: bool = False,
    trade_log_path: str | None = None,
) -> int:
    cmd = [
        sys.executable, "-X", "faulthandler", "-m", "scripts.run_tcn_candidate_validation",
        *COMMON_BASE,
        "--regime-filter", regime_filter,
        "--run-id", run_id,
    ]
    if vol_threshold is not None:
        cmd.extend(["--vol-threshold", str(vol_threshold)])
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

    # 1) Run all 5 configs
    for run_id, regime_filter, vol_th in RUNS:
        print(f"  Run {run_id} (regime={regime_filter}" + (f", vol_threshold={vol_th}" if vol_th is not None else "") + ")...")
        if run_validation(run_id, regime_filter, vol_th) != 0:
            print(f"ERROR: {run_id} failed.", file=sys.stderr)
            return 1

    # 2) Load results and validate baseline
    all_rows = []
    for run_id, regime_filter, vol_th in RUNS:
        d = load_json(run_id)
        if not d:
            print(f"WARNING: JSON not found for {run_id}", file=sys.stderr)
            continue
        res = d.get("results", [])
        r30 = get_row(res, 30)
        r365 = get_row(res, 365)
        row = {
            "run_id": run_id,
            "regime": regime_filter,
            "c30": r30.get("cost_on_return") if r30 else None,
            "c365": r365.get("cost_on_return") if r365 else None,
            "mdd365": r365.get("max_drawdown") if r365 else None,
            "trades365": r365.get("trades") if r365 else None,
        }
        all_rows.append(row)
        if run_id == "phase_d3_baseline":
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

    # 3) Verdict: adopt if cost_on >= baseline-0.01 AND MDD <= baseline*0.85
    baseline_row = next((r for r in all_rows if r["run_id"] == "phase_d3_baseline"), None)
    cost_baseline = baseline_row["c365"] if baseline_row else BASELINE_COST_ON
    mdd_baseline = baseline_row["mdd365"] if baseline_row else BASELINE_MDD
    adopt_candidates = [
        r for r in all_rows
        if r.get("c365") is not None and r.get("mdd365") is not None
        and r["c365"] >= COST_ON_FLOOR and r["mdd365"] <= MDD_CEILING
    ]
    # BEST = adopt candidate with lowest MDD; if none, best cost_on
    if adopt_candidates:
        best_row = min(adopt_candidates, key=lambda x: (x["mdd365"], -x["c365"]))
    else:
        best_row = max([r for r in all_rows if r.get("c365") is not None], key=lambda x: x["c365"])
    best_run_id = best_row["run_id"]
    best_regime = next(r[1] for r in RUNS if r[0] == best_run_id)
    best_vol_th = next((r[2] for r in RUNS if r[0] == best_run_id), None)

    verdict = "ADOPT_CANDIDATE" if best_row in adopt_candidates else "REJECT"

    # 4) Spotcheck for BEST: 2 more runs
    print(f"  Spotcheck BEST {best_run_id} (2 runs)...")
    if run_validation("phase_d3_best_spot1", best_regime, best_vol_th) != 0:
        print("ERROR: phase_d3_best_spot1 failed.", file=sys.stderr)
        return 1
    if run_validation("phase_d3_best_spot2", best_regime, best_vol_th) != 0:
        print("ERROR: phase_d3_best_spot2 failed.", file=sys.stderr)
        return 1
    d0 = load_json(best_run_id)
    spot1 = load_json("phase_d3_best_spot1")
    spot2 = load_json("phase_d3_best_spot2")
    r365_0 = get_row(d0.get("results", []), 365) if d0 else None
    r365_1 = get_row(spot1.get("results", []), 365) if spot1 else None
    r365_2 = get_row(spot2.get("results", []), 365) if spot2 else None
    c_vals = [r365_0.get("cost_on_return"), r365_1.get("cost_on_return") if r365_1 else None, r365_2.get("cost_on_return") if r365_2 else None]
    c_vals = [x for x in c_vals if x is not None]
    mdd_vals = [r365_0.get("max_drawdown"), r365_1.get("max_drawdown") if r365_1 else None, r365_2.get("max_drawdown") if r365_2 else None]
    mdd_vals = [x for x in mdd_vals if x is not None]
    tr_vals = [r365_0.get("trades"), r365_1.get("trades") if r365_1 else None, r365_2.get("trades") if r365_2 else None]
    tr_vals = [x for x in tr_vals if x is not None]
    cost_on_range = (max(c_vals) - min(c_vals)) if len(c_vals) >= 2 else 0.0
    mdd_range = (max(mdd_vals) - min(mdd_vals)) if len(mdd_vals) >= 2 else 0.0
    trades_range = (max(tr_vals) - min(tr_vals)) if len(tr_vals) >= 2 else 0

    # 5) Report MD
    out_md = REPORTS / "phase_d3_regime_filter_summary.md"
    lines = [
        "# Phase D3 regime filter summary",
        "",
        f"- anchor: p057_e130, baseline (off) cost_on≈{cost_baseline:.4f}, MDD≈{mdd_baseline:.4f}, trades≈{baseline_row['trades365']}",
        f"- adopt: cost_on >= {COST_ON_FLOOR:.4f}, MDD <= {MDD_CEILING:.4f} (≥15% MDD reduction)",
        "",
        "| run_id | 30d cost_on | 365d cost_on | 365d MDD | trades |",
        "|--------|------------|-------------|----------|-------|",
    ]
    for r in all_rows:
        c30 = f"{r['c30']:.4f}" if r.get("c30") is not None else "N/A"
        c365 = f"{r['c365']:.4f}" if r.get("c365") is not None else "N/A"
        mdd = f"{r['mdd365']:.4f}" if r.get("mdd365") is not None else "N/A"
        tr = str(int(r["trades365"])) if r.get("trades365") is not None else "N/A"
        lines.append(f"| {r['run_id']} | {c30} | {c365} | {mdd} | {tr} |")
    lines.extend([
        "",
        "## BEST spotcheck (3 runs)",
        f"- cost_on_range: {cost_on_range:.4f}",
        f"- MDD_range: {mdd_range:.4f}",
        f"- trades_range: {trades_range}",
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
        shutil.copy(src_json, BACKTESTS / "phase_d3_best_run.json")
        print(f"  Exported {BACKTESTS / 'phase_d3_best_run.json'}")

    # 7) Export BEST trade log
    print("  Exporting BEST trade log...")
    if run_validation("phase_d3_best_export", best_regime, best_vol_th, emit_trade_log=True, trade_log_path=str(BACKTESTS)) != 0:
        print("WARNING: Trade log export failed.", file=sys.stderr)
    else:
        csv_src = BACKTESTS / "phase_d3_best_export_365d.csv"
        if csv_src.exists():
            with open(csv_src, encoding="utf-8") as f:
                reader = csv_module.DictReader(f)
                rows_csv = list(reader)
            if rows_csv:
                out_csv = BACKTESTS / "phase_d3_best_trades.csv"
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

    # 8) Console output
    print("")
    print("=" * 60)
    print("Phase D3 completed")
    print("=" * 60)
    print(f"BEST run_id:        {best_run_id}")
    print(f"365d cost_on:      {best_row['c365']:.4f}" if best_row.get("c365") is not None else "365d cost_on:      N/A")
    print(f"365d MDD:          {best_row['mdd365']:.4f}" if best_row.get("mdd365") is not None else "365d MDD:          N/A")
    print(f"trades:            {best_row['trades365']}" if best_row.get("trades365") is not None else "trades:            N/A")
    print(f"final verdict:     {verdict}")
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())

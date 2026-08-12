#!/usr/bin/env python3
"""
운영용: Baseline (partial_tp OFF) vs C7 Adopt (partial_tp ON) 비교.
출력: 30d/365d cost_on, cost_off, MDD, trades + partial_tp stats 표.
판정: ADOPT (365d cost_on 개선 >= +0.001, MDD 악화 <= +0.005, trades 감소 <= 5%) / FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG = PROJECT_ROOT / "data" / "diagnostics"
PREFIX = "tcn_candidate_validation_BTCUSDT_5m_h15_t0p004"


def load_json(p: Path) -> dict | None:
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Load error {p}: {e}", file=sys.stderr)
        return None


def get_row(results: list, days: int) -> dict | None:
    for r in results:
        if r.get("days") == days:
            return r
    return None


def find_by_run_id(run_id: str) -> Path | None:
    candidates = sorted(
        DIAG.glob(f"{PREFIX}_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        if run_id in c.stem:
            d = load_json(c)
            if d and d.get("meta", {}).get("run_id") == run_id:
                return c
    for c in candidates:
        if run_id in c.stem:
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline vs C7 Adopt compare (ADOPT/FAIL)")
    ap.add_argument("--baseline-run-id", type=str, default="ops_base_c7", help="Baseline run_id (partial_tp=off)")
    ap.add_argument("--other-run-ids", type=str, default="ops_c7_tp025_r50", help="Comma-separated C7 run_ids")
    args = ap.parse_args()

    base_path = find_by_run_id(args.baseline_run_id)
    if not base_path or not base_path.exists():
        print(f"Baseline JSON not found (run_id={args.baseline_run_id}). Run (A) first.", file=sys.stderr)
        return 1

    other_ids = [x.strip() for x in args.other_run_ids.split(",") if x.strip()]
    other_paths = []
    other_data = []
    for rid in other_ids:
        p = find_by_run_id(rid)
        if not p or not p.exists():
            print(f"Other run JSON not found (run_id={rid}).", file=sys.stderr)
            return 1
        other_paths.append(p)
        d = load_json(p)
        if not d:
            return 1
        other_data.append(d)

    d_base = load_json(base_path)
    if not d_base:
        return 1

    r_base = d_base.get("results", [])
    row_b30 = get_row(r_base, 30)
    row_b365 = get_row(r_base, 365)
    if not row_b365:
        print("Baseline has no 365d result.", file=sys.stderr)
        return 1

    cost_b365 = row_b365.get("cost_on_return")
    mdd_b365 = row_b365.get("max_drawdown")
    tr_b365 = row_b365.get("trades")

    # Table: metric | 30d_baseline | 30d_other1 | ... | 365d_baseline | 365d_other1 | ...
    print("=" * 72)
    print("Baseline vs C7 (Partial TP) — 30d / 365d")
    print("=" * 72)
    print(f"baseline: {base_path.name} (run_id={args.baseline_run_id})")
    for i, rid in enumerate(other_ids):
        print(f"  other:  {other_paths[i].name} (run_id={rid})")
    print()

    cols_30 = ["30d_base"] + [f"30d_{rid}" for rid in other_ids]
    cols_365 = ["365d_base"] + [f"365d_{rid}" for rid in other_ids]
    col_headers = cols_30 + cols_365
    w = 12
    print(f"{'metric':<20} " + " ".join(f"{h:>{w}}" for h in col_headers))
    print("-" * (20 + (w + 1) * len(col_headers)))

    for key, fmt in [
        ("cost_on_return", ".4f"),
        ("cost_off_return", ".4f"),
        ("max_drawdown", ".4f"),
        ("trades", "d"),
    ]:
        vals = []
        vals.append(row_b30.get(key) if row_b30 else None)
        for d in other_data:
            r30 = get_row(d.get("results", []), 30)
            vals.append(r30.get(key) if r30 else None)
        vals.append(row_b365.get(key))
        for d in other_data:
            r365 = get_row(d.get("results", []), 365)
            vals.append(r365.get(key) if r365 else None)
        parts = [f"{v:{fmt}}" if v is not None else "N/A" for v in vals]
        print(f"{key:<20} " + " ".join(f"{p:>{w}}" for p in parts))

    # Partial TP stats (30d / 365d)
    for label, days in [("partial_tp_count", 30), ("partial_tp_count", 365)]:
        row_name = f"partial_tp_count_{days}d"
        vals = []
        r_b = get_row(r_base, days)
        vals.append(r_b.get("partial_tp_count") if r_b else None)
        for d in other_data:
            r = get_row(d.get("results", []), days)
            vals.append(r.get("partial_tp_count") if r else None)
        parts = [f"{v}" if v is not None else "N/A" for v in vals]
        print(f"{row_name:<20} " + " ".join(f"{p:>{w}}" for p in parts))
    for label, days in [("pct_partial_tp", 30), ("pct_partial_tp", 365)]:
        row_name = f"pct_partial_tp_{days}d"
        vals = []
        r_b = get_row(r_base, days)
        vals.append(r_b.get("pct_partial_tp") if r_b is not None else None)
        for d in other_data:
            r = get_row(d.get("results", []), days)
            vals.append(r.get("pct_partial_tp") if r else None)
        parts = [f"{v:.4f}" if v is not None else "N/A" for v in vals]
        print(f"{row_name:<20} " + " ".join(f"{p:>{w}}" for p in parts))

    # Verdict: first "other" run
    verdict = "FAIL"
    if other_data:
        r365 = get_row(other_data[0].get("results", []), 365)
        if r365:
            cost_o = r365.get("cost_on_return")
            mdd_o = r365.get("max_drawdown")
            tr_o = r365.get("trades")
            if (
                cost_o is not None
                and cost_b365 is not None
                and cost_o >= cost_b365 + 0.001
                and mdd_o is not None
                and mdd_b365 is not None
                and mdd_o <= mdd_b365 + 0.005
                and tr_o is not None
                and tr_b365 is not None
                and tr_o >= tr_b365 * 0.95
            ):
                verdict = "ADOPT"

    print()
    print("--- verdict ---")
    print(verdict)
    return 0 if verdict == "ADOPT" else 1


if __name__ == "__main__":
    sys.exit(main())

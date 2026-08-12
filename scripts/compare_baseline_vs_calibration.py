#!/usr/bin/env python3
"""
Phase B1: Baseline (phase_b1_base) vs calibration runs (phase_b1_t11, t12, t14) 비교.
판정: SUCCESS / NO-IMPROVE / FAIL (각 other run별).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG = PROJECT_ROOT / "data" / "diagnostics"
PREFIX = "tcn_candidate_validation_BTCUSDT_5m_h15_t0p004"


def load_json(p: Path) -> dict | None:
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Load error {p}: {e}")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run-id", type=str, default="phase_b1_base", help="Baseline run_id")
    ap.add_argument("--other-run-ids", type=str, default="phase_b1_t11,phase_b1_t12,phase_b1_t14", help="Comma-separated other run_ids")
    args = ap.parse_args()

    base_path = find_by_run_id(args.baseline_run_id)
    if not base_path or not base_path.exists():
        print(f"Baseline JSON not found (run_id={args.baseline_run_id}). Run Phase B1 base first.")
        return 1

    other_ids = [x.strip() for x in args.other_run_ids.split(",") if x.strip()]
    other_paths = []
    other_data = []
    for rid in other_ids:
        p = find_by_run_id(rid)
        if not p or not p.exists():
            print(f"Other run JSON not found (run_id={rid}).")
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
        print("Baseline has no 365d result.")
        return 1

    cost_b30 = row_b30.get("cost_on_return") if row_b30 else None
    cost_b365 = row_b365.get("cost_on_return")
    mdd_b365 = row_b365.get("max_drawdown")
    trades_b365 = row_b365.get("trades")

    # Table: baseline + each other (30d / 365d)
    print("=" * 72)
    print("Phase B1: Baseline vs Calibration (Temperature Scaling) — 30d / 365d")
    print("=" * 72)
    print(f"baseline: {base_path.name} (run_id={args.baseline_run_id})")
    for i, rid in enumerate(other_ids):
        print(f"  other:  {other_paths[i].name} (run_id={rid})")
    print()

    # Header: metric, 30d_baseline, 30d_t11, 30d_t12, 30d_t14, 365d_baseline, 365d_t11, ...
    cols_30 = ["30d_base"] + [f"30d_{rid.replace('phase_b1_', '')}" for rid in other_ids]
    cols_365 = ["365d_base"] + [f"365d_{rid.replace('phase_b1_', '')}" for rid in other_ids]
    col_headers = cols_30 + cols_365
    w = 10
    print(f"{'metric':<20} " + " ".join(f"{h:>{w}}" for h in col_headers))
    print("-" * (20 + (w + 1) * len(col_headers)))

    for key, fmt in [
        ("cost_on_return", ".4f"),
        ("cost_off_return", ".4f"),
        ("max_drawdown", ".4f"),
        ("trades", "d"),
    ]:
        vals = []
        vb30 = row_b30.get(key) if row_b30 else None
        vals.append(vb30)
        for d in other_data:
            r30 = get_row(d.get("results", []), 30)
            vals.append(r30.get(key) if r30 else None)
        vb365 = row_b365.get(key)
        vals.append(vb365)
        for d in other_data:
            r365 = get_row(d.get("results", []), 365)
            vals.append(r365.get(key) if r365 else None)
        parts = [f"{v:{fmt}}" if v is not None else "N/A" for v in vals]
        print(f"{key:<20} " + " ".join(f"{p:>{w}}" for p in parts))

    print()
    print("Calibration T per run:")
    print(f"  baseline: calibration={d_base.get('meta', {}).get('calibration', 'N/A')}, T={d_base.get('meta', {}).get('calibration_T', 'N/A')}")
    for i, d in enumerate(other_data):
        meta = d.get("meta", {})
        print(f"  {other_ids[i]}: calibration={meta.get('calibration', 'N/A')}, T={meta.get('calibration_T', 'N/A')}")

    # Best 365d cost_on among baseline and others
    candidates = [("baseline", cost_b365)]
    for i, d in enumerate(other_data):
        r365 = get_row(d.get("results", []), 365)
        c = r365.get("cost_on_return") if r365 else None
        if c is not None:
            candidates.append((other_ids[i], c))
    best_run = max(candidates, key=lambda x: x[1] if x[1] is not None else -1e9)
    print()
    print(f"최고 365d cost_on: {best_run[0]} (cost_on={best_run[1]:.4f})")

    # Verdict per other run
    print()
    for i, rid in enumerate(other_ids):
        d = other_data[i]
        r30 = get_row(d.get("results", []), 30)
        r365 = get_row(d.get("results", []), 365)
        if not r365:
            print(f"판정 [{rid}]: FAIL (no 365d result)")
            continue
        cost_o30 = r30.get("cost_on_return") if r30 else None
        cost_o365 = r365.get("cost_on_return")
        mdd_o365 = r365.get("max_drawdown")
        tr_o365 = r365.get("trades")

        fail = False
        if mdd_b365 is not None and mdd_o365 is not None and mdd_o365 > mdd_b365 + 0.005:
            fail = True
        if trades_b365 and tr_o365 is not None and tr_o365 < trades_b365 * 0.95:
            fail = True
        if fail:
            print(f"판정 [{rid}]: FAIL")
            continue

        success = (
            cost_o365 is not None
            and cost_b365 is not None
            and cost_o365 >= cost_b365 + 0.003
            and mdd_o365 is not None
            and mdd_b365 is not None
            and mdd_o365 <= mdd_b365 + 0.005
            and tr_o365 is not None
            and trades_b365
            and tr_o365 >= trades_b365 * 0.95
        )
        if cost_b30 is not None and cost_o30 is not None and cost_b30 > 0 and cost_o30 < cost_b30 * 0.8:
            success = False  # 30d cost_on 20% 이상 악화 금지
        if success:
            print(f"판정 [{rid}]: SUCCESS")
        else:
            print(f"판정 [{rid}]: NO-IMPROVE")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

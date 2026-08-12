#!/usr/bin/env python3
"""
Phase A: base vs conservative commission/slippage 비교.
PASS: 두 run 존재 + 비교표. FLAG: 보수 시나리오에서 365d cost_on 0.010 이상 악화 또는 MDD 0.010 이상 악화.
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
    candidates = sorted(DIAG.glob(f"{PREFIX}_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
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
    ap.add_argument("--base-run-id", type=str, default="phase_a_base")
    ap.add_argument("--other-run-id", type=str, default="phase_a_conservative")
    ap.add_argument("--base-json", type=Path, default=None)
    ap.add_argument("--other-json", type=Path, default=None)
    args = ap.parse_args()

    base_path = args.base_json or find_by_run_id(args.base_run_id)
    other_path = args.other_json or find_by_run_id(args.other_run_id)

    if not base_path or not base_path.exists():
        print(f"Base JSON not found (run_id={args.base_run_id}). Run Phase A first.")
        return 1
    if not other_path or not other_path.exists():
        print(f"Other JSON not found (run_id={args.other_run_id}). Run Phase A first.")
        return 1

    d_base = load_json(base_path)
    d_other = load_json(other_path)
    if not d_base or not d_other:
        return 1

    r_base = d_base.get("results", [])
    r_other = d_other.get("results", [])
    row_b30 = get_row(r_base, 30)
    row_b365 = get_row(r_base, 365)
    row_o30 = get_row(r_other, 30)
    row_o365 = get_row(r_other, 365)

    print("=" * 72)
    print("Phase A: base vs conservative")
    print("=" * 72)
    print(f"base:        {base_path.name}")
    print(f"conservative: {other_path.name}")
    print()
    print(f"{'metric':<24} {'30d_base':>12} {'30d_other':>12} {'365d_base':>12} {'365d_other':>12}")
    print("-" * 72)

    for key, fmt in [("cost_on_return", ".4f"), ("cost_off_return", ".4f"), ("max_drawdown", ".4f"), ("trades", "d")]:
        vb30 = row_b30.get(key) if row_b30 else None
        vo30 = row_o30.get(key) if row_o30 else None
        vb365 = row_b365.get(key) if row_b365 else None
        vo365 = row_o365.get(key) if row_o365 else None
        sb30 = f"{vb30:{fmt}}" if vb30 is not None else "N/A"
        so30 = f"{vo30:{fmt}}" if vo30 is not None else "N/A"
        sb365 = f"{vb365:{fmt}}" if vb365 is not None else "N/A"
        so365 = f"{vo365:{fmt}}" if vo365 is not None else "N/A"
        print(f"{key:<24} {sb30:>12} {so30:>12} {sb365:>12} {so365:>12}")

    cost_b365 = row_b365.get("cost_on_return") if row_b365 else None
    cost_o365 = row_o365.get("cost_on_return") if row_o365 else None
    mdd_b365 = row_b365.get("max_drawdown") if row_b365 else None
    mdd_o365 = row_o365.get("max_drawdown") if row_o365 else None

    verdict = "PASS"
    if cost_b365 is not None and cost_o365 is not None and cost_o365 <= cost_b365 - 0.010:
        verdict = "FLAG (365d cost_on 0.010+ 악화)"
    if mdd_b365 is not None and mdd_o365 is not None and mdd_o365 >= mdd_b365 + 0.010:
        verdict = "FLAG (MDD 0.010+ 악화)"
    print()
    print(f"판정: {verdict}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

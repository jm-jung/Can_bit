#!/usr/bin/env python3
"""
Baseline (phase_a_base) vs Flat-gate (phase_c_flatgate_045) 비교.
SUCCESS: 365d cost_on baseline 대비 +0.003 이상 개선.
NO-IMPROVE: 그 외. OVERFILTER: trades -5% 이상 감소 시 경고.
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
    ap.add_argument("--baseline-run-id", type=str, default="phase_a_base", help="Baseline run_id (default phase_a_base)")
    ap.add_argument("--flatgate-run-id", type=str, default="phase_c_flatgate_045", help="Flat-gate run_id (default phase_c_flatgate_045)")
    ap.add_argument("--baseline", type=Path, default=None, help="Override baseline JSON path")
    ap.add_argument("--flatgate", type=Path, default=None, help="Override flatgate JSON path")
    args = ap.parse_args()

    base_path = args.baseline or find_by_run_id(args.baseline_run_id)
    fg_path = args.flatgate or find_by_run_id(args.flatgate_run_id)

    if not base_path or not base_path.exists():
        print(f"Baseline JSON not found (run_id={args.baseline_run_id}). Run Phase A first.")
        return 1
    if not fg_path or not fg_path.exists():
        print(f"Flatgate JSON not found (run_id={args.flatgate_run_id}). Run Phase C first.")
        return 1

    d_base = load_json(base_path)
    d_fg = load_json(fg_path)
    if not d_base or not d_fg:
        return 1

    r_base = d_base.get("results", [])
    r_fg = d_fg.get("results", [])
    row_b30 = get_row(r_base, 30)
    row_b365 = get_row(r_base, 365)
    row_f30 = get_row(r_fg, 30)
    row_f365 = get_row(r_fg, 365)

    print("=" * 72)
    print("Baseline (phase_a_base) vs Flat-gate (phase_c_flatgate_045) — 30d / 365d")
    print("=" * 72)
    print(f"baseline:  {base_path.name}")
    print(f"flatgate:  {fg_path.name}")
    print()
    print(f"{'metric':<24} {'30d_baseline':>14} {'30d_flatgate':>14} {'365d_baseline':>14} {'365d_flatgate':>14}")
    print("-" * 82)

    for key, fmt in [
        ("cost_on_return", ".4f"),
        ("cost_off_return", ".4f"),
        ("max_drawdown", ".4f"),
        ("trades", "d"),
    ]:
        vb30 = row_b30.get(key) if row_b30 else None
        vf30 = row_f30.get(key) if row_f30 else None
        vb365 = row_b365.get(key) if row_b365 else None
        vf365 = row_f365.get(key) if row_f365 else None
        sb30 = f"{vb30:{fmt}}" if vb30 is not None else "N/A"
        sf30 = f"{vf30:{fmt}}" if vf30 is not None else "N/A"
        sb365 = f"{vb365:{fmt}}" if vb365 is not None else "N/A"
        sf365 = f"{vf365:{fmt}}" if vf365 is not None else "N/A"
        print(f"{key:<24} {sb30:>14} {sf30:>14} {sb365:>14} {sf365:>14}")

    print()
    print("Flat-gate stats (365d)")
    if row_f365 is not None:
        print(f"  pct_flat_gate_block: {row_f365.get('pct_flat_gate_block')}, entries_blocked_by_flat_gate: {row_f365.get('entries_blocked_by_flat_gate')}, max_flat_proba: {row_f365.get('max_flat_proba')}")
    print()

    cost_b365 = row_b365.get("cost_on_return") if row_b365 else None
    cost_f365 = row_f365.get("cost_on_return") if row_f365 else None
    tr_b365 = row_b365.get("trades") if row_b365 else None
    tr_f365 = row_f365.get("trades") if row_f365 else None

    verdict = "NO-IMPROVE"
    if cost_b365 is not None and cost_f365 is not None:
        if cost_f365 >= cost_b365 + 0.003:
            verdict = "SUCCESS (365d cost_on +0.003 이상 개선)"
        else:
            verdict = "NO-IMPROVE"
    overfilter = ""
    if tr_b365 and tr_f365 is not None and tr_b365 > 0:
        if (tr_b365 - tr_f365) / tr_b365 >= 0.05:
            overfilter = " [OVERFILTER: trades -5% 이상 감소]"
    print(f"판정: {verdict}{overfilter}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

#!/usr/bin/env python3
"""
Baseline (early_exit=off) vs Early Exit (early_exit=on) 결과 비교 (30d / 365d).
사용법:
  python -m scripts.compare_baseline_vs_early_exit
  python -m scripts.compare_baseline_vs_early_exit --baseline path/to/baseline.json --early-exit path/to/early_exit.json
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


def find_latest_baseline() -> Path | None:
    """early_exit=off 최신 JSON (같은 id/symbol/timeframe)."""
    candidates = sorted(
        DIAG.glob(f"{PREFIX}_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        d = load_json(c)
        if not d:
            continue
        meta = d.get("meta", {})
        if meta.get("early_exit") in ("off", None) or meta.get("early_exit_enabled") in (False, None):
            return c
    return None


def find_latest_early_exit_on() -> Path | None:
    """early_exit=on 최신 JSON."""
    candidates = sorted(
        DIAG.glob(f"{PREFIX}_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        d = load_json(c)
        if d and (d.get("meta", {}).get("early_exit") == "on" or d.get("meta", {}).get("early_exit_enabled") is True):
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=None, help="baseline JSON (기본: 최신 early_exit=off)")
    ap.add_argument("--early-exit", type=Path, default=None, help="early exit JSON (기본: 최신 early_exit=on)")
    args = ap.parse_args()

    if args.baseline is None:
        args.baseline = find_latest_baseline()
    if args.early_exit is None:
        args.early_exit = find_latest_early_exit_on()

    if not args.baseline or not args.baseline.exists():
        print("baseline JSON not found (early_exit=off). Run candidate_validation with --early-exit off first.")
        return 1
    if not args.early_exit or not args.early_exit.exists():
        print("early_exit JSON not found. Run candidate_validation with --early-exit on first.")
        return 1

    d_base = load_json(args.baseline)
    d_ee = load_json(args.early_exit)
    if not d_base or not d_ee:
        return 1

    r_base = d_base.get("results", [])
    r_ee = d_ee.get("results", [])
    meta_ee = d_ee.get("meta", {})

    print("=" * 72)
    print("Baseline (early_exit OFF) vs Early Exit (early_exit ON) — 30d / 365d")
    print("=" * 72)
    print(f"baseline:   {args.baseline.name}")
    print(f"early_exit: {args.early_exit.name} (lookback={meta_ee.get('early_exit_lookback')}, p_floor={meta_ee.get('early_exit_p_floor')}, bad_k={meta_ee.get('early_exit_bad_k')})")
    print()
    print(f"{'metric':<24} {'30d_baseline':>14} {'30d_early_exit':>16} {'365d_baseline':>14} {'365d_early_exit':>16}")
    print("-" * 86)

    row_b30 = get_row(r_base, 30)
    row_e30 = get_row(r_ee, 30)
    row_b365 = get_row(r_base, 365)
    row_e365 = get_row(r_ee, 365)

    for key, fmt in [
        ("cost_on_return", ".4f"),
        ("cost_off_return", ".4f"),
        ("max_drawdown", ".4f"),
        ("trades", "d"),
    ]:
        vb30 = row_b30.get(key) if row_b30 else None
        ve30 = row_e30.get(key) if row_e30 else None
        vb365 = row_b365.get(key) if row_b365 else None
        ve365 = row_e365.get(key) if row_e365 else None
        sb30 = f"{vb30:{fmt}}" if vb30 is not None else "N/A"
        se30 = f"{ve30:{fmt}}" if ve30 is not None else "N/A"
        sb365 = f"{vb365:{fmt}}" if vb365 is not None else "N/A"
        se365 = f"{ve365:{fmt}}" if ve365 is not None else "N/A"
        print(f"{key:<24} {sb30:>14} {se30:>16} {sb365:>14} {se365:>16}")

    print()
    print("Early exit stats (365d)")
    if row_e365 is not None:
        print(f"  early_exit_count: {row_e365.get('early_exit_count')}, early_exit_rate: {row_e365.get('early_exit_rate')}, lookback: {row_e365.get('early_exit_lookback')}, p_floor: {row_e365.get('early_exit_p_floor')}, bad_k: {row_e365.get('early_exit_bad_k')}")
    print()
    cost_b365 = row_b365.get("cost_on_return") if row_b365 else None
    cost_e365 = row_e365.get("cost_on_return") if row_e365 else None
    if cost_b365 is not None and cost_e365 is not None:
        improved = cost_e365 >= cost_b365 - 0.001
        print("365d cost_on 개선 여부:", "개선 또는 유지" if improved else "악화", f"(baseline={cost_b365:.4f}, early_exit={cost_e365:.4f})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

#!/usr/bin/env python3
"""
baseline (regime=off, position_scaling=off) vs position_scaling (linear/sigmoid) 결과 비교 (30d / 365d).
사용법:
  python -m scripts.compare_baseline_vs_position_scaling
  python -m scripts.compare_baseline_vs_position_scaling --baseline path/to/baseline.json --scaled path/to/scaled.json
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
    """position_scaling=off 또는 미설정 최신 JSON (같은 id/symbol/timeframe)."""
    candidates = sorted(
        DIAG.glob(f"{PREFIX}_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        d = load_json(c)
        if not d:
            continue
        ps = d.get("meta", {}).get("position_scaling")
        if ps is None or ps == "off":
            return c
    return None


def find_latest_scaled() -> Path | None:
    """position_scaling in (linear, sigmoid) 최신 JSON."""
    candidates = sorted(
        DIAG.glob(f"{PREFIX}_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        d = load_json(c)
        if d and d.get("meta", {}).get("position_scaling") in ("linear", "sigmoid"):
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=None, help="baseline JSON (기본: 최신 regime=off, position_scaling=off)")
    ap.add_argument("--scaled", type=Path, default=None, help="scaled JSON (기본: 최신 position_scaling=linear/sigmoid)")
    args = ap.parse_args()

    if args.baseline is None:
        args.baseline = find_latest_baseline()
    if args.scaled is None:
        args.scaled = find_latest_scaled()

    if not args.baseline or not args.baseline.exists():
        print("baseline JSON not found (regime=off, position_scaling=off).")
        return 1
    if not args.scaled or not args.scaled.exists():
        print("scaled JSON not found. Run candidate_validation with --position-scaling linear (or sigmoid) first.")
        return 1

    d_base = load_json(args.baseline)
    d_scaled = load_json(args.scaled)
    if not d_base or not d_scaled:
        return 1

    r_base = d_base.get("results", [])
    r_scaled = d_scaled.get("results", [])
    meta_scaled = d_scaled.get("meta", {})

    print("=" * 72)
    print("baseline (scaling OFF) vs position_scaling — 30d / 365d")
    print("=" * 72)
    print(f"baseline: {args.baseline.name}")
    print(f"scaled:   {args.scaled.name} (position_scaling={meta_scaled.get('position_scaling')}, p_floor={meta_scaled.get('position_p_floor')}, p_full={meta_scaled.get('position_p_full')})")
    print()
    print(f"{'metric':<24} {'30d_baseline':>14} {'30d_scaled':>14} {'365d_baseline':>14} {'365d_scaled':>14}")
    print("-" * 72)

    row_b30 = get_row(r_base, 30)
    row_s30 = get_row(r_scaled, 30)
    row_b365 = get_row(r_base, 365)
    row_s365 = get_row(r_scaled, 365)

    for key, fmt in [
        ("cost_on_return", ".4f"),
        ("cost_off_return", ".4f"),
        ("max_drawdown", ".4f"),
        ("trades", "d"),
    ]:
        vb30 = row_b30.get(key) if row_b30 else None
        vs30 = row_s30.get(key) if row_s30 else None
        vb365 = row_b365.get(key) if row_b365 else None
        vs365 = row_s365.get(key) if row_s365 else None
        sb30 = f"{vb30:{fmt}}" if vb30 is not None else "N/A"
        ss30 = f"{vs30:{fmt}}" if vs30 is not None else "N/A"
        sb365 = f"{vb365:{fmt}}" if vb365 is not None else "N/A"
        ss365 = f"{vs365:{fmt}}" if vs365 is not None else "N/A"
        print(f"{key:<24} {sb30:>14} {ss30:>14} {sb365:>14} {ss365:>14}")

    print()
    print("Position scaling stats (365d)")
    if row_s365:
        print(f"  scaled: scale_mean={row_s365.get('scale_mean')}, entries_scaled_count={row_s365.get('entries_scaled_count')}, entries_scaled_applied_count={row_s365.get('entries_scaled_applied_count')}, scale_bins={row_s365.get('scale_bins')}")
    print()
    cost_b365 = row_b365.get("cost_on_return") if row_b365 else None
    cost_s365 = row_s365.get("cost_on_return") if row_s365 else None
    if cost_b365 is not None and cost_s365 is not None:
        improved = cost_s365 > cost_b365
        print("365d cost_on 개선 여부:", "개선" if improved else "미개선", f"(baseline={cost_b365:.4f}, scaled={cost_s365:.4f})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

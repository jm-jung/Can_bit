#!/usr/bin/env python3
"""
vol_compress (baseline) vs vol_breakout 결과 비교 (30d / 365d).
사용법:
  python -m scripts.compare_vol_compress_vs_vol_breakout
  python -m scripts.compare_vol_compress_vs_vol_breakout --baseline path/to/vol_compress.json --vol-breakout path/to/vol_breakout.json
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


def find_latest_by_regime(regime_rule: str) -> Path | None:
    candidates = sorted(
        DIAG.glob(f"{PREFIX}_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        d = load_json(c)
        if d and d.get("meta", {}).get("regime_rule") == regime_rule:
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=None, help="vol_compress JSON (기본: vol_t0p0010 또는 최신)")
    ap.add_argument("--vol-breakout", type=Path, default=None, help="vol_breakout JSON (기본: 최신 vol_breakout)")
    args = ap.parse_args()

    if args.baseline is None:
        args.baseline = DIAG / f"{PREFIX}_vol_t0p0010.json"
        if not args.baseline.exists():
            args.baseline = find_latest_by_regime("vol_compress")
    if args.vol_breakout is None:
        args.vol_breakout = find_latest_by_regime("vol_breakout")

    if not args.baseline or not args.baseline.exists():
        print("vol_compress (baseline) JSON not found.")
        return 1
    if not args.vol_breakout or not args.vol_breakout.exists():
        print("vol_breakout JSON not found. Run candidate_validation with --regime-filter vol_breakout first.")
        return 1

    d_base = load_json(args.baseline)
    d_bo = load_json(args.vol_breakout)
    if not d_base or not d_bo:
        return 1

    r_base = d_base.get("results", [])
    r_bo = d_bo.get("results", [])
    meta_bo = d_bo.get("meta", {})
    breakout_mode = meta_bo.get("breakout_mode", "?")

    print("=" * 72)
    print("vol_compress (baseline) vs vol_breakout — 30d / 365d")
    print("=" * 72)
    print(f"baseline:     {args.baseline.name}")
    print(f"vol_breakout: {args.vol_breakout.name} (breakout_mode={breakout_mode})")
    print()
    print(f"{'metric':<24} {'30d_baseline':>14} {'30d_breakout':>14} {'365d_baseline':>14} {'365d_breakout':>14}")
    print("-" * 72)

    row_b30 = get_row(r_base, 30)
    row_bo30 = get_row(r_bo, 30)
    row_b365 = get_row(r_base, 365)
    row_bo365 = get_row(r_bo, 365)

    for key, fmt in [
        ("cost_on_return", ".4f"),
        ("cost_off_return", ".4f"),
        ("max_drawdown", ".4f"),
        ("trades", "d"),
    ]:
        vb30 = row_b30.get(key) if row_b30 else None
        vbo30 = row_bo30.get(key) if row_bo30 else None
        vb365 = row_b365.get(key) if row_b365 else None
        vbo365 = row_bo365.get(key) if row_bo365 else None
        sb30 = f"{vb30:{fmt}}" if vb30 is not None else "N/A"
        sbo30 = f"{vbo30:{fmt}}" if vbo30 is not None else "N/A"
        sb365 = f"{vb365:{fmt}}" if vb365 is not None else "N/A"
        sbo365 = f"{vbo365:{fmt}}" if vbo365 is not None else "N/A"
        print(f"{key:<24} {sb30:>14} {sbo30:>14} {sb365:>14} {sbo365:>14}")

    print()
    print("Regime stats (365d)")
    if row_b365:
        print(f"  baseline:     pct_compress={row_b365.get('pct_compress')}, entries_blocked_by_regime_vol={row_b365.get('entries_blocked_by_regime_vol')}, blocked_ratio={row_b365.get('blocked_ratio')}")
    if row_bo365:
        print(f"  vol_breakout: pct_compress={row_bo365.get('pct_compress')}, entries_blocked_by_regime_vol={row_bo365.get('entries_blocked_by_regime_vol')}, entries_blocked_by_regime_vol_breakout={row_bo365.get('entries_blocked_by_regime_vol_breakout')}, entries_allowed_on_decompress={row_bo365.get('entries_allowed_on_decompress')}, blocked_ratio={row_bo365.get('blocked_ratio')}")
    print()
    print("Summary")
    print(f"  baseline:     overblock_nogo={d_base.get('summary', {}).get('overblock_nogo')}, overblock_warning={d_base.get('summary', {}).get('overblock_warning')}")
    print(f"  vol_breakout: overblock_nogo={d_bo.get('summary', {}).get('overblock_nogo')}, overblock_warning={d_bo.get('summary', {}).get('overblock_warning')}")
    cost_b365 = row_b365.get("cost_on_return") if row_b365 else None
    cost_bo365 = row_bo365.get("cost_on_return") if row_bo365 else None
    if cost_b365 is not None and cost_bo365 is not None:
        improved = cost_bo365 > cost_b365
        print()
        print(f"  365d cost_on: baseline={cost_b365:.4f}, vol_breakout={cost_bo365:.4f} -> {'개선' if improved else '미개선'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

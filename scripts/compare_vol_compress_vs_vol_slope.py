#!/usr/bin/env python3
"""
vol_compress vs vol_slope 1차 검증 결과 비교 (30d / 365d).
사용법: vol_slope run 완료 후
  python -m scripts.compare_vol_compress_vs_vol_slope

또는 JSON 경로 지정:
  python -m scripts.compare_vol_compress_vs_vol_slope --vol-compress path/to/vol_compress.json --vol-slope path/to/vol_slope.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG = PROJECT_ROOT / "data" / "diagnostics"


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol-compress", type=Path, default=None, help="vol_compress 결과 JSON (기본: diagnostics에서 vol_t0p0010)")
    ap.add_argument("--vol-slope", type=Path, default=None, help="vol_slope 결과 JSON (기본: diagnostics에서 최신 vol_slope)")
    args = ap.parse_args()

    if args.vol_compress is None:
        p_vol = DIAG / "tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_vol_t0p0010.json"
        if not p_vol.exists():
            candidates = list(DIAG.glob("tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_*.json"))
            for c in candidates:
                d = load_json(c)
                if d and d.get("meta", {}).get("regime_rule") == "vol_compress":
                    p_vol = c
                    break
        args.vol_compress = p_vol
    if args.vol_slope is None:
        candidates = sorted(DIAG.glob("tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        for c in candidates:
            d = load_json(c)
            if d and d.get("meta", {}).get("regime_rule") == "vol_slope":
                args.vol_slope = c
                break
        else:
            args.vol_slope = None

    if not args.vol_compress or not args.vol_compress.exists():
        print("vol_compress JSON not found.")
        return 1
    if not args.vol_slope or not args.vol_slope.exists():
        print("vol_slope JSON not found. Run candidate_validation with --regime-filter vol_slope first.")
        return 1

    d_vol = load_json(args.vol_compress)
    d_slope = load_json(args.vol_slope)
    if not d_vol or not d_slope:
        return 1

    r_vol = d_vol.get("results", [])
    r_slope = d_slope.get("results", [])
    s_vol = d_vol.get("summary", {})
    s_slope = d_slope.get("summary", {})

    print("=" * 70)
    print("vol_compress (baseline) vs vol_slope — 30d / 365d")
    print("=" * 70)
    print(f"vol_compress: {args.vol_compress.name}")
    print(f"vol_slope:    {args.vol_slope.name}")
    print()
    print(f"{'metric':<24} {'30d_vol_compress':>18} {'30d_vol_slope':>18} {'365d_vol_compress':>18} {'365d_vol_slope':>18}")
    print("-" * 70)

    row_vol_30 = get_row(r_vol, 30)
    row_slope_30 = get_row(r_slope, 30)
    row_vol_365 = get_row(r_vol, 365)
    row_slope_365 = get_row(r_slope, 365)

    for key, fmt in [
        ("cost_on_return", ".4f"),
        ("cost_off_return", ".4f"),
        ("max_drawdown", ".4f"),
        ("trades", "d"),
    ]:
        v_v30 = row_vol_30.get(key) if row_vol_30 else None
        v_s30 = row_slope_30.get(key) if row_slope_30 else None
        v_v365 = row_vol_365.get(key) if row_vol_365 else None
        v_s365 = row_slope_365.get(key) if row_slope_365 else None
        s_v30 = f"{v_v30:{fmt}}" if v_v30 is not None else "N/A"
        s_s30 = f"{v_s30:{fmt}}" if v_s30 is not None else "N/A"
        s_v365 = f"{v_v365:{fmt}}" if v_v365 is not None else "N/A"
        s_s365 = f"{v_s365:{fmt}}" if v_s365 is not None else "N/A"
        print(f"{key:<24} {s_v30:>18} {s_s30:>18} {s_v365:>18} {s_s365:>18}")

    print()
    print("Regime stats (365d)")
    if row_vol_365:
        print(f"  vol_compress: pct_compress={row_vol_365.get('pct_compress')}, entries_blocked_by_regime_vol={row_vol_365.get('entries_blocked_by_regime_vol')}, blocked_ratio={row_vol_365.get('blocked_ratio')}")
    if row_slope_365:
        print(f"  vol_slope:    pct_vol_slope_block={row_slope_365.get('pct_vol_slope_block')}, entries_blocked_by_regime_vol_slope={row_slope_365.get('entries_blocked_by_regime_vol_slope')}, blocked_ratio={row_slope_365.get('blocked_ratio')}")
    print()
    print("Summary")
    print(f"  vol_compress: overblock_nogo={s_vol.get('overblock_nogo')}, overblock_warning={s_vol.get('overblock_warning')}")
    print(f"  vol_slope:    overblock_nogo={s_slope.get('overblock_nogo')}, overblock_warning={s_slope.get('overblock_warning')}")
    print()
    cost_365_vol = row_vol_365.get("cost_on_return") if row_vol_365 else None
    cost_365_slope = row_slope_365.get("cost_on_return") if row_slope_365 else None
    if cost_365_vol is not None and cost_365_slope is not None:
        improved = cost_365_slope >= -0.03 or cost_365_slope > cost_365_vol
        print(f"  365d cost_on: vol_compress={cost_365_vol:.4f}, vol_slope={cost_365_slope:.4f} -> {'개선' if improved else '미개선'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

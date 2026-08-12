#!/usr/bin/env python3
"""
baseline (vol_compress 또는 off) vs proba_quantile 결과 비교 (30d / 365d).
사용법:
  python -m scripts.compare_vol_compress_vs_proba_quantile
  python -m scripts.compare_vol_compress_vs_proba_quantile --baseline path/to/baseline.json --proba-quantile path/to/proba_quantile.json
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


def find_latest_baseline_non_pq() -> Path | None:
    """regime_rule != proba_quantile 인 최신 JSON (regime=off 저장이 ema_only인 경우 대비)."""
    candidates = sorted(
        DIAG.glob(f"{PREFIX}_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        d = load_json(c)
        if d and d.get("meta", {}).get("regime_rule") != "proba_quantile":
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=None, help="baseline JSON (기본: 최신 vol_compress, 없으면 최신 off)")
    ap.add_argument("--proba-quantile", type=Path, default=None, help="proba_quantile JSON (기본: 최신 proba_quantile)")
    args = ap.parse_args()

    if args.baseline is None:
        args.baseline = find_latest_by_regime("vol_compress")
        if not args.baseline or not args.baseline.exists():
            args.baseline = find_latest_by_regime("off")
        if not args.baseline or not args.baseline.exists():
            args.baseline = find_latest_baseline_non_pq()
    if args.proba_quantile is None:
        args.proba_quantile = find_latest_by_regime("proba_quantile")

    if not args.baseline or not args.baseline.exists():
        print("baseline JSON not found (vol_compress or off).")
        return 1
    if not args.proba_quantile or not args.proba_quantile.exists():
        print("proba_quantile JSON not found. Run candidate_validation with --regime-filter proba_quantile first.")
        return 1

    d_base = load_json(args.baseline)
    d_pq = load_json(args.proba_quantile)
    if not d_base or not d_pq:
        return 1

    r_base = d_base.get("results", [])
    r_pq = d_pq.get("results", [])
    meta_pq = d_pq.get("meta", {})
    base_rule = d_base.get("meta", {}).get("regime_rule", "?")

    print("=" * 72)
    print("baseline vs proba_quantile — 30d / 365d")
    print("=" * 72)
    print(f"baseline:       {args.baseline.name} (regime_rule={base_rule})")
    print(f"proba_quantile: {args.proba_quantile.name} (q_window={meta_pq.get('q_window')}, q={meta_pq.get('q')}, p_floor={meta_pq.get('p_floor')})")
    print()
    print(f"{'metric':<24} {'30d_baseline':>14} {'30d_pq':>14} {'365d_baseline':>14} {'365d_pq':>14}")
    print("-" * 72)

    row_b30 = get_row(r_base, 30)
    row_pq30 = get_row(r_pq, 30)
    row_b365 = get_row(r_base, 365)
    row_pq365 = get_row(r_pq, 365)

    for key, fmt in [
        ("cost_on_return", ".4f"),
        ("cost_off_return", ".4f"),
        ("max_drawdown", ".4f"),
        ("trades", "d"),
    ]:
        vb30 = row_b30.get(key) if row_b30 else None
        vpq30 = row_pq30.get(key) if row_pq30 else None
        vb365 = row_b365.get(key) if row_b365 else None
        vpq365 = row_pq365.get(key) if row_pq365 else None
        sb30 = f"{vb30:{fmt}}" if vb30 is not None else "N/A"
        spq30 = f"{vpq30:{fmt}}" if vpq30 is not None else "N/A"
        sb365 = f"{vb365:{fmt}}" if vb365 is not None else "N/A"
        spq365 = f"{vpq365:{fmt}}" if vpq365 is not None else "N/A"
        print(f"{key:<24} {sb30:>14} {spq30:>14} {sb365:>14} {spq365:>14}")

    print()
    print("Regime stats (365d)")
    if row_b365:
        print(f"  baseline:       blocked_ratio={row_b365.get('blocked_ratio')}, entries_blocked_by_regime={row_b365.get('entries_blocked_by_regime')}")
    if row_pq365:
        print(f"  proba_quantile: q_threshold_mean={row_pq365.get('q_threshold_mean')}, pct_proba_quantile_block={row_pq365.get('pct_proba_quantile_block')}, entries_blocked_by_regime_proba_quantile={row_pq365.get('entries_blocked_by_regime_proba_quantile')}, blocked_ratio={row_pq365.get('blocked_ratio')}")
    print()
    print("Summary")
    print(f"  baseline:       overblock_nogo={d_base.get('summary', {}).get('overblock_nogo')}, overblock_warning={d_base.get('summary', {}).get('overblock_warning')}")
    print(f"  proba_quantile: overblock_nogo={d_pq.get('summary', {}).get('overblock_nogo')}, overblock_warning={d_pq.get('summary', {}).get('overblock_warning')}")
    cost_b365 = row_b365.get("cost_on_return") if row_b365 else None
    cost_pq365 = row_pq365.get("cost_on_return") if row_pq365 else None
    if cost_b365 is not None and cost_pq365 is not None:
        improved = cost_pq365 > cost_b365
        print()
        print("365d cost_on 개선 여부:", "개선" if improved else "미개선", f"(baseline={cost_b365:.4f}, proba_quantile={cost_pq365:.4f})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

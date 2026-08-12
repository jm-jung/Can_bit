#!/usr/bin/env python3
"""
Phase D1: baseline 모델 vs 다른 모델 ID(들) 비교. pinned end_date 기준 30d/365d.
판정: 365d cost_on +0.003 이상 개선 AND MDD baseline+0.005 이내 → ADOPT_CANDIDATE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG = PROJECT_ROOT / "data" / "diagnostics"


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


def find_json(symbol: str, timeframe: str, model_id: str, run_id: str) -> Path | None:
    prefix = f"tcn_candidate_validation_{symbol}_{timeframe}_{model_id}"
    candidates = sorted(
        DIAG.glob(f"{prefix}_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        if run_id in c.stem or run_id == model_id:
            d = load_json(c)
            if d and d.get("meta", {}).get("id") == model_id:
                return c
    for c in candidates:
        if run_id in c.stem or model_id in c.stem:
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-id", type=str, default="h15_t0p004")
    ap.add_argument("--other-ids", type=str, default="h15_t0p004_v2")
    ap.add_argument("--pinned-end-date", type=str, default="2026-03-03")
    ap.add_argument("--days-list", type=str, default="30,365")
    ap.add_argument("--symbol", type=str, default="BTCUSDT")
    ap.add_argument("--timeframe", type=str, default="5m")
    args = ap.parse_args()

    days_list = [int(x.strip()) for x in args.days_list.split(",") if x.strip()]
    other_ids = [x.strip() for x in args.other_ids.split(",") if x.strip()]

    # Baseline: 가장 최근 base-id JSON (run_id 무관, meta.id=base_id)
    base_prefix = f"tcn_candidate_validation_{args.symbol}_{args.timeframe}_{args.base_id}"
    base_candidates = sorted(
        DIAG.glob(f"{base_prefix}_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    base_path = None
    for c in base_candidates:
        d = load_json(c)
        if d and d.get("meta", {}).get("id") == args.base_id:
            base_path = c
            break
    if not base_path or not base_path.exists():
        print(f"Baseline JSON not found (id={args.base_id}).", file=sys.stderr)
        return 1

    d_base = load_json(base_path)
    if not d_base:
        return 1
    r_base = d_base.get("results", [])
    row_b365 = get_row(r_base, 365)
    if not row_b365:
        print("Baseline has no 365d result.", file=sys.stderr)
        return 1
    cost_b365 = row_b365.get("cost_on_return")
    mdd_b365 = row_b365.get("max_drawdown")

    print("=" * 60)
    print(f"Model version compare (pinned end_date={args.pinned_end_date})")
    print("=" * 60)
    print(f"base: {args.base_id}  cost_on_365d={cost_b365}  MDD_365d={mdd_b365}")
    print()

    w = 14
    cols = ["base"] + other_ids
    print(f"{'metric':<20} " + " ".join(f"{c:>{w}}" for c in cols))
    print("-" * (20 + (w + 1) * len(cols)))

    for days in days_list:
        row_b = get_row(r_base, days)
        vals = [row_b.get("cost_on_return") if row_b else None]
        for oid in other_ids:
            path = find_json(args.symbol, args.timeframe, oid, oid)
            if path:
                d = load_json(path)
                row = get_row(d.get("results", []), days) if d else None
                vals.append(row.get("cost_on_return") if row else None)
            else:
                vals.append(None)
        parts = [f"{v:.4f}" if v is not None else "N/A" for v in vals]
        print(f"cost_on_{days}d          " + " ".join(f"{p:>{w}}" for p in parts))

    for days in days_list:
        row_b = get_row(r_base, days)
        vals = [row_b.get("max_drawdown") if row_b else None]
        for oid in other_ids:
            path = find_json(args.symbol, args.timeframe, oid, oid)
            if path:
                d = load_json(path)
                row = get_row(d.get("results", []), days) if d else None
                vals.append(row.get("max_drawdown") if row else None)
            else:
                vals.append(None)
        parts = [f"{v:.4f}" if v is not None else "N/A" for v in vals]
        print(f"MDD_{days}d             " + " ".join(f"{p:>{w}}" for p in parts))

    print()
    print("판정 (365d): ADOPT_CANDIDATE = cost_on >= base+0.003 AND MDD <= base+0.005")
    for oid in other_ids:
        path = find_json(args.symbol, args.timeframe, oid, oid)
        if not path:
            print(f"  {oid}: (JSON not found)")
            continue
        d = load_json(path)
        row = get_row(d.get("results", []), 365) if d else None
        if not row:
            print(f"  {oid}: (no 365d)")
            continue
        cost_o = row.get("cost_on_return")
        mdd_o = row.get("max_drawdown")
        if cost_o is None or cost_b365 is None:
            print(f"  {oid}: REJECT")
            continue
        if cost_o >= cost_b365 + 0.003 and (mdd_o is None or mdd_b365 is None or mdd_o <= mdd_b365 + 0.005):
            print(f"  {oid}: ADOPT_CANDIDATE")
        elif cost_o <= cost_b365 - 0.003 or (mdd_o is not None and mdd_b365 is not None and mdd_o > mdd_b365 + 0.005):
            print(f"  {oid}: REJECT")
        else:
            print(f"  {oid}: NO_IMPROVE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

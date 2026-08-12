#!/usr/bin/env python3
"""
Ops 비교: 기간 pinning된 baseline vs other run(s).
출력: pinned (start_date/end_date) 표시, 30d/365d 표, ops verdict (WARN/FAIL/FLAG).
룰: 30d cost_on 악화 -0.003 이하 -> WARN, -0.006 이하 or 30d MDD +0.003 이상 -> FAIL.
    30d trades -10% 이하 -> WARN (과필터). 365d는 참고만 (365d_WARN: baseline 대비 -0.006 이하).
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
    ap = argparse.ArgumentParser(description="Ops compare with pinned window (WARN/FAIL/FLAG)")
    ap.add_argument("--baseline-run-id", type=str, default="pinned_base", help="Baseline run_id")
    ap.add_argument("--other-run-ids", type=str, default="pinned_tp025_r50", help="Comma-separated run_ids")
    ap.add_argument("--mode", type=str, default="ops", choices=["ops"], help="Compare mode (ops)")
    args = ap.parse_args()

    base_path = find_by_run_id(args.baseline_run_id)
    if not base_path or not base_path.exists():
        print(f"Baseline JSON not found (run_id={args.baseline_run_id}).", file=sys.stderr)
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

    meta_base = d_base.get("meta", {})
    r_base = d_base.get("results", [])
    row_b30 = get_row(r_base, 30)
    row_b365 = get_row(r_base, 365)

    # Pinned window 표시
    end_date = meta_base.get("end_date")
    start_30 = meta_base.get("start_date_30")
    start_365 = meta_base.get("start_date_365")
    pinned_str = "none"
    if end_date:
        pinned_str = f"end_date={end_date}"
        if start_30:
            pinned_str += f", start_date_30={start_30}"
        if start_365:
            pinned_str += f", start_date_365={start_365}"

    print("=" * 72)
    print("Ops compare (pinned window) — 30d / 365d")
    print("=" * 72)
    print(f"pinned: {pinned_str}")
    print(f"baseline: {base_path.name} (run_id={args.baseline_run_id})")
    for i, rid in enumerate(other_ids):
        print(f"  other:  {other_paths[i].name} (run_id={rid})")
    print()

    if not row_b365:
        print("Baseline has no 365d result.", file=sys.stderr)
        return 1

    cost_b30 = row_b30.get("cost_on_return") if row_b30 else None
    cost_b365 = row_b365.get("cost_on_return")
    mdd_b30 = row_b30.get("max_drawdown") if row_b30 else None
    tr_b30 = row_b30.get("trades") if row_b30 else None

    w = 12
    cols = ["30d_base", "365d_base"] + [f"30d_{rid}" for rid in other_ids] + [f"365d_{rid}" for rid in other_ids]
    print(f"{'metric':<20} " + " ".join(f"{c:>{w}}" for c in cols))
    print("-" * (20 + (w + 1) * len(cols)))

    for key, fmt in [
        ("cost_on_return", ".4f"),
        ("cost_off_return", ".4f"),
        ("max_drawdown", ".4f"),
        ("trades", "d"),
    ]:
        vals = [row_b30.get(key) if row_b30 else None, row_b365.get(key)]
        for d in other_data:
            r30 = get_row(d.get("results", []), 30)
            vals.append(r30.get(key) if r30 else None)
        for d in other_data:
            r365 = get_row(d.get("results", []), 365)
            vals.append(r365.get(key) if r365 else None)
        parts = [f"{v:{fmt}}" if v is not None else "N/A" for v in vals]
        print(f"{key:<20} " + " ".join(f"{p:>{w}}" for p in parts))

    # partial_tp_count: 30d row
    vals_pt30 = [row_b30.get("partial_tp_count") if row_b30 else None, None]
    for d in other_data:
        r = get_row(d.get("results", []), 30)
        vals_pt30.append(r.get("partial_tp_count") if r else None)
    for _ in other_data:
        vals_pt30.append(None)
    parts = [f"{v}" if v is not None else "N/A" for v in vals_pt30]
    print(f"partial_tp_count_30d    " + " ".join(f"{p:>{w}}" for p in parts))
    # 365d row
    vals_pt365 = [None, row_b365.get("partial_tp_count") if row_b365 else None]
    for _ in other_data:
        vals_pt365.append(None)
    for d in other_data:
        r = get_row(d.get("results", []), 365)
        vals_pt365.append(r.get("partial_tp_count") if r else None)
    parts = [f"{v}" if v is not None else "N/A" for v in vals_pt365]
    print(f"partial_tp_count_365d   " + " ".join(f"{p:>{w}}" for p in parts))

    # Ops verdict: first other run
    verdict = "OK"
    if other_data and row_b30:
        r30 = get_row(other_data[0].get("results", []), 30)
        r365 = get_row(other_data[0].get("results", []), 365)
        if r30:
            cost_o30 = r30.get("cost_on_return")
            mdd_o30 = r30.get("max_drawdown")
            tr_o30 = r30.get("trades")
            if cost_b30 is not None and cost_o30 is not None:
                delta_cost = cost_o30 - cost_b30
                if delta_cost <= -0.006:
                    verdict = "FAIL"
                elif delta_cost <= -0.003:
                    verdict = "WARN" if verdict == "OK" else verdict
            if mdd_b30 is not None and mdd_o30 is not None and (mdd_o30 - mdd_b30) >= 0.003:
                verdict = "FAIL"
            if tr_b30 and tr_o30 is not None and tr_o30 <= tr_b30 * 0.90:
                verdict = "WARN" if verdict == "OK" else verdict
        if r365 and verdict == "OK" and cost_b365 is not None:
            cost_o365 = r365.get("cost_on_return")
            if cost_o365 is not None and (cost_o365 - cost_b365) <= -0.006:
                verdict = "FLAG"  # 365d 참고 경고

    print()
    print("--- verdict (ops) ---")
    print(verdict)
    return 0 if verdict in ("OK", "WARN", "FLAG") else 1


if __name__ == "__main__":
    sys.exit(main())

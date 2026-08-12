#!/usr/bin/env python3
"""
Phase C2: Baseline (flat-gate OFF) vs Flat-gate sweep runs 비교.
판정: SUCCESS / FAIL / NO-IMPROVE per run.
최종: BEST run_id, max_flat_proba, 365d cost_on, delta vs baseline, ADOPT or KEEP_BASELINE.
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
    ap.add_argument("--baseline-run-id", type=str, default="phase_c2_base", help="Baseline run_id")
    ap.add_argument("--other-run-ids", type=str, default="phase_c2_flatgate_045,phase_c2_flatgate_040,phase_c2_flatgate_035,phase_c2_flatgate_030", help="Comma-separated flat-gate run_ids")
    args = ap.parse_args()

    base_path = find_by_run_id(args.baseline_run_id)
    if not base_path or not base_path.exists():
        print(f"Baseline JSON not found (run_id={args.baseline_run_id}). Run Phase C2 base first.")
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
    tr_b365 = row_b365.get("trades")

    # Table: 30d / 365d for baseline + each other
    print("=" * 72)
    print("Phase C2: Baseline vs Flat-gate sweep — 30d / 365d")
    print("=" * 72)
    print(f"baseline: {base_path.name} (run_id={args.baseline_run_id})")
    for i, rid in enumerate(other_ids):
        print(f"  other:  {other_paths[i].name} (run_id={rid})")
    print()

    cols_30 = ["30d_base"] + [f"30d_{rid.replace('phase_c2_flatgate_', '')}" for rid in other_ids]
    cols_365 = ["365d_base"] + [f"365d_{rid.replace('phase_c2_flatgate_', '')}" for rid in other_ids]
    col_headers = cols_30 + cols_365
    w = 10
    print(f"{'metric':<22} " + " ".join(f"{h:>{w}}" for h in col_headers))
    print("-" * (22 + (w + 1) * len(col_headers)))

    for key, fmt in [
        ("cost_on_return", ".4f"),
        ("cost_off_return", ".4f"),
        ("max_drawdown", ".4f"),
        ("trades", "d"),
    ]:
        vals = []
        vals.append(row_b30.get(key) if row_b30 else None)
        for d in other_data:
            r30 = get_row(d.get("results", []), 30)
            vals.append(r30.get(key) if r30 else None)
        vals.append(row_b365.get(key))
        for d in other_data:
            r365 = get_row(d.get("results", []), 365)
            vals.append(r365.get(key) if r365 else None)
        parts = [f"{v:{fmt}}" if v is not None else "N/A" for v in vals]
        print(f"{key:<22} " + " ".join(f"{p:>{w}}" for p in parts))

    print()
    print("Flat-gate stats (365d):")
    print(f"  baseline: entry_flat_gate=off")
    for i, d in enumerate(other_data):
        r365 = get_row(d.get("results", []), 365)
        meta = d.get("meta", {})
        pct = r365.get("pct_flat_gate_block") if r365 else None
        blocked = r365.get("entries_blocked_by_flat_gate") if r365 else None
        mfp = meta.get("max_flat_proba")
        print(f"  {other_ids[i]}: max_flat_proba={mfp}, pct_flat_gate_block={pct}, entries_blocked_by_flat_gate={blocked}")

    # Verdict per other run
    print()
    verdicts = {}
    for i, rid in enumerate(other_ids):
        d = other_data[i]
        r30 = get_row(d.get("results", []), 30)
        r365 = get_row(d.get("results", []), 365)
        if not r365:
            verdicts[rid] = "FAIL"
            print(f"판정 [{rid}]: FAIL (no 365d)")
            continue
        cost_o30 = r30.get("cost_on_return") if r30 else None
        cost_o365 = r365.get("cost_on_return")
        mdd_o365 = r365.get("max_drawdown")
        tr_o365 = r365.get("trades")

        fail = False
        if mdd_b365 is not None and mdd_o365 is not None and mdd_o365 > mdd_b365 + 0.005:
            fail = True
        if tr_b365 and tr_o365 is not None and tr_o365 < tr_b365 * 0.95:
            fail = True
        if cost_b30 is not None and cost_o30 is not None and cost_b30 > 0 and cost_o30 < cost_b30 * 0.8:
            fail = True  # 30d cost_on 20% 이상 악화
        if fail:
            verdicts[rid] = "FAIL"
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
            and tr_b365
            and tr_o365 >= tr_b365 * 0.95
        )
        if success:
            verdicts[rid] = "SUCCESS"
            print(f"판정 [{rid}]: SUCCESS")
        else:
            verdicts[rid] = "NO-IMPROVE"
            print(f"판정 [{rid}]: NO-IMPROVE")

    # Best run (among baseline and others that are not FAIL): max 365d cost_on
    candidates = [("baseline", args.baseline_run_id, None, cost_b365, tr_b365, mdd_b365)]
    for i, rid in enumerate(other_ids):
        if verdicts.get(rid) == "FAIL":
            continue
        r365 = get_row(other_data[i].get("results", []), 365)
        if not r365:
            continue
        c = r365.get("cost_on_return")
        if c is not None:
            mfp = other_data[i].get("meta", {}).get("max_flat_proba")
            candidates.append((rid, rid, mfp, c, r365.get("trades"), r365.get("max_drawdown")))

    best = max(candidates, key=lambda x: x[3] if x[3] is not None else -1e9)
    best_label, best_run_id, best_mfp, best_cost, best_trades, best_mdd = best

    print()
    print("--- 요약 ---")
    print(f"BEST run_id: {best_run_id}")
    print(f"max_flat_proba: {best_mfp if best_mfp is not None else 'N/A (baseline)'}")
    print(f"365d cost_on: {best_cost:.4f}")
    print(f"365d trades: {best_trades}")
    print(f"365d MDD: {best_mdd:.4f}" if best_mdd is not None else "365d MDD: N/A")
    if cost_b365 is not None and best_cost is not None:
        print(f"baseline 대비 delta(cost_on): {best_cost - cost_b365:+.4f}")
    if tr_b365 is not None and best_trades is not None:
        print(f"baseline 대비 delta(trades): {best_trades - tr_b365:+d}")
    if mdd_b365 is not None and best_mdd is not None:
        print(f"baseline 대비 delta(MDD): {best_mdd - mdd_b365:+.4f}")

    if best_run_id == args.baseline_run_id:
        print("최종 판정: KEEP_BASELINE (baseline 유지)")
    else:
        print("최종 판정: ADOPT (채택)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

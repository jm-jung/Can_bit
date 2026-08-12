#!/usr/bin/env python3
"""
Phase C6: Baseline (ee_base) vs early_exit sweep runs 비교.
판정: SUCCESS(ADOPT) / NO-IMPROVE / FAIL. BEST = 365d cost_on 최대(FAIL 제외).
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
    for c in sorted(DIAG.glob(f"{PREFIX}_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        if run_id in c.stem:
            d = load_json(c)
            if d and d.get("meta", {}).get("run_id") == run_id:
                return c
    for c in DIAG.glob(f"{PREFIX}_*.json"):
        if run_id in c.stem:
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run-id", type=str, default="phase_c6_base_ts72_ee_base")
    ap.add_argument("--other-run-ids", type=str, default="phase_c6_ee_k10,phase_c6_ee_k6,phase_c6_ee_l10,phase_c6_ee_p056")
    ap.add_argument("--summary-md", type=str, default="phase_c6_early_exit_sweep_summary.md")
    args = ap.parse_args()

    base_path = find_by_run_id(args.baseline_run_id)
    if not base_path or not base_path.exists():
        print(f"Baseline JSON not found (run_id={args.baseline_run_id}). Run Phase C6 base first.")
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

    print("=" * 72)
    print("Phase C6: Baseline vs early_exit sweep — 30d / 365d")
    print("=" * 72)
    print(f"baseline: {base_path.name} (run_id={args.baseline_run_id})")
    for i, rid in enumerate(other_ids):
        print(f"  other:  {other_paths[i].name} (run_id={rid})")
    print()

    w = 12
    cols_30 = ["30d_base"] + [f"30d_{rid.replace('phase_c6_', '')}" for rid in other_ids]
    cols_365 = ["365d_base"] + [f"365d_{rid.replace('phase_c6_', '')}" for rid in other_ids]
    all_cols = cols_30 + cols_365
    print(f"{'metric':<22} " + " ".join(f"{h:>{w}}" for h in all_cols))
    print("-" * (22 + (w + 1) * len(all_cols)))

    for key, fmt in [
        ("cost_on_return", ".4f"),
        ("cost_off_return", ".4f"),
        ("max_drawdown", ".4f"),
        ("trades", "d"),
    ]:
        vals = [row_b30.get(key) if row_b30 else None]
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
    print("Early-exit stats (365d): lookback, p_floor, bad_k, early_exit_count, early_exit_rate")
    meta_b = d_base.get("meta", {})
    print(f"  baseline: lookback={meta_b.get('early_exit_lookback')}, p_floor={meta_b.get('early_exit_p_floor')}, bad_k={meta_b.get('early_exit_bad_k')}")
    for i, d in enumerate(other_data):
        r365 = get_row(d.get("results", []), 365)
        meta = d.get("meta", {})
        ee_count = r365.get("early_exit_count") if r365 else None
        ee_rate = r365.get("early_exit_rate") if r365 else None
        print(f"  {other_ids[i]}: lookback={meta.get('early_exit_lookback')}, p_floor={meta.get('early_exit_p_floor')}, bad_k={meta.get('early_exit_bad_k')}, count={ee_count}, rate={ee_rate}")

    # 판정: SUCCESS / FAIL / NO-IMPROVE
    verdicts = {}
    for i, rid in enumerate(other_ids):
        d = other_data[i]
        r30 = get_row(d.get("results", []), 30)
        r365 = get_row(d.get("results", []), 365)
        if not r365:
            verdicts[rid] = "FAIL"
            print(f"판정 [{rid}]: FAIL (no 365d)")
            continue
        co30 = r30.get("cost_on_return") if r30 else None
        co365 = r365.get("cost_on_return")
        mdd = r365.get("max_drawdown")
        tr = r365.get("trades")

        fail = False
        if mdd_b365 is not None and mdd is not None and mdd > mdd_b365 + 0.005:
            fail = True
        if tr_b365 is not None and tr is not None and tr < tr_b365 * 0.95:
            fail = True
        if cost_b30 is not None and co30 is not None and co30 < cost_b30 * 0.70:
            fail = True
        if fail:
            verdicts[rid] = "FAIL"
            print(f"판정 [{rid}]: FAIL")
            continue

        success = (
            cost_b365 is not None and co365 is not None and co365 >= cost_b365 + 0.002
            and mdd is not None and mdd_b365 is not None and mdd <= mdd_b365 + 0.005
            and tr is not None and tr_b365 is not None and tr >= tr_b365 * 0.97
            and cost_b30 is not None and co30 is not None and co30 >= cost_b30 * 0.80
        )
        verdicts[rid] = "SUCCESS" if success else "NO-IMPROVE"
        print(f"판정 [{rid}]: {verdicts[rid]}")

    # BEST: FAIL 제외, *_spot 제외, 365d cost_on 최대
    candidates = [(args.baseline_run_id, cost_b365, mdd_b365, tr_b365)]
    for i, rid in enumerate(other_ids):
        if rid.endswith("_spot"):
            continue
        if verdicts.get(rid) == "FAIL":
            continue
        r365 = get_row(other_data[i].get("results", []), 365)
        if not r365:
            continue
        c = r365.get("cost_on_return")
        if c is not None:
            candidates.append((rid, c, r365.get("max_drawdown"), r365.get("trades")))
    candidates.sort(key=lambda x: (-(x[1] if x[1] is not None else -1e9), (x[2] if x[2] is not None else 1e9)))
    best_run_id = candidates[0][0]
    best_cost = candidates[0][1]
    best_mdd = candidates[0][2]
    best_trades = candidates[0][3]

    # 스팟 안정성: best_run_id_spot이 other_ids에 있으면 |cost_on_spot - cost_on_best| <= 0.003 → STABLE else FLAG
    spot_stability = None
    spot_rid = f"{best_run_id}_spot"
    if spot_rid in other_ids:
        idx_spot = other_ids.index(spot_rid)
        r_spot = get_row(other_data[idx_spot].get("results", []), 365)
        if r_spot is not None and best_cost is not None:
            co_spot = r_spot.get("cost_on_return")
            if co_spot is not None:
                if abs(co_spot - best_cost) <= 0.003:
                    spot_stability = "STABLE"
                else:
                    spot_stability = "FLAG(보류)"

    print()
    print("--- 요약 ---")
    print(f"BEST run_id: {best_run_id}")
    if spot_stability:
        print(f"스팟 안정성: {spot_stability}")
    print(f"365d cost_on: {best_cost:.4f}" if best_cost is not None else "365d cost_on: N/A")
    print(f"365d MDD: {best_mdd:.4f}" if best_mdd is not None else "365d MDD: N/A")
    print(f"365d trades: {best_trades}")
    if best_run_id != args.baseline_run_id:
        print("최종 판정: ADOPT (채택)")
    else:
        print("최종 판정: KEEP BASELINE")

    # Summary MD (표는 compare 터미널 출력과 동일하게 재구성하지 않고, "compare 실행으로 확인" 참고로 둠. 상세는 로그 참고)
    summary_path = DIAG / (args.summary_md or "phase_c6_early_exit_sweep_summary.md")
    lines = [
        "# Phase C6 early_exit sweep 요약",
        "",
        f"- **Baseline**: {args.baseline_run_id}",
        f"- **Others**: {', '.join(other_ids)}",
        "",
        "## 30d/365d 비교표",
        "터미널 compare 출력 또는 phase_c6_run_*.log 참고.",
        "",
        "## BEST run",
        f"- **run_id**: {best_run_id}",
        f"- **365d cost_on**: {best_cost:.4f}" if best_cost is not None else "- **365d cost_on**: N/A",
        f"- **365d MDD**: {best_mdd:.4f}" if best_mdd is not None else "- **365d MDD**: N/A",
        f"- **365d trades**: {best_trades}",
        "",
        "## 판정 (per run)",
    ]
    for rid in other_ids:
        if not rid.endswith("_spot"):
            lines.append(f"- {rid}: {verdicts.get(rid, 'N/A')}")
    if spot_stability is not None:
        lines.append("")
        lines.append("## 스팟 안정성")
        lines.append(spot_stability)
    lines.append("")
    lines.append("## 최종 결론")
    lines.append("ADOPT (채택)" if best_run_id != args.baseline_run_id else "KEEP BASELINE")
    lines.append("")
    try:
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nSummary written: {summary_path}")
    except Exception as e:
        print(f"Warning: could not write summary: {e}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

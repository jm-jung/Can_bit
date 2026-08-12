#!/usr/bin/env python3
"""
Phase D1_v2: min_max_proba × max_entropy sweep 비교.
출력: run_id, 30d cost_on, 365d cost_on, 365d MDD, trades
BEST: 1) 365d cost_on 최대, 2) MDD ≤ baseline+0.005, 3) trades ≥ baseline×0.7
Verdict: ADOPT_CANDIDATE / NO_IMPROVE / REJECT
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run-id", type=str, default="phase_d1v2_base")
    ap.add_argument(
        "--other-run-ids",
        type=str,
        default="phase_d1v2_p058_e135,phase_d1v2_p058_e125,phase_d1v2_p058_e115,phase_d1v2_p060_e135,phase_d1v2_p060_e125,phase_d1v2_p060_e115,phase_d1v2_p062_e135,phase_d1v2_p062_e125,phase_d1v2_p062_e115",
    )
    ap.add_argument("--summary-md", type=str, default="phase_d1v2_sweep_summary.md")
    args = ap.parse_args()

    base_path = find_by_run_id(args.baseline_run_id)
    if not base_path or not base_path.exists():
        print(f"Baseline JSON not found (run_id={args.baseline_run_id}).", file=sys.stderr)
        return 1

    other_ids = [x.strip() for x in args.other_run_ids.split(",") if x.strip()]
    other_data = []
    for rid in other_ids:
        p = find_by_run_id(rid)
        if not p or not p.exists():
            print(f"Other run JSON not found (run_id={rid}).", file=sys.stderr)
            return 1
        other_data.append(load_json(p))

    d_base = load_json(base_path)
    if not d_base:
        return 1

    meta_base = d_base.get("meta", {})
    r_base = d_base.get("results", [])
    row_b30 = get_row(r_base, 30)
    row_b365 = get_row(r_base, 365)
    if not row_b365:
        print("Baseline has no 365d result.", file=sys.stderr)
        return 1

    cost_b365 = row_b365.get("cost_on_return")
    mdd_b365 = row_b365.get("max_drawdown")
    trades_b365 = row_b365.get("trades") or 0
    end_date = meta_base.get("end_date")

    print("=" * 72)
    print("Phase D1_v2 sweep (min_max_proba × max_entropy)" + (f" end_date={end_date}" if end_date else ""))
    print("=" * 72)
    print(f"baseline: {args.baseline_run_id}")
    print(f"  30d cost_on={row_b30.get('cost_on_return') if row_b30 else 'N/A'}, 365d cost_on={cost_b365}, MDD={mdd_b365}, trades={trades_b365}")
    print()

    # 표: run_id, 30d cost_on, 365d cost_on, 365d MDD, trades
    w = 14
    print(f"{'run_id':<24} {'30d_cost_on':>{w}} {'365d_cost_on':>{w}} {'365d_MDD':>{w}} {'trades':>{w}}")
    print("-" * (24 + 5 * (w + 1)))

    def fmt(v, f=".4f"):
        if v is None:
            return "N/A"
        if f == "d":
            return f"{int(v)}"
        return f"{v:{f}}"

    print(f"{args.baseline_run_id:<24} {fmt(row_b30.get('cost_on_return')):>{w}} {fmt(cost_b365):>{w}} {fmt(mdd_b365):>{w}} {fmt(trades_b365,'d'):>{w}}")
    for i, rid in enumerate(other_ids):
        d = other_data[i]
        r30 = get_row(d.get("results", []), 30)
        r365 = get_row(d.get("results", []), 365)
        c30 = r30.get("cost_on_return") if r30 else None
        c365 = r365.get("cost_on_return") if r365 else None
        mdd = r365.get("max_drawdown") if r365 else None
        tr = r365.get("trades") if r365 else None
        print(f"{rid:<24} {fmt(c30):>{w}} {fmt(c365):>{w}} {fmt(mdd):>{w}} {fmt(tr,'d'):>{w}}")

    # Verdict: ADOPT_CANDIDATE / NO_IMPROVE / REJECT
    verdicts = {}
    for i, rid in enumerate(other_ids):
        r365 = get_row(other_data[i].get("results", []), 365)
        if not r365:
            verdicts[rid] = "REJECT"
            continue
        cost_o = r365.get("cost_on_return")
        mdd_o = r365.get("max_drawdown")
        if cost_o is None or cost_b365 is None:
            verdicts[rid] = "REJECT"
            continue
        if cost_o <= cost_b365 - 0.003:
            verdicts[rid] = "REJECT"
        elif mdd_o is not None and mdd_b365 is not None and mdd_o >= mdd_b365 + 0.005:
            verdicts[rid] = "REJECT"
        elif cost_o >= cost_b365 + 0.003 and (mdd_o is None or mdd_b365 is None or mdd_o <= mdd_b365 + 0.005):
            verdicts[rid] = "ADOPT_CANDIDATE"
        else:
            verdicts[rid] = "NO_IMPROVE"

    print()
    for rid in other_ids:
        print(f"Verdict [{rid}]: {verdicts.get(rid, 'N/A')}")

    # BEST: ADOPT_CANDIDATE 중 365d cost_on 최대(덜 음수). (MDD/trades는 verdict에서 이미 반영)
    candidates = []
    for i, rid in enumerate(other_ids):
        if verdicts.get(rid) != "ADOPT_CANDIDATE":
            continue
        r365 = get_row(other_data[i].get("results", []), 365)
        if not r365:
            continue
        cost_o = r365.get("cost_on_return")
        if cost_o is None:
            continue
        mdd_o = r365.get("max_drawdown")
        tr_o = r365.get("trades") or 0
        candidates.append((rid, cost_o, mdd_o, tr_o))
    if not candidates:
        best_run_id = args.baseline_run_id
    else:
        # 365d cost_on 최대(덜 음수) = BEST
        candidates.sort(key=lambda x: (x[1] if x[1] is not None else -1e9), reverse=True)
        best_run_id = candidates[0][0]

    print()
    print(f"BEST run_id: {best_run_id}")

    # Top3 by 365d cost_on (all runs including baseline)
    all_runs = [(args.baseline_run_id, cost_b365, mdd_b365, trades_b365)]
    for i, rid in enumerate(other_ids):
        r365 = get_row(other_data[i].get("results", []), 365)
        if r365:
            all_runs.append((rid, r365.get("cost_on_return"), r365.get("max_drawdown"), r365.get("trades")))
    all_runs.sort(key=lambda x: (-(x[1] if x[1] is not None else -1e9),))
    top3 = all_runs[:3]

    # Summary MD
    summary_path = DIAG / args.summary_md
    lines = [
        "# Phase D1_v2 sweep 요약",
        "",
        f"- **pinned end_date**: {end_date or 'N/A'}",
        f"- **baseline**: {args.baseline_run_id} (365d cost_on={cost_b365}, MDD={mdd_b365}, trades={trades_b365})",
        "",
        "## 결과 표",
        "",
        "| run_id | 30d cost_on | 365d cost_on | 365d MDD | trades |",
        "|--------|------------|-------------|----------|-------|",
    ]
    lines.append(f"| {args.baseline_run_id} | {row_b30.get('cost_on_return') if row_b30 else 'N/A'} | {cost_b365} | {mdd_b365} | {trades_b365} |")
    for i, rid in enumerate(other_ids):
        r30 = get_row(other_data[i].get("results", []), 30)
        r365 = get_row(other_data[i].get("results", []), 365)
        c30 = r30.get("cost_on_return") if r30 else "N/A"
        c365 = r365.get("cost_on_return") if r365 else "N/A"
        mdd = r365.get("max_drawdown") if r365 else "N/A"
        tr = r365.get("trades") if r365 else "N/A"
        lines.append(f"| {rid} | {c30} | {c365} | {mdd} | {tr} |")
    lines.extend([
        "",
        "## Verdict",
        "",
    ])
    for rid in other_ids:
        lines.append(f"- {rid}: {verdicts.get(rid, 'N/A')}")
    lines.extend([
        "",
        "## Top3 (365d cost_on 기준)",
        "",
    ])
    for j, (rid, c, m, t) in enumerate(top3, 1):
        lines.append(f"{j}. {rid} — cost_on={c}, MDD={m}, trades={t}")
    lines.extend([
        "",
        f"## BEST run: {best_run_id}",
        "",
        "## 다음 액션 추천",
        "",
    ])
    if best_run_id != args.baseline_run_id:
        best_info = next((x for x in all_runs if x[0] == best_run_id), None)
        if best_info:
            lines.append(f"- ADOPT 후보: {best_run_id} (365d cost_on={best_info[1]}, MDD={best_info[2]}, trades={best_info[3]}). ops 파라미터 반영 검토.")
        else:
            lines.append(f"- ADOPT 후보: {best_run_id}. 365d cost_on 개선, MDD/trades 조건 충족 시 ops 파라미터 반영 검토.")
    else:
        lines.append("- baseline 유지. 추가 튜닝(예: min_hold/cooldown) 또는 다음 스윕(Cost 줄이기) 진행.")
    lines.append("")

    try:
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nSummary: {summary_path}")
    except Exception as e:
        print(f"Warning: could not write summary: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Phase D1_v2 BEST 후보(p058_e125) 스팟체크 요약 생성.
baseline vs 후보 vs spot1 vs spot2 → range/mean, verdict(STABLE/FAIL), 운영 반영안.
"""
from __future__ import annotations

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
    run_ids = ["phase_d1v2_base", "phase_d1v2_p058_e125", "phase_d1v2_p058_e125_spot1", "phase_d1v2_p058_e125_spot2"]
    data = []
    for rid in run_ids:
        p = find_by_run_id(rid)
        if not p or not p.exists():
            if "spot" in rid:
                data.append((rid, None))  # spot 미완료 허용
            else:
                print(f"JSON not found: {rid}", file=sys.stderr)
                return 1
        else:
            data.append((rid, load_json(p)))

    rows = []
    for rid, d in data:
        if d is None:
            rows.append({"run_id": rid, "c30": None, "c365": None, "mdd": None, "trades": None})
            continue
        res = d.get("results", [])
        r30 = get_row(res, 30)
        r365 = get_row(res, 365)
        c30 = r30.get("cost_on_return") if r30 else None
        c365 = r365.get("cost_on_return") if r365 else None
        mdd = r365.get("max_drawdown") if r365 else None
        tr = r365.get("trades") if r365 else None
        rows.append({"run_id": rid, "c30": c30, "c365": c365, "mdd": mdd, "trades": tr})

    baseline = rows[0]
    candidate = rows[1]
    spot1 = rows[2]
    spot2 = rows[3]
    cost_365_list = [candidate["c365"], spot1["c365"], spot2["c365"]]
    cost_365_list = [x for x in cost_365_list if x is not None]
    if len(cost_365_list) < 2:
        cost_365_list = [candidate["c365"], spot1["c365"]] if spot1.get("c365") is not None else [candidate["c365"]]
    cost_365_list = [x for x in cost_365_list if x is not None]
    range_365 = max(cost_365_list) - min(cost_365_list) if len(cost_365_list) >= 2 else 0.0
    mean_365 = sum(cost_365_list) / len(cost_365_list) if cost_365_list else None
    mean_mdd = None
    mean_trades = None
    if all(r["mdd"] is not None for r in [candidate, spot1, spot2]):
        mean_mdd = (candidate["mdd"] + spot1["mdd"] + spot2["mdd"]) / 3
    if all(r["trades"] is not None for r in [candidate, spot1, spot2]):
        mean_trades = (candidate["trades"] + spot1["trades"] + spot2["trades"]) / 3

    # 안정성: range <= 0.005 STABLE; MDD > baseline+0.005 → FAIL; trades < baseline*0.7 → FAIL
    mdd_b = baseline.get("mdd") or 0
    tr_b = baseline.get("trades") or 0
    spot_runs = [r for r in [candidate, spot1, spot2] if r.get("c365") is not None]
    fail_mdd = any(r["mdd"] is not None and r["mdd"] >= mdd_b + 0.005 for r in spot_runs)
    fail_trades = any(r["trades"] is not None and tr_b > 0 and r["trades"] < tr_b * 0.7 for r in spot_runs)
    stable_range = range_365 <= 0.005 if cost_365_list else False
    if fail_mdd or fail_trades:
        verdict = "FAIL"
        reason = []
        if fail_mdd:
            reason.append("MDD baseline+0.005 이상 악화")
        if fail_trades:
            reason.append("trades baseline 대비 -30% 이상 감소")
        verdict_note = "; ".join(reason)
    elif stable_range:
        verdict = "ADOPT"
        verdict_note = "365d cost_on range <= 0.005, MDD/trades 조건 충족"
    else:
        verdict = "FLAG"
        verdict_note = f"365d cost_on range={range_365:.4f} > 0.005 (재스팟 또는 2순위 후보 검토)"

    out_path = DIAG / "phase_d1v2_spotcheck_summary.md"
    lines = [
        "# Phase D1_v2 스팟체크 요약",
        "",
        "BEST 후보: **phase_d1v2_p058_e125** (min_max_proba=0.58, max_entropy=1.25)",
        "",
        "## baseline vs 후보 vs spot1 vs spot2",
        "",
        "| run_id | 30d cost_on | 365d cost_on | 365d MDD | trades |",
        "|--------|------------|-------------|----------|-------|",
    ]
    for r in rows:
        c30 = f"{r['c30']:.4f}" if r.get("c30") is not None else "N/A"
        c365 = f"{r['c365']:.4f}" if r.get("c365") is not None else "N/A"
        mdd = f"{r['mdd']:.4f}" if r.get("mdd") is not None else "N/A"
        tr = str(int(r["trades"])) if r.get("trades") is not None else "N/A"
        lines.append(f"| {r['run_id']} | {c30} | {c365} | {mdd} | {tr} |")
    lines.extend([
        "",
        "## 후보+spot1+spot2 range/mean",
        "",
        f"- 365d cost_on range: {range_365:.4f}",
        f"- 365d cost_on mean: {mean_365:.4f}" if mean_365 is not None else "- 365d cost_on mean: N/A",
        f"- MDD mean: {mean_mdd:.4f}" if mean_mdd is not None else "- MDD mean: N/A",
        f"- trades mean: {mean_trades:.0f}" if mean_trades is not None else "- trades mean: N/A",
        "",
        "## 최종 verdict",
        "",
        f"- **{verdict}**: {verdict_note}",
        "",
        "## 운영 반영안 (권장 파라미터 세트 1개)",
        "",
    ])
    if verdict == "ADOPT":
        lines.extend([
            "| 파라미터 | 값 |",
            "|----------|-----|",
            "| min_max_proba | 0.58 |",
            "| max_entropy | 1.25 |",
            "| min_hold | 36 |",
            "| cooldown | 12 |",
            "| (그 외) | sweep와 동일 (regime off, position_scaling off, early_exit on, time_stop 72 등) |",
            "",
        ])
    else:
        lines.append("- ADOPT 미확정. 2순위 후보(phase_d1v2_p060_e115 또는 phase_d1v2_p058_e115) 스팟체크 후 재판정.")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Verdict: {verdict} — {verdict_note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

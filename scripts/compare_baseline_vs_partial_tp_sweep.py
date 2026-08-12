#!/usr/bin/env python3
"""
Phase C7: Baseline (partial_tp OFF) vs Partial TP sweep runs 비교.
판정: SUCCESS / ADOPT_CANDIDATE / FAIL / NO-IMPROVE. Best run (cost_on 우선, tie면 MDD).
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
    ap.add_argument("--baseline-run-id", type=str, default="phase_c7_base", help="Baseline run_id (partial_tp=off)")
    ap.add_argument("--other-run-ids", type=str,
                    default="phase_c7_tp025_r50,phase_c7_tp030_r50,phase_c7_tp040_r50,phase_c7_tp030_r33",
                    help="Comma-separated partial TP run_ids")
    ap.add_argument("--summary-md", type=str, default="phase_c7_partial_tp_summary.md", help="Summary MD filename")
    args = ap.parse_args()

    base_path = find_by_run_id(args.baseline_run_id)
    if not base_path or not base_path.exists():
        print(f"Baseline JSON not found (run_id={args.baseline_run_id}). Run Phase C7 base first.")
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
    print("Phase C7: Baseline vs Partial TP sweep — 30d / 365d")
    print("=" * 72)
    print(f"baseline: {base_path.name} (run_id={args.baseline_run_id})")
    for i, rid in enumerate(other_ids):
        print(f"  other:  {other_paths[i].name} (run_id={rid})")
    print()

    cols_30 = ["30d_base"] + [f"30d_{rid.replace('phase_c7_', '')}" for rid in other_ids]
    cols_365 = ["365d_base"] + [f"365d_{rid.replace('phase_c7_', '')}" for rid in other_ids]
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
    print("Partial TP stats (365d):")
    meta_base = d_base.get("meta", {})
    print(f"  baseline: partial_tp=off")
    for i, d in enumerate(other_data):
        r365 = get_row(d.get("results", []), 365)
        meta = d.get("meta", {})
        pt_count = r365.get("partial_tp_count") if r365 else None
        pct_pt = r365.get("pct_partial_tp") if r365 else None
        pt_avg = r365.get("partial_tp_avg_pnl") if r365 else None
        th = meta.get("partial_tp_threshold")
        ratio = meta.get("partial_tp_ratio")
        print(f"  {other_ids[i]}: threshold={th}, ratio={ratio}, partial_tp_count={pt_count}, pct_partial_tp={pct_pt}, partial_tp_avg_pnl={pt_avg}")

    print()
    print("Exit reason breakdown (365d): time_stop / early_exit / normal_exit (count, avg_pnl)")
    def _fmt_counts(counts):
        if not counts:
            return "N/A"
        return ", ".join(f"{k}={v}" for k, v in sorted((counts or {}).items()))
    def _fmt_avg_pnl(avg_pnl):
        if not avg_pnl:
            return "N/A"
        return ", ".join(f"{k}={v:.4f}" if v is not None else f"{k}=N/A" for k, v in sorted((avg_pnl or {}).items()))
    r365_base = row_b365
    print(f"  baseline: counts={_fmt_counts(r365_base.get('exit_reason_counts'))}, avg_pnl={_fmt_avg_pnl(r365_base.get('exit_reason_avg_pnl'))}")
    for i, d in enumerate(other_data):
        r365 = get_row(d.get("results", []), 365)
        print(f"  {other_ids[i]}: counts={_fmt_counts(r365.get('exit_reason_counts') if r365 else None)}, avg_pnl={_fmt_avg_pnl(r365.get('exit_reason_avg_pnl') if r365 else None)}")

    print()
    verdicts = {}
    for i, rid in enumerate(other_ids):
        d = other_data[i]
        r365 = get_row(d.get("results", []), 365)
        r30 = get_row(d.get("results", []), 30)
        if not r365:
            verdicts[rid] = "FAIL"
            print(f"판정 [{rid}]: FAIL (no 365d)")
            continue
        cost_o365 = r365.get("cost_on_return")
        cost_o30 = r30.get("cost_on_return") if r30 else None
        mdd_o365 = r365.get("max_drawdown")
        tr_o365 = r365.get("trades")

        # FAIL: MDD +0.005 초과 악화 OR trades -5% 초과 감소 OR 30d cost_on 20% 이상 악화
        fail = False
        if mdd_b365 is not None and mdd_o365 is not None and mdd_o365 > mdd_b365 + 0.005:
            fail = True
        if tr_b365 and tr_o365 is not None and tr_o365 < tr_b365 * 0.95:
            fail = True
        if cost_b30 is not None and cost_o30 is not None and cost_b30 > 0 and cost_o30 < cost_b30 * 0.80:
            fail = True  # 30d cost_on 20% 이상 악화
        if fail:
            verdicts[rid] = "FAIL"
            print(f"판정 [{rid}]: FAIL")
            continue

        # SUCCESS: 365d cost_on >= baseline + 0.003 AND MDD/trades 조건 만족 (이미 위에서 fail 아님)
        success = (
            cost_o365 is not None
            and cost_b365 is not None
            and cost_o365 >= cost_b365 + 0.003
        )
        # ADOPT_CANDIDATE: 365d cost_on 개선(>0) and MDD/trades 조건 만족
        adopt_candidate = (
            cost_o365 is not None
            and cost_b365 is not None
            and cost_o365 > cost_b365
        )
        if success:
            verdicts[rid] = "SUCCESS"
        elif adopt_candidate:
            verdicts[rid] = "ADOPT_CANDIDATE"
        else:
            verdicts[rid] = "NO-IMPROVE"
        print(f"판정 [{rid}]: {verdicts[rid]}")

    # Best: cost_on 우선, tie면 MDD 낮은 쪽
    candidates = [(args.baseline_run_id, cost_b365, mdd_b365, tr_b365)]
    for i, rid in enumerate(other_ids):
        if verdicts.get(rid) == "FAIL":
            continue
        r365 = get_row(other_data[i].get("results", []), 365)
        if not r365:
            continue
        c = r365.get("cost_on_return")
        if c is not None:
            candidates.append((rid, c, r365.get("max_drawdown"), r365.get("trades")))
    candidates.sort(key=lambda x: (-(x[1] if x[1] is not None else -1e9), (x[2] if x[2] is not None else 1e9)))
    best_run_id, best_cost, best_mdd, best_trades = candidates[0]

    best_row_30 = None
    best_row_365 = None
    if best_run_id == args.baseline_run_id:
        best_row_30, best_row_365 = row_b30, row_b365
    else:
        for i, rid in enumerate(other_ids):
            if rid == best_run_id:
                best_row_30 = get_row(other_data[i].get("results", []), 30)
                best_row_365 = get_row(other_data[i].get("results", []), 365)
                break

    print()
    print("--- 요약 ---")
    print(f"BEST run_id: {best_run_id}")
    if best_row_30:
        c30 = best_row_30.get("cost_on_return")
        m30 = best_row_30.get("max_drawdown")
        t30 = best_row_30.get("trades")
        print(f"30d (short-term): cost_on={c30:.4f}, MDD={m30:.4f}, trades={t30}" if c30 is not None and m30 is not None else f"30d: cost_on={c30}, MDD={m30}, trades={t30}")
    else:
        print("30d: N/A")
    print(f"365d cost_on: {best_cost:.4f}" if best_cost is not None else "365d cost_on: N/A")
    print(f"365d trades: {best_trades}")
    print(f"365d MDD: {best_mdd:.4f}" if best_mdd is not None else "365d MDD: N/A")
    if cost_b365 is not None and best_cost is not None:
        print(f"baseline 대비 delta(cost_on): {best_cost - cost_b365:+.4f}")
    if mdd_b365 is not None and best_mdd is not None:
        print(f"baseline 대비 delta(MDD): {best_mdd - mdd_b365:+.4f}")
    if tr_b365 is not None and best_trades is not None:
        print(f"baseline 대비 delta(trades): {best_trades - tr_b365:+d}")
    if best_run_id == args.baseline_run_id:
        print("최종 판정: KEEP_BASELINE (baseline 유지)")
    else:
        v = verdicts.get(best_run_id, "NO-IMPROVE")
        print(f"최종 판정: {v} (채택 후보 이상이면 ADOPT 검토)")

    # Partial TP stats for best
    if best_row_365:
        pt_count = best_row_365.get("partial_tp_count")
        pct_pt = best_row_365.get("pct_partial_tp")
        pt_avg = best_row_365.get("partial_tp_avg_pnl")
        if pt_count is not None or pct_pt is not None:
            print(f"Partial TP (365d): count={pt_count}, pct={pct_pt}, avg_pnl={pt_avg}")

    # Summary MD
    summary_name = (args.summary_md or "phase_c7_partial_tp_summary.md").strip()
    if not summary_name:
        summary_name = "phase_c7_partial_tp_summary.md"
    summary_path = DIAG / summary_name
    lines = [
        "# Phase C7 Partial Take-Profit sweep 요약",
        "",
        f"- **Baseline run_id**: {args.baseline_run_id}",
        f"- **Other runs**: {', '.join(other_ids)}",
        "",
        "## Best run",
        f"- **run_id**: {best_run_id}",
    ]
    if best_run_id != args.baseline_run_id:
        for i, rid in enumerate(other_ids):
            if rid == best_run_id:
                meta = other_data[i].get("meta", {})
                lines.append(f"- **partial_tp_threshold**: {meta.get('partial_tp_threshold')}")
                lines.append(f"- **partial_tp_ratio**: {meta.get('partial_tp_ratio')}")
                break
    lines.extend(["", "### 30d (short-term)",])
    if best_row_30:
        c30 = best_row_30.get("cost_on_return")
        m30 = best_row_30.get("max_drawdown")
        t30 = best_row_30.get("trades")
        lines.append(f"- 30d cost_on={c30:.4f}, 30d MDD={m30:.4f}, 30d trades={t30}")
    else:
        lines.append("- N/A")
    lines.extend([
        "",
        "### 365d",
        f"- **365d cost_on**: {best_cost:.4f}" if best_cost is not None else "- **365d cost_on**: N/A",
        f"- **365d MDD**: {best_mdd:.4f}" if best_mdd is not None else "- **365d MDD**: N/A",
        f"- **365d trades**: {best_trades}",
        "",
        "## Baseline 대비 delta",
        f"- delta(cost_on): {best_cost - cost_b365:+.4f}" if cost_b365 is not None and best_cost is not None else "- delta(cost_on): N/A",
        f"- delta(MDD): {best_mdd - mdd_b365:+.4f}" if mdd_b365 is not None and best_mdd is not None else "- delta(MDD): N/A",
        f"- delta(trades): {best_trades - tr_b365:+d}" if tr_b365 is not None and best_trades is not None else "- delta(trades): N/A",
        "",
        "## Partial TP stats (365d, best run)",
    ])
    if best_row_365:
        lines.append(f"- partial_tp_count: {best_row_365.get('partial_tp_count')}")
        lines.append(f"- pct_partial_tp: {best_row_365.get('pct_partial_tp')}")
        lines.append(f"- partial_tp_avg_pnl: {best_row_365.get('partial_tp_avg_pnl')}")
    else:
        lines.append("- N/A")
    lines.extend([
        "",
        "## Exit reason breakdown (365d, best run)",
    ])
    if best_row_365:
        lines.append(f"- counts: {best_row_365.get('exit_reason_counts')}")
        lines.append(f"- avg_pnl: {best_row_365.get('exit_reason_avg_pnl')}")
    else:
        lines.append("- N/A")
    # Spotcheck stability: best + spot1 + spot2 의 365d cost_on range
    spot_verdict = ""
    if "phase_c7_best_spot1" in other_ids and "phase_c7_best_spot2" in other_ids:
        cost_vals = []
        if best_run_id == args.baseline_run_id:
            cost_vals.append(best_cost)
        else:
            cost_vals.append(best_cost)
        for rid in ("phase_c7_best_spot1", "phase_c7_best_spot2"):
            for i, oid in enumerate(other_ids):
                if oid == rid:
                    r365 = get_row(other_data[i].get("results", []), 365)
                    if r365 and r365.get("cost_on_return") is not None:
                        cost_vals.append(r365["cost_on_return"])
                    break
        if len(cost_vals) >= 2:
            rng = max(cost_vals) - min(cost_vals)
            spot_verdict = "STABLE (ADOPT 확정 가능)" if rng <= 0.005 else "FLAG (보류, baseline 유지)"
            spot_verdict = f" 365d cost_on range={rng:.4f} → {spot_verdict}"
        else:
            spot_verdict = " (데이터 부족)"
    lines.extend([
        "",
        "## 스팟체크",
        "- BEST와 동일 설정으로 phase_c7_best_spot1, phase_c7_best_spot2 2회 실행." + spot_verdict,
        "",
        "## 판정",
        f"- **BEST verdict**: {verdicts.get(best_run_id, 'N/A')}",
        f"- **최종**: " + ("KEEP_BASELINE" if best_run_id == args.baseline_run_id else "ADOPT_CANDIDATE 이상 시 스팟체크 후 ADOPT 확정 가능"),
        "",
    ])
    try:
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nSummary written: {summary_path}")
    except Exception as e:
        print(f"Warning: could not write summary: {e}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

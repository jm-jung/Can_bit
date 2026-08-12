#!/usr/bin/env python3
"""
Phase별 검증 결과 비교: diagnostics JSON을 읽어 30d/365d cost_on, MDD, trades 등 비교표 출력.
로드맵: docs/roadmap_model_improvement.md

사용법:
  python -m scripts.compare_phase_results.py --phase A
  python -m scripts.compare_phase_results.py --phase A --baseline path/to/phase_a_baseline.json
  python -m scripts.compare_phase_results.py --list   # phase run_id 목록만 출력
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG = PROJECT_ROOT / "data" / "diagnostics"
PREFIX = "tcn_candidate_validation_BTCUSDT_5m_h15_t0p004"

# Phase별 run_id 패턴 (비교 대상)
PHASE_RUN_IDS = {
    "A": ["phase_a_baseline", "phase_a_commission_0012"],
    "B": ["phase_b_placeholder", "phase_b_temp_1p2"],
    "C": ["phase_c_placeholder", "phase_c_entropy_1p25"],
    "D": ["phase_d_placeholder", "phase_d_v2"],
    "E": ["phase_e_placeholder", "phase_e_partial_tp"],
}


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


def find_jsons_by_run_ids(run_ids: list[str]) -> list[tuple[str, Path]]:
    """(run_id, path) 리스트. run_id는 파일명에서 추출 (PREFIX_ 제거)."""
    out = []
    for rid in run_ids:
        path = DIAG / f"{PREFIX}_{rid}.json"
        if path.exists():
            out.append((rid, path))
        else:
            # 타임스탬프 붙은 것 검색 (예: phase_a_baseline_20260228_120000)
            for f in DIAG.glob(f"{PREFIX}_{rid}_*.json"):
                out.append((rid, f))
                break
            else:
                out.append((rid, path))  # 없어도 표에 N/A로 표시
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=str, choices=["A", "B", "C", "D", "E"], help="Phase to compare")
    ap.add_argument("--baseline", type=Path, default=None, help="Override baseline JSON path")
    ap.add_argument("--list", action="store_true", help="List phase run_id patterns and exit")
    args = ap.parse_args()

    if args.list:
        for phase, rids in PHASE_RUN_IDS.items():
            print(f"Phase {phase}: {rids}")
        return 0

    if not args.phase:
        ap.print_help()
        return 0

    run_ids = PHASE_RUN_IDS[args.phase]
    pairs = find_jsons_by_run_ids(run_ids)
    baseline_path = args.baseline
    if not baseline_path and pairs:
        baseline_path = pairs[0][1]  # 첫 번째를 baseline으로

    if not baseline_path or not baseline_path.exists():
        print(f"Baseline not found. Run phase with run_id {run_ids[0]} first.")
        return 1

    d_base = load_json(baseline_path)
    if not d_base:
        return 1
    r_base = d_base.get("results", [])
    row_b30 = get_row(r_base, 30)
    row_b365 = get_row(r_base, 365)
    cost_b365 = row_b365.get("cost_on_return") if row_b365 else None
    mdd_b365 = row_b365.get("max_drawdown") if row_b365 else None
    tr_b365 = row_b365.get("trades") if row_b365 else None

    print("=" * 80)
    print(f"Phase {args.phase} — 30d / 365d comparison (baseline: {baseline_path.name})")
    print("=" * 80)
    print(f"{'run_id':<32} {'cost_on_30':>12} {'cost_on_365':>12} {'mdd_30':>10} {'mdd_365':>10} {'trades_30':>10} {'trades_365':>10} {'vs_baseline':>12}")
    print("-" * 80)

    for i, (rid, path) in enumerate(pairs):
        if not path.exists():
            print(f"{rid:<32} {'N/A':>12} {'N/A':>12} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'(missing)':>12}")
            continue
        d = load_json(path)
        if not d:
            print(f"{rid:<32} {'(load err)':>12}")
            continue
        r = d.get("results", [])
        row30 = get_row(r, 30)
        row365 = get_row(r, 365)
        co30 = row30.get("cost_on_return") if row30 else None
        co365 = row365.get("cost_on_return") if row365 else None
        m30 = row30.get("max_drawdown") if row30 else None
        m365 = row365.get("max_drawdown") if row365 else None
        t30 = row30.get("trades") if row30 else None
        t365 = row365.get("trades") if row365 else None

        vs = "baseline" if path == baseline_path else ""
        if vs == "" and cost_b365 is not None and co365 is not None:
            delta = co365 - cost_b365
            if delta >= 0.003:
                vs = "OK +" + f"{delta:.3f}"
            elif delta <= -0.003:
                vs = "FAIL " + f"{delta:.3f}"
            else:
                vs = f"{delta:+.3f}"
        if mdd_b365 is not None and m365 is not None and (m365 - mdd_b365) > 0.005:
            vs = "FAIL MDD"
        if tr_b365 and t365 is not None and (tr_b365 - t365) / tr_b365 > 0.05:
            vs = "FAIL trades"

        co30s = f"{co30:.4f}" if co30 is not None else "N/A"
        co365s = f"{co365:.4f}" if co365 is not None else "N/A"
        m30s = f"{m30:.4f}" if m30 is not None else "N/A"
        m365s = f"{m365:.4f}" if m365 is not None else "N/A"
        t30s = str(t30) if t30 is not None else "N/A"
        t365s = str(t365) if t365 is not None else "N/A"
        print(f"{rid:<32} {co30s:>12} {co365s:>12} {m30s:>10} {m365s:>10} {t30s:>10} {t365s:>10} {vs:>12}")

    print()
    print("Success (로드맵 기준): 365d cost_on +0.003 vs baseline, MDD +0.005 미만 악화, trades -5% 미만 감소, 30d <20% 악화")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

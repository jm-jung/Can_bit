#!/usr/bin/env python3
"""
TCN filter sweep JSON(30일 full 권장)을 읽어 (4) 결과 분석 + (5) quick 재설계 축약 그리드 제안을 출력.
사용: python -m scripts.analyze_filter_sweep_result [경로]
  경로 생략 시 data/diagnostics/ 에서 최신 tcn_filter_sweep_top2_*.json 사용.
  meta.days==30 이어야 전체 분석 수행; 아니면 안내만 출력.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG = PROJECT_ROOT / "data" / "diagnostics"


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        candidates = sorted(DIAG.glob("tcn_filter_sweep_top2_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("data/diagnostics/tcn_filter_sweep_top2_*.json 없음", file=sys.stderr)
            return 1
        path = candidates[0]
    path = Path(path)
    if not path.exists():
        print(f"파일 없음: {path}", file=sys.stderr)
        return 1

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    days = meta.get("days")
    quick = meta.get("quick", True)
    combos = meta.get("combos_per_id", 0)

    if days != 30:
        print(f"[안내] 선택 파일은 days={days} 입니다. 30일 full sweep 결과로 분석하려면 30일 실행 후 생성된 JSON 경로를 넘겨주세요.")
        print(f"  예: python -m scripts.analyze_filter_sweep_result data/diagnostics/tcn_filter_sweep_top2_YYYYMMDD.json")
        return 0

    verdict = data.get("verdict", {})
    best_one = verdict.get("best_one")
    runner_up = verdict.get("runner_up")
    cost_achieved = verdict.get("cost_on_return_ge_zero_achieved", False)
    num_ge_zero = verdict.get("num_combos_ge_zero", 0)

    top10_per_id = data.get("top10_per_id", {})

    print("=" * 60)
    print("TCN Top2 Filter Sweep — 30일 Full 결과 분석")
    print("=" * 60)
    print(f"파일: {path} | quick={quick} combos_per_id={combos}\n")

    # (4-1) best_one 요약
    print("--- (4-1) best_one 요약 ---")
    if best_one:
        b = best_one
        print(f"id={b.get('id')} min_max_proba={b.get('min_max_proba')} max_entropy={b.get('max_entropy')} min_hold={b.get('min_hold')} cooldown={b.get('cooldown')}")
        print(f"  trades={b.get('trades')} cost_on_return={b.get('cost_on_return')} cost_off_return={b.get('cost_off_return')} max_drawdown={b.get('max_drawdown')}")
        print(f"  cost_on_win_rate={b.get('cost_on_win_rate')} cost_off_win_rate={b.get('cost_off_win_rate')}")
    else:
        print("best_one: 없음")

    if runner_up:
        r = runner_up
        print(f"\nrunner_up: id={r.get('id')} mmp={r.get('min_max_proba')} me={r.get('max_entropy')} mh={r.get('min_hold')} cd={r.get('cooldown')} cost_on={r.get('cost_on_return')} trades={r.get('trades')}")

    # (4-2) 강한 필터 영역 근거 (top10_per_id 기반)
    print("\n--- (4-2) 강한 필터 영역 (ID별 top10 분포) ---")
    for sid, rows in top10_per_id.items():
        if not rows:
            continue
        mmp_vals = [r.get("min_max_proba") for r in rows if r.get("min_max_proba") is not None]
        me_vals = [r.get("max_entropy") for r in rows if r.get("max_entropy") is not None]
        mh_vals = [r.get("min_hold") for r in rows if r.get("min_hold") is not None]
        cd_vals = [r.get("cooldown") for r in rows if r.get("cooldown") is not None]
        trades_vals = [r.get("trades") for r in rows if r.get("trades") is not None]
        n_55 = sum(1 for v in mmp_vals if v is not None and v >= 0.55)
        n_me_low = sum(1 for v in me_vals if v is not None and v <= 1.45)
        n_hold_24 = sum(1 for v in mh_vals if v is not None and v >= 24)
        n_cd_12 = sum(1 for v in cd_vals if v is not None and v >= 12)
        print(f"{sid}: min_max_proba>=0.55 비율 {n_55}/{len(rows)}, max_entropy<=1.45 비율 {n_me_low}/{len(rows)}, min_hold>=24 비율 {n_hold_24}/{len(rows)}, cooldown>=12 비율 {n_cd_12}/{len(rows)}")
        print(f"  min_max_proba 범위: {min(mmp_vals):.2f}~{max(mmp_vals):.2f}, max_entropy: {min(me_vals):.2f}~{max(me_vals):.2f}, min_hold: {min(mh_vals)}~{max(mh_vals)}, cooldown: {min(cd_vals)}~{max(cd_vals)}")
        print(f"  trades 범위: {min(trades_vals)}~{max(trades_vals)}")

    # (4-3) cost_on_return >= 0 조합 여부
    print("\n--- (4-3) cost_on_return >= 0 달성 ---")
    if cost_achieved:
        print(f"달성: {num_ge_zero}개 조합")
    else:
        print("없음.")
        if best_one:
            b = best_one
            print(f"  best cost_on_return={b.get('cost_on_return')}, trades={b.get('trades')}, max_drawdown={b.get('max_drawdown')}")
            print(f"  cost_off_return={b.get('cost_off_return')} — cost_off는 양수인데 cost_on이 음수인 이유: 수수료/슬리피지 부담으로 진입 횟수가 많을 때 수익이 깎임.")

    # (5) quick 재설계 축약 그리드 제안 (3×3×2×2=36)
    print("\n--- (5) quick 재설계용 축약 그리드 제안 (36조합) ---")
    all_top = []
    for sid, rows in top10_per_id.items():
        for r in rows:
            r = dict(r)
            r["_id"] = sid
            all_top.append(r)
    if all_top:
        mmp_set = sorted({r["min_max_proba"] for r in all_top if r.get("min_max_proba") is not None})
        me_set = sorted({r["max_entropy"] for r in all_top if r.get("max_entropy") is not None})
        mh_set = sorted({r["min_hold"] for r in all_top if r.get("min_hold") is not None})
        cd_set = sorted({r["cooldown"] for r in all_top if r.get("cooldown") is not None})
        # 3×3×2×2 골라서 제안
        mmp_rec = (mmp_set[-3:] if len(mmp_set) >= 3 else mmp_set) or [0.50, 0.55, 0.57]
        me_rec = (me_set[:3] if len(me_set) >= 3 else me_set) or [1.35, 1.40, 1.45]
        mh_rec = (mh_set[-2:] if len(mh_set) >= 2 else mh_set) or [24, 36]
        cd_rec = (cd_set[-2:] if len(cd_set) >= 2 else cd_set) or [12, 24]
        print("  min_max_proba (3):", mmp_rec)
        print("  max_entropy (3):", me_rec)
        print("  min_hold (2):", mh_rec)
        print("  cooldown (2):", cd_rec)
        print("  → 3×3×2×2 = 36 조합으로 quick 그리드 재설계 시 위 값대로 QUICK_GRID 수정 권장.")
    else:
        print("  top10 데이터 없음 — full 결과 확인 후 재실행.")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

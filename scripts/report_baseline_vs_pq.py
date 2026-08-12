#!/usr/bin/env python3
"""
Baseline vs PQ 최종 리포트: diagnostics JSON에서 최신 Baseline / 최신 PQ를 찾아
30d·365d 비교표, PQ regime stats, 채택 판정을 출력하고 REPORT_BASELINE_VS_PQ.md 에 저장.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG = PROJECT_ROOT / "data" / "diagnostics"
PREFIX = "tcn_candidate_validation_BTCUSDT_5m_h15_t0p004"
OUT_MD = DIAG / "REPORT_BASELINE_VS_PQ.md"


def load(p: Path) -> dict | None:
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_row(results: list, days: int) -> dict | None:
    for r in results:
        if r.get("days") == days:
            return r
    return None


def find_latest_baseline() -> Path | None:
    """regime != proba_quantile 이며 position_scaling off인 Run A 스타일 최신 JSON 우선."""
    candidates = sorted(DIAG.glob(f"{PREFIX}_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    for c in candidates:
        d = load(c)
        if not d:
            continue
        m = d.get("meta", {})
        if m.get("regime_rule") != "proba_quantile" and m.get("position_scaling") in (None, "off"):
            return c
    for c in candidates:
        d = load(c)
        if d and d.get("meta", {}).get("regime_rule") != "proba_quantile":
            return c
    return None


def find_latest_pq() -> Path | None:
    candidates = sorted(DIAG.glob(f"{PREFIX}_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    for c in candidates:
        d = load(c)
        if d and d.get("meta", {}).get("regime_rule") == "proba_quantile":
            return c
    return None


def main() -> int:
    path_baseline = find_latest_baseline()
    path_pq = find_latest_pq()

    d_base = load(path_baseline) if path_baseline else None
    d_pq = load(path_pq) if path_pq else None
    r_base = d_base.get("results", []) if d_base else []
    r_pq = d_pq.get("results", []) if d_pq else []

    name_b = path_baseline.stem.replace(f"{PREFIX}_", "") if path_baseline else "—"
    name_pq = path_pq.stem.replace(f"{PREFIX}_", "") if path_pq else "—"

    def fmt(v, f=".4f"):
        if v is None:
            return "-"
        if isinstance(v, (int, float)):
            return f"{v:{f}}"
        return str(v)

    lines = [
        "# Baseline vs PQ 최종 리포트",
        "",
        "## Run 목록 및 JSON 파일명",
        "",
        "| Run | JSON 파일명 |",
        "|-----|-------------|",
        f"| Baseline | {path_baseline.name if path_baseline else '(없음)'} |",
        f"| PQ (최신) | {path_pq.name if path_pq else '(없음)'} |",
        "",
        "---",
        "",
        "## 30d / 365d 비교표 (cost_on, cost_off, MDD, trades)",
        "",
        "| 구조 | span | cost_on_return | cost_off_return | max_drawdown | trades |",
        "|------|------|----------------|-----------------|--------------|--------|",
    ]

    for label, rows, path in [
        ("Baseline", r_base, path_baseline),
        ("PQ", r_pq, path_pq),
    ]:
        if not rows:
            lines.append(f"| {label} | - | - | - | - | - |")
            continue
        for days in (30, 365):
            row = get_row(rows, days)
            if not row:
                lines.append(f"| {label} | {days}d | - | - | - | - |")
            else:
                lines.append(
                    f"| {label} | {days}d | {fmt(row.get('cost_on_return'))} | {fmt(row.get('cost_off_return'))} | {fmt(row.get('max_drawdown'))} | {row.get('trades')} |"
                )

    lines.extend([
        "",
        "---",
        "",
        "## PQ Regime stats (365d)",
        "",
    ])
    row_pq_365 = get_row(r_pq, 365) if r_pq else None
    if row_pq_365 is not None:
        lines.append("| pct_proba_quantile_block | entries_blocked_by_regime_proba_quantile | blocked_ratio | q_threshold_mean |")
        lines.append("|--------------------------|------------------------------------------|---------------|-----------------|")
        lines.append(
            f"| {fmt(row_pq_365.get('pct_proba_quantile_block'))} | {row_pq_365.get('entries_blocked_by_regime_proba_quantile')} | {fmt(row_pq_365.get('blocked_ratio'))} | {fmt(row_pq_365.get('q_threshold_mean'))} |"
        )
        summary = d_pq.get("summary", {}) if d_pq else {}
        lines.append("")
        lines.append(f"- **overblock_warning**: {summary.get('overblock_warning')}")
        lines.append(f"- **overblock_nogo**: {summary.get('overblock_nogo')}")
    else:
        lines.append("PQ 365d 결과 없음.")

    lines.extend([
        "",
        "---",
        "",
        "## 결론 (채택안)",
        "",
    ])

    base_365 = get_row(r_base, 365) if r_base else None
    cost_on_b = base_365.get("cost_on_return") if base_365 else None
    mdd_b = base_365.get("max_drawdown") if base_365 else None
    cost_on_pq = row_pq_365.get("cost_on_return") if row_pq_365 else None
    mdd_pq = row_pq_365.get("max_drawdown") if row_pq_365 else None
    blocked = (row_pq_365.get("blocked_ratio") or 0) if row_pq_365 else None
    entries_blocked = row_pq_365.get("entries_blocked_by_regime_proba_quantile") or 0 if row_pq_365 else 0
    overblock_nogo = (d_pq.get("summary", {}) or {}).get("overblock_nogo", False) if d_pq else False

    if path_pq is None or row_pq_365 is None:
        lines.append("**PQ 채택 보류.** PQ 결과 JSON 없음. Run B(C) 실행 후 다시 리포트 생성.")
    elif entries_blocked == 0 and (blocked is None or blocked == 0):
        meta_pq = (d_pq or {}).get("meta", {})
        lines.append(
            "**PQ 채택 보류.** 현재 min_max_proba=0.58(또는 0.56)에서 PQ가 실제로 차단을 발생시키지 않음 (entries_blocked=0). "
            "필터 효과가 아닌 노이즈 가능성."
        )
        lines.append("")
        lines.append("**다음 액션 (대안):**")
        lines.append("- (a) min_max_proba 0.55~0.57 구간에서 PQ 1~2회 추가 실행 후 blocked > 0 확인")
        lines.append("- (b) PQ를 ‘필터’가 아니라 position scaling 입력으로만 사용하는 구조 검토")
    elif overblock_nogo:
        lines.append("**PQ 미채택.** overblock_nogo=True (blocked_ratio ≥ 95%). 과차단.")
    elif cost_on_b is not None and cost_on_pq is not None:
        improved = cost_on_pq >= cost_on_b
        within = cost_on_b is not None and cost_on_pq is not None and abs(cost_on_pq - cost_on_b) <= 0.001
        mdd_ok = mdd_pq is not None and mdd_b is not None and mdd_pq <= mdd_b
        if (improved or within) and mdd_ok:
            meta_pq = (d_pq or {}).get("meta", {})
            q = meta_pq.get("q")
            qw = meta_pq.get("q_window")
            mmp = meta_pq.get("min_max_proba")
            lines.append(f"**채택안 = PQ** (min_max_proba={mmp}, q={q}, q_window={qw}). 365d cost_on 기준 개선/유지, MDD 유지/감소, PQ 실제 차단 발생.")
        else:
            lines.append("**채택안 = Baseline.** PQ는 차단 발생하나 365d cost_on 악화 또는 MDD 증가로 미채택.")
    else:
        lines.append("**채택안 = Baseline.** (비교 데이터 부족 또는 PQ 미충족)")

    lines.append("")
    DIAG.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

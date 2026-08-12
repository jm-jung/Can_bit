#!/usr/bin/env python3
"""
Baseline(regime off) / PQ-30d(proba_quantile q_window=8640) / PQ-14d(proba_quantile q_window=4032)
3개 구조를 비교해 data/diagnostics/COMPARE_BASELINE_PQ30D_PQ14D.md 를 생성한다.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG = PROJECT_ROOT / "data" / "diagnostics"
PREFIX = "tcn_candidate_validation_BTCUSDT_5m_h15_t0p004"
OUT_MD = DIAG / "COMPARE_BASELINE_PQ30D_PQ14D.md"


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
    """position_scaling=off, regime not proba_quantile (즉 Baseline = regime_filter off)."""
    candidates = sorted(DIAG.glob(f"{PREFIX}_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    for c in candidates:
        d = load(c)
        if not d:
            continue
        m = d.get("meta", {})
        if m.get("position_scaling") in (None, "off") and m.get("regime_rule") != "proba_quantile":
            return c
    return None


def find_latest_pq_by_window(q_window: int) -> Path | None:
    """regime_rule=proba_quantile, q_window=주어진 값."""
    candidates = sorted(DIAG.glob(f"{PREFIX}_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    for c in candidates:
        d = load(c)
        if not d:
            continue
        m = d.get("meta", {})
        if m.get("regime_rule") == "proba_quantile" and m.get("q_window") == q_window:
            return c
    return None


def main() -> int:
    path_baseline = find_latest_baseline()
    path_pq30d = find_latest_pq_by_window(8640)
    path_pq14d = find_latest_pq_by_window(4032)

    def data(p: Path | None):
        if not p or not p.exists():
            return None, None, "?"
        d = load(p)
        return d, d.get("results", []) if d else [], p.stem.replace(PREFIX + "_", "")

    d_base, r_base, name_base = data(path_baseline)
    d_pq30, r_pq30, name_pq30 = data(path_pq30d)
    d_pq14, r_pq14, name_pq14 = data(path_pq14d)

    lines = [
        "# Baseline / PQ-30d / PQ-14d 종합 비교",
        "",
        "생성: from diagnostics JSON (Baseline = regime off, PQ-30d = proba_quantile q_window=8640, PQ-14d = proba_quantile q_window=4032).",
        "",
        "---",
        "",
        "## 1) 30d / 365d span 결과 표",
        "",
        "| 구조 | span | trades | cost_on_return | cost_off_return | max_drawdown |",
        "|------|------|--------|----------------|-----------------|--------------|",
    ]

    def fmt(v, fmt_str=".4f"):
        if v is None:
            return "-"
        if isinstance(v, (int, float)):
            return f"{v:{fmt_str}}"
        return str(v)

    for label, rows, _ in [
        ("Baseline", r_base, name_base),
        ("PQ-30d (q_window=8640)", r_pq30, name_pq30),
        ("PQ-14d (q_window=4032)", r_pq14, name_pq14),
    ]:
        if not rows:
            lines.append(f"| {label} | (없음) | - | - | - | - |")
            continue
        for days in (30, 365):
            row = get_row(rows, days)
            if not row:
                lines.append(f"| {label} | {days}d | - | - | - | - |")
            else:
                lines.append(
                    f"| {label} | {days}d | {row.get('trades')} | {fmt(row.get('cost_on_return'))} | {fmt(row.get('cost_off_return'))} | {fmt(row.get('max_drawdown'))} |"
                )

    lines.extend([
        "",
        "---",
        "",
        "## 2) Regime stats (365d) — PQ-30d / PQ-14d",
        "",
        "| 구조 | pct_proba_quantile_block | entries_blocked_by_regime_proba_quantile | blocked_ratio | q_threshold_mean |",
        "|------|--------------------------|------------------------------------------|---------------|-----------------|",
    ])

    for label, rows in [("PQ-30d", r_pq30), ("PQ-14d", r_pq14)]:
        if not rows:
            lines.append(f"| {label} | - | - | - | - |")
            continue
        row = get_row(rows, 365)
        if not row:
            lines.append(f"| {label} | - | - | - | - |")
        else:
            lines.append(
                f"| {label} | {fmt(row.get('pct_proba_quantile_block'))} | {row.get('entries_blocked_by_regime_proba_quantile')} | {fmt(row.get('blocked_ratio'))} | {fmt(row.get('q_threshold_mean'))} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3) Baseline / PQ-30d / PQ-14d 비교 요약",
        "",
    ])

    # 365d cost_on 기준 순위, MDD 비교, trades 변화율, overblock
    base_365 = get_row(r_base, 365) if r_base else None
    pq30_365 = get_row(r_pq30, 365) if r_pq30 else None
    pq14_365 = get_row(r_pq14, 365) if r_pq14 else None

    cost_on_b = base_365.get("cost_on_return") if base_365 else None
    cost_on_30 = pq30_365.get("cost_on_return") if pq30_365 else None
    cost_on_14 = pq14_365.get("cost_on_return") if pq14_365 else None

    mdd_b = base_365.get("max_drawdown") if base_365 else None
    mdd_30 = pq30_365.get("max_drawdown") if pq30_365 else None
    mdd_14 = pq14_365.get("max_drawdown") if pq14_365 else None

    trades_b = base_365.get("trades") if base_365 else None
    trades_30 = pq30_365.get("trades") if pq30_365 else None
    trades_14 = pq14_365.get("trades") if pq14_365 else None

    overblock_30 = (pq30_365.get("blocked_ratio") or 0) >= 0.95 if pq30_365 else None
    overblock_14 = (pq14_365.get("blocked_ratio") or 0) >= 0.95 if pq14_365 else None

    def pct_change(base_val, val):
        if base_val is None or val is None or base_val == 0:
            return "-"
        return f"{(val - base_val) / abs(base_val) * 100:+.1f}%"

    lines.append("| 구조 | 365d cost_on | 365d cost_on 순위 | 365d MDD | MDD vs Baseline | 365d trades | trades 변화율 | overblock(≥95%) |")
    lines.append("|------|-------------|-------------------|----------|------------------|-------------|---------------|-----------------|")

    # 순위: cost_on 높은 순 1,2,3
    candidates = [
        ("Baseline", cost_on_b, mdd_b, trades_b, None),
        ("PQ-30d", cost_on_30, mdd_30, trades_30, overblock_30),
        ("PQ-14d", cost_on_14, mdd_14, trades_14, overblock_14),
    ]
    ranked = sorted(
        [(name, cost_on) for name, cost_on, _mdd, _tr, _ob in candidates if cost_on is not None],
        key=lambda x: x[1],
        reverse=True,
    )
    rank_map = {name: i + 1 for i, (name, _) in enumerate(ranked)}

    for name, cost_on, mdd, trades, overblock in candidates:
        rk = rank_map.get(name, "-")
        mdd_vs = pct_change(mdd_b, mdd) if name != "Baseline" else "-"
        tr_vs = pct_change(trades_b, trades) if name != "Baseline" else "-"
        ob = "YES" if overblock else ("NO" if overblock is False else "-")
        lines.append(f"| {name} | {fmt(cost_on)} | {rk} | {fmt(mdd)} | {mdd_vs} | {trades} | {tr_vs} | {ob} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4) 판정",
        "",
    ])

    baseline_cost = cost_on_b
    if baseline_cost is None:
        lines.append("- Baseline 결과 없음. Baseline 실행 후 재비교.")
    elif cost_on_14 is None:
        lines.append("- PQ-14d 결과 없음. 아래 명령으로 PQ-14d 실행 후 이 스크립트를 다시 실행하세요.")
        lines.append("")
        lines.append("```bash")
        lines.append("python -X faulthandler -m scripts.run_tcn_candidate_validation \\")
        lines.append("  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m \\")
        lines.append("  --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24 \\")
        lines.append("  --regime-filter proba_quantile --q-window 4032 --q 0.95 --p-floor 0.55 \\")
        lines.append("  --position-scaling off \\")
        lines.append("  --days-list 30,365")
        lines.append("```")
    else:
        improved = cost_on_14 >= baseline_cost
        mdd_ok = mdd_14 is not None and mdd_b is not None and mdd_14 <= mdd_b
        if improved and mdd_ok:
            lines.append("**PQ-14d 채택.** 365d cost_on이 Baseline 이상이며 MDD 감소.")
        elif improved:
            lines.append("**PQ-14d 조건부 채택.** 365d cost_on 개선. MDD는 유지/증가 → 확인 필요.")
        else:
            lines.append("**PQ-14d 미채택.** 365d cost_on이 Baseline(-0.0542) 대비 악화. Baseline 또는 PQ-30d 유지 권장.")

    lines.append("")
    DIAG.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

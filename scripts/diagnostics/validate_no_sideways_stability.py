"""
no_sideways 필터 안정성 검증 (다중 기간, baseline 대비).

실행:
  python -m scripts.diagnostics.validate_no_sideways_stability

  python -m scripts.diagnostics.validate_no_sideways_stability \\
    --lookback-list 14,30,60,90 --cooldown-bars 12
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "diagnostics" / "regime"


def _mask_last_days(df: pd.DataFrame, lookback_days: int) -> np.ndarray:
    ts = pd.to_datetime(df["timestamp"])
    end = ts.max()
    start = end - pd.Timedelta(days=int(lookback_days))
    return (ts >= start).values


def _get(result: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _profit_factor(trades: List[Mapping[str, Any]]) -> Optional[float]:
    profits = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    if not profits:
        return None
    wins = sum(p for p in profits if p > 0)
    losses = sum(p for p in profits if p < 0)
    if losses == 0:
        return None if wins == 0 else float("inf")
    return wins / abs(losses)


def _expectancy(trades: List[Mapping[str, Any]]) -> Optional[float]:
    profits = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    if not profits:
        return None
    return float(np.mean(profits))


def _entry_indices_from_trade_events(events: Sequence[Mapping[str, Any]], n_trades: int) -> List[int]:
    """EXIT/FLIP 이벤트에서 entry_idx = exit_idx - holding_bars."""
    idxs: List[int] = []
    for e in events:
        ev = str(e.get("event", ""))
        if ev.startswith("EXIT") or ev == "FLIP":
            exit_idx = int(e["idx"])
            hb_raw = e.get("holding_bars")
            if hb_raw is None:
                hb_raw = e.get("bars_held")
            hb = int(hb_raw) if hb_raw is not None else 0
            idxs.append(exit_idx - hb)
    if len(idxs) != n_trades:
        idxs_alt = [
            int(e["idx"])
            for e in events
            if str(e.get("event", "")).startswith("ENTRY_") and e.get("idx") is not None
        ]
        if len(idxs_alt) == n_trades:
            return idxs_alt
        raise RuntimeError(
            f"trade_events와 trades 길이 불일치: exit_pairs={len(idxs)}, "
            f"ENTRY={len(idxs_alt)}, trades={n_trades}"
        )
    return idxs


def _profit_map_by_entry_idx(entry_idxs: Sequence[int], trades: List[Mapping[str, Any]]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for i, ei in enumerate(entry_idxs):
        out[int(ei)] = float(trades[i]["profit"]) if trades[i].get("profit") is not None else 0.0
    return out


def _trade_subset_stats(profits: List[float]) -> Tuple[int, Optional[float], float]:
    n = len(profits)
    if n == 0:
        return 0, None, 0.0
    wr = sum(1 for p in profits if p > 0) / n
    return n, wr, float(sum(profits))


def _sideways_split_at_entry(
    trades: List[Mapping[str, Any]],
    entry_idxs: Sequence[int],
    sideways_mask: np.ndarray,
) -> Tuple[Tuple[int, Optional[float], float], Tuple[int, Optional[float], float]]:
    sw_profits: List[float] = []
    nsw_profits: List[float] = []
    for i, ei in enumerate(entry_idxs):
        ei = int(ei)
        if ei < 0 or ei >= len(sideways_mask):
            continue
        p = float(trades[i]["profit"]) if trades[i].get("profit") is not None else 0.0
        if sideways_mask[ei]:
            sw_profits.append(p)
        else:
            nsw_profits.append(p)
    return _trade_subset_stats(sw_profits), _trade_subset_stats(nsw_profits)


def _final_verdict(period_rows: List[Dict[str, Any]]) -> str:
    """
    STEP 10:
    A — 3/4 이상 return 개선, win_rate 개선 구간 다수, 제거분 손실 우세, MDD 유지·개선.
    B — 일부 기간만 개선 또는 trade 변동 큼.
    C — 대부분 악화 또는 구조적 실패.
    """
    if not period_rows:
        return "C"

    n_ret = sum(1 for r in period_rows if float(r.get("improvement_vs_baseline") or 0) > 0)
    n_wr = sum(1 for r in period_rows if float(r.get("win_rate_diff") or 0) > 0)
    n_mdd = sum(
        1
        for r in period_rows
        if float(r.get("no_sideways_max_drawdown") or 1e9)
        <= float(r.get("baseline_max_drawdown") or 1e9) + 1e-15
    )
    n_removed_lossy = sum(
        1
        for r in period_rows
        if int(r.get("removed_trade_count") or 0) == 0
        or float(r.get("removed_win_rate") or 1.0) < 0.5
        or float(r.get("removed_total_profit") or 0) < 0
    )

    if n_ret >= 3 and n_wr >= 3 and n_removed_lossy >= 3 and n_mdd >= 3:
        return "A"
    if n_ret >= 2 or (n_ret >= 1 and n_wr >= 2):
        return "B"
    return "C"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lookback-list",
        type=str,
        default="14,30,60,90",
        help="comma-separated lookback days",
    )
    parser.add_argument("--cooldown-bars", type=int, default=12)
    parser.add_argument("--max-holding-bars", type=int, default=12)
    args = parser.parse_args()

    lookbacks = [int(x.strip()) for x in args.lookback_list.split(",") if x.strip()]
    if not lookbacks:
        raise SystemExit("empty --lookback-list")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine
    from src.strategy_filters.regime_ablation import compute_sideways_bar_mask

    engine = get_ml_backtest_engine("ml_lstm_attn", "BTCUSDT", "5m")
    proba_long, proba_short, df_full = engine.load_predictions()

    common_bt = dict(
        long_threshold=None,
        short_threshold=None,
        use_optimized_threshold=True,
        long_only=True,
        signal_confirmation_bars=1,
        min_hold_bars=12,
        cooldown_bars=args.cooldown_bars,
        use_strategy_guard=True,
        use_strategy_guard_v2=True,
        strategy_guard_v2_mode="soft",
        strategy_guard_v2_scale_floor=0.02,
        use_stage2=True,
        proba_long_cache=proba_long,
        proba_short_cache=proba_short,
        df_with_proba=df_full,
        emit_trade_log=True,
        max_holding_bars=args.max_holding_bars,
        enable_profit_lock_exit=False,
    )

    period_rows: List[Dict[str, Any]] = []
    improvements: List[float] = []

    for days in lookbacks:
        mask = _mask_last_days(df_full, days)
        if int(mask.sum()) <= 0:
            raise RuntimeError(f"빈 mask (lookback_days={days})")

        df_slice = df_full.loc[mask].reset_index(drop=True)
        sideways_mask = compute_sideways_bar_mask(df_slice)

        kw_bl = {
            **common_bt,
            "index_mask": mask,
            "enable_regime_filter": False,
            "regime_filter_mode": "none",
        }
        kw_ns = {
            **common_bt,
            "index_mask": mask,
            "enable_regime_filter": True,
            "regime_filter_mode": "no_sideways",
        }

        res_bl = engine.run_backtest(**kw_bl)
        res_ns = engine.run_backtest(**kw_ns)

        trades_bl = list(_get(res_bl, "trades") or [])
        trades_ns = list(_get(res_ns, "trades") or [])
        ev_bl = list(_get(res_bl, "trade_events") or [])
        ev_ns = list(_get(res_ns, "trade_events") or [])

        bl_ret = float(_get(res_bl, "total_return") or 0.0)
        ns_ret = float(_get(res_ns, "total_return") or 0.0)
        bl_wr = float(_get(res_bl, "win_rate") or 0.0)
        ns_wr = float(_get(res_ns, "win_rate") or 0.0)
        bl_n = int(_get(res_bl, "total_trades") or 0)
        ns_n = int(_get(res_ns, "total_trades") or 0)

        improvement = ns_ret - bl_ret
        improvements.append(improvement)

        ei_bl = _entry_indices_from_trade_events(ev_bl, len(trades_bl))
        ei_ns = _entry_indices_from_trade_events(ev_ns, len(trades_ns))
        map_bl = _profit_map_by_entry_idx(ei_bl, trades_bl)
        map_ns = _profit_map_by_entry_idx(ei_ns, trades_ns)

        set_bl = set(map_bl.keys())
        set_ns = set(map_ns.keys())
        common_idx = set_bl & set_ns
        removed_idx = set_bl - set_ns
        new_idx = set_ns - set_bl

        removed_profits = [map_bl[i] for i in sorted(removed_idx)]
        new_profits = [map_ns[i] for i in sorted(new_idx)]
        common_bl_profits = [map_bl[i] for i in sorted(common_idx)]
        common_ns_profits = [map_ns[i] for i in sorted(common_idx)]

        rc_r, rc_wr, rc_tp = _trade_subset_stats(removed_profits)
        nw_r, nw_wr, nw_tp = _trade_subset_stats(new_profits)
        cb_r, cb_wr, cb_tp = _trade_subset_stats(common_bl_profits)
        cf_r, cf_wr, cf_tp = _trade_subset_stats(common_ns_profits)

        (sw_n, sw_wr, sw_tp), (nsw_n, nsw_wr, nsw_tp) = _sideways_split_at_entry(
            trades_bl, ei_bl, sideways_mask
        )

        row: Dict[str, Any] = {
            "lookback_days": days,
            "baseline_total_return": bl_ret,
            "baseline_win_rate": bl_wr,
            "baseline_total_trades": bl_n,
            "baseline_avg_profit": _get(res_bl, "avg_profit"),
            "baseline_median_profit": _get(res_bl, "median_profit"),
            "baseline_max_drawdown": _get(res_bl, "max_drawdown"),
            "baseline_profit_factor": _profit_factor(trades_bl),
            "baseline_expectancy": _expectancy(trades_bl),
            "baseline_trades_per_day": float(bl_n) / float(days) if days > 0 else None,
            "no_sideways_total_return": ns_ret,
            "no_sideways_win_rate": ns_wr,
            "no_sideways_total_trades": ns_n,
            "no_sideways_avg_profit": _get(res_ns, "avg_profit"),
            "no_sideways_median_profit": _get(res_ns, "median_profit"),
            "no_sideways_max_drawdown": _get(res_ns, "max_drawdown"),
            "no_sideways_profit_factor": _profit_factor(trades_ns),
            "no_sideways_expectancy": _expectancy(trades_ns),
            "no_sideways_trades_per_day": float(ns_n) / float(days) if days > 0 else None,
            "improvement_vs_baseline": improvement,
            "win_rate_diff": ns_wr - bl_wr,
            "trade_diff": ns_n - bl_n,
            "common_trade_count": len(common_idx),
            "removed_trade_count": len(removed_idx),
            "new_trade_count": len(new_idx),
            "removed_win_rate": rc_wr,
            "removed_total_profit": rc_tp,
            "new_win_rate": nw_wr,
            "new_total_profit": nw_tp,
            "common_baseline_win_rate": cb_wr,
            "common_baseline_total_profit": cb_tp,
            "common_no_sideways_win_rate": cf_wr,
            "common_no_sideways_total_profit": cf_tp,
            "baseline_trades_sideways_bar_count": sw_n,
            "baseline_sideways_entry_win_rate": sw_wr,
            "baseline_sideways_entry_total_profit": sw_tp,
            "baseline_trades_nonsideways_bar_count": nsw_n,
            "baseline_nonsideways_entry_win_rate": nsw_wr,
            "baseline_nonsideways_entry_total_profit": nsw_tp,
        }
        period_rows.append(row)

    verdict = _final_verdict(period_rows)

    csv_path = OUT_DIR / f"no_sideways_stability_{stamp}.csv"
    if period_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(period_rows[0].keys()))
            w.writeheader()
            for r in period_rows:
                w.writerow(r)

    n_improved = sum(1 for r in period_rows if float(r["improvement_vs_baseline"]) > 0)
    ratio_pct = 100.0 * n_improved / len(period_rows) if period_rows else 0.0
    best_row = max(period_rows, key=lambda x: float(x["improvement_vs_baseline"])) if period_rows else None
    worst_row = min(period_rows, key=lambda x: float(x["improvement_vs_baseline"])) if period_rows else None

    md_lines = [
        "# no_sideways Stability Validation",
        "",
        "## 1. 목적",
        "",
        "`no_sideways`가 단일 기간 우연이 아니라 여러 기간에서 일관되게 baseline 대비 유리한지 검증한다.",
        "",
        "## 2. 실험 설정",
        "",
        f"- lookback_days: {lookbacks}",
        f"- cooldown_bars: {args.cooldown_bars}",
        f"- max_holding_bars: {args.max_holding_bars}",
        "- profit_lock: OFF",
        "- Case A: baseline / Case B: `enable_regime_filter` + `no_sideways`",
        "",
        "## 3. 기간별 결과",
        "",
        "| period | baseline_return | no_sideways_return | improvement | win_rate_diff | trade_diff |",
        "|--------|-----------------|-------------------|---------------|---------------|------------|",
    ]
    for r in period_rows:
        md_lines.append(
            f"| {r['lookback_days']}d | {r['baseline_total_return']} | "
            f"{r['no_sideways_total_return']} | {r['improvement_vs_baseline']} | "
            f"{r['win_rate_diff']} | {r['trade_diff']} |"
        )
    n_wr_up = sum(1 for r in period_rows if float(r.get("win_rate_diff") or 0) > 0)
    md_lines.extend(
        [
            "",
            "## 4. Consistency",
            "",
            f"- Return 개선 기간 수: **{n_improved} / {len(period_rows)}**",
            f"- Win rate 개선 기간 수: **{n_wr_up} / {len(period_rows)}**",
            f"- 개선 비율 (return 기준): **{ratio_pct:.1f}%**",
            "",
            "## 5. Trade set 분석",
            "",
            "| period | removed_n | removed_wr | removed_profit | new_n | common_n |",
            "|--------|-----------|------------|----------------|-------|---------|",
        ]
    )
    for r in period_rows:
        md_lines.append(
            f"| {r['lookback_days']}d | {r['removed_trade_count']} | "
            f"{r['removed_win_rate']} | {r['removed_total_profit']} | "
            f"{r['new_trade_count']} | {r['common_trade_count']} |"
        )
    md_lines.extend(
        [
            "",
            "## 6. Sideways 진입 트레이드 성과 (baseline만)",
            "",
            "| period | sw_trades | sw_wr | sw_profit | nsw_trades | nsw_wr | nsw_profit |",
            "|--------|-----------|-------|-----------|------------|--------|------------|",
        ]
    )
    for r in period_rows:
        md_lines.append(
            f"| {r['lookback_days']}d | {r['baseline_trades_sideways_bar_count']} | "
            f"{r['baseline_sideways_entry_win_rate']} | {r['baseline_sideways_entry_total_profit']} | "
            f"{r['baseline_trades_nonsideways_bar_count']} | {r['baseline_nonsideways_entry_win_rate']} | "
            f"{r['baseline_nonsideways_entry_total_profit']} |"
        )
    md_lines.extend(
        [
            "",
            "## 7. 리스크 (MDD)",
            "",
            "| period | baseline_mdd | no_sideways_mdd |",
            "|--------|--------------|-----------------|",
        ]
    )
    for r in period_rows:
        md_lines.append(
            f"| {r['lookback_days']}d | {r['baseline_max_drawdown']} | "
            f"{r['no_sideways_max_drawdown']} |"
        )
    md_lines.extend(
        [
            "",
            "## 8. 최종 판정",
            "",
            f"- **Verdict: {verdict}** (A=강한 유효, B=부분, C=무효)",
            "",
        ]
    )

    md_path = OUT_DIR / f"no_sideways_stability_{stamp}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("[NO SIDEWAYS STABILITY SUMMARY]")
    print("")
    for r in period_rows:
        print(
            f"{r['lookback_days']}d: baseline_return={r['baseline_total_return']} "
            f"no_sideways_return={r['no_sideways_total_return']} "
            f"improvement={r['improvement_vs_baseline']}"
        )
    print("")
    print("overall:")
    print(f"- improvement ratio (%): {ratio_pct:.1f}")
    if best_row is not None:
        print(
            f"- best period: {best_row['lookback_days']}d "
            f"(improvement={best_row['improvement_vs_baseline']})"
        )
    if worst_row is not None:
        print(
            f"- worst period: {worst_row['lookback_days']}d "
            f"(improvement={worst_row['improvement_vs_baseline']})"
        )
    print("")
    print(f"final verdict: {verdict}")
    print("")
    print(f"CSV: {csv_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()

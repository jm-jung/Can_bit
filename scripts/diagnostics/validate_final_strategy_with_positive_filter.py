"""
Baseline vs Positive Regime Filter (P7 allow_ema_below_or_strong_up) 최종 전략 검증.

실행:
  python -m scripts.diagnostics.validate_final_strategy_with_positive_filter
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "diagnostics" / "final"

FILTER_MODE = "allow_ema_below_or_strong_up"


def _get(result: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _entry_indices_from_trade_events(events: Sequence[Mapping[str, Any]], n_trades: int) -> List[int]:
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
        raise RuntimeError("trade_events vs trades length mismatch")
    return idxs


def _profit_map_by_entry_idx(entry_idxs: Sequence[int], trades: List[Mapping[str, Any]]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for i in range(min(len(entry_idxs), len(trades))):
        out[int(entry_idxs[i])] = (
            float(trades[i]["profit"]) if trades[i].get("profit") is not None else 0.0
        )
    return out


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


def _sharpe_simple_equity(equity_curve: List[float]) -> Optional[float]:
    """단순 샤프: 바 단위 수익률 mean/std (연율화 없음)."""
    if len(equity_curve) < 3:
        return None
    ec = np.array(equity_curve, dtype=float)
    r = np.diff(ec) / np.maximum(ec[:-1], 1e-12)
    std = float(np.std(r))
    if std < 1e-18:
        return None
    return float(np.mean(r) / std)


def _calmar_simple(total_return: float, max_drawdown: float) -> Optional[float]:
    """기간 수익률 / max_drawdown (둘 다 소수)."""
    mdd = abs(float(max_drawdown))
    if mdd < 1e-18:
        return None
    return float(total_return / mdd)


def _worst_trade(trades: List[Mapping[str, Any]]) -> Optional[float]:
    profits = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    return float(min(profits)) if profits else None


def _drawdown_stats(equity_curve: List[float]) -> Dict[str, Any]:
    """고점 대비 낙폭 이벤트(연속 구간) 개수·underwater 비율."""
    if len(equity_curve) < 2:
        return {"underwater_bar_ratio": None, "drawdown_episodes": None}
    ec = np.array(equity_curve, dtype=float)
    rm = np.maximum.accumulate(ec)
    underwater = ec < rm - 1e-15
    underwater_ratio = float(np.mean(underwater))
    episodes = 0
    prev = False
    for u in underwater:
        if u and not prev:
            episodes += 1
        prev = u
    return {"underwater_bar_ratio": underwater_ratio, "drawdown_episodes": episodes}


def _trade_set_extended(res_bl: Mapping[str, Any], res_f: Mapping[str, Any]) -> Dict[str, Any]:
    tr_bl = list(_get(res_bl, "trades") or [])
    tr_f = list(_get(res_f, "trades") or [])
    ev_bl = list(_get(res_bl, "trade_events") or [])
    ev_f = list(_get(res_f, "trade_events") or [])
    ei_bl = _entry_indices_from_trade_events(ev_bl, len(tr_bl))
    ei_f = _entry_indices_from_trade_events(ev_f, len(tr_f))
    map_bl = _profit_map_by_entry_idx(ei_bl, tr_bl)
    map_f = _profit_map_by_entry_idx(ei_f, tr_f)
    set_bl = set(map_bl.keys())
    set_f = set(map_f.keys())
    common = set_bl & set_f
    removed = set_bl - set_f
    new_only = set_f - set_bl

    def pack(keys: set[int], dmap: Dict[int, float]) -> Dict[str, Any]:
        ps = [dmap[k] for k in sorted(keys)]
        if not ps:
            return {"count": 0, "win_rate": None, "total_profit": 0.0, "avg_profit": None}
        wr = sum(1 for p in ps if p > 0) / len(ps)
        tp = float(sum(ps))
        return {
            "count": len(ps),
            "win_rate": wr,
            "total_profit": tp,
            "avg_profit": tp / len(ps),
        }

    out = {}
    out.update({f"common_{k}": v for k, v in pack(common, map_bl).items()})
    out.update({f"removed_{k}": v for k, v in pack(removed, map_bl).items()})
    out.update({f"new_{k}": v for k, v in pack(new_only, map_f).items()})
    return out


def _final_verdict(period_rows: List[Dict[str, Any]]) -> str:
    imps = [float(r["improvement_vs_baseline"]) for r in period_rows]
    improved = sum(1 for x in imps if x > 1e-9)
    avg_imp = float(sum(imps) / len(imps)) if imps else 0.0
    mdd_ok = sum(
        1
        for r in period_rows
        if float(r["filtered_max_drawdown"]) <= float(r["baseline_max_drawdown"]) + 1e-15
    )
    removed_lossy_periods = 0
    periods_with_removed = 0
    for r in period_rows:
        rc = int(r.get("removed_count") or 0)
        if rc == 0:
            continue
        periods_with_removed += 1
        rw = r.get("removed_win_rate")
        rtp = float(r.get("removed_total_profit") or 0)
        if rw is not None and float(rw) < 0.5:
            removed_lossy_periods += 1
        elif rtp < 0:
            removed_lossy_periods += 1

    removed_ok = periods_with_removed == 0 or removed_lossy_periods >= min(3, periods_with_removed)

    if improved >= 3 and avg_imp > 0 and mdd_ok >= 3 and removed_ok:
        return "A"
    if improved >= 2 or avg_imp > 0:
        return "B"
    return "C"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-list", type=str, default="14,30,60,90")
    parser.add_argument("--cooldown-bars", type=int, default=12)
    parser.add_argument("--max-holding-bars", type=int, default=12)
    args = parser.parse_args()

    lookbacks = [int(x.strip()) for x in args.lookback_list.split(",") if x.strip()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from scripts.diagnostics.analyze_regime_combinations import mask_last_days
    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine

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
        enable_regime_filter=False,
        regime_filter_mode="none",
    )

    detail_rows: List[Dict[str, Any]] = []

    for days in lookbacks:
        mask = mask_last_days(df_full, days)
        if int(mask.sum()) <= 0:
            continue
        kw = {**common_bt, "index_mask": mask}

        res_bl = engine.run_backtest(
            **kw,
            enable_positive_regime_filter=False,
            positive_regime_filter_mode="none",
        )
        res_f = engine.run_backtest(
            **kw,
            enable_positive_regime_filter=True,
            positive_regime_filter_mode=FILTER_MODE,
        )

        trades_bl = list(_get(res_bl, "trades") or [])
        trades_f = list(_get(res_f, "trades") or [])
        eq_bl = list(_get(res_bl, "equity_curve") or [1.0])
        eq_f = list(_get(res_f, "equity_curve") or [1.0])

        bl_ret = float(_get(res_bl, "total_return") or 0)
        fr_ret = float(_get(res_f, "total_return") or 0)
        bl_mdd = float(_get(res_bl, "max_drawdown") or 0)
        fr_mdd = float(_get(res_f, "max_drawdown") or 0)
        bl_wr = float(_get(res_bl, "win_rate") or 0)
        fr_wr = float(_get(res_f, "win_rate") or 0)
        bl_ntr = int(_get(res_bl, "total_trades") or 0)
        fr_ntr = int(_get(res_f, "total_trades") or 0)

        ts = _trade_set_extended(res_bl, res_f)

        dd_bl = _drawdown_stats(eq_bl)
        dd_f = _drawdown_stats(eq_f)

        row = {
            "lookback_days": days,
            "case": "per_period",
            "baseline_total_return": bl_ret,
            "filtered_total_return": fr_ret,
            "improvement_vs_baseline": fr_ret - bl_ret,
            "baseline_win_rate": bl_wr,
            "filtered_win_rate": fr_wr,
            "win_rate_diff": fr_wr - bl_wr,
            "baseline_total_trades": bl_ntr,
            "filtered_total_trades": fr_ntr,
            "trade_diff": fr_ntr - bl_ntr,
            "baseline_avg_profit": _get(res_bl, "avg_profit"),
            "filtered_avg_profit": _get(res_f, "avg_profit"),
            "baseline_median_profit": _get(res_bl, "median_profit"),
            "filtered_median_profit": _get(res_f, "median_profit"),
            "baseline_max_drawdown": bl_mdd,
            "filtered_max_drawdown": fr_mdd,
            "MDD_diff": fr_mdd - bl_mdd,
            "baseline_profit_factor": _profit_factor(trades_bl),
            "filtered_profit_factor": _profit_factor(trades_f),
            "baseline_expectancy": _expectancy(trades_bl),
            "filtered_expectancy": _expectancy(trades_f),
            "baseline_trades_per_day": float(bl_ntr) / float(days) if days > 0 else None,
            "filtered_trades_per_day": float(fr_ntr) / float(days) if days > 0 else None,
            "baseline_sharpe_simple": _sharpe_simple_equity(eq_bl),
            "filtered_sharpe_simple": _sharpe_simple_equity(eq_f),
            "baseline_calmar": _calmar_simple(bl_ret, bl_mdd),
            "filtered_calmar": _calmar_simple(fr_ret, fr_mdd),
            "baseline_worst_trade": _worst_trade(trades_bl),
            "filtered_worst_trade": _worst_trade(trades_f),
            "baseline_underwater_ratio": dd_bl["underwater_bar_ratio"],
            "filtered_underwater_ratio": dd_f["underwater_bar_ratio"],
            "baseline_dd_episodes": dd_bl["drawdown_episodes"],
            "filtered_dd_episodes": dd_f["drawdown_episodes"],
            "baseline_equity_final": eq_bl[-1] if eq_bl else None,
            "filtered_equity_final": eq_f[-1] if eq_f else None,
            "blocked_positive_regime": _get(res_f, "entries_blocked_by_positive_regime"),
            **ts,
        }
        detail_rows.append(row)

    imps = [float(r["improvement_vs_baseline"]) for r in detail_rows]
    improved_n = sum(1 for x in imps if x > 1e-9)
    avg_imp = float(sum(imps) / len(imps)) if imps else 0.0
    worst_imp = float(min(imps)) if imps else 0.0
    avg_tc = (
        float(sum(int(r["filtered_total_trades"]) for r in detail_rows) / len(detail_rows))
        if detail_rows
        else None
    )

    verdict = _final_verdict(detail_rows)

    cons_row = {
        "lookback_days": -1,
        "case": "consistency_summary",
        "periods_tested": len(detail_rows),
        "improved_periods": improved_n,
        "improvement_ratio": improved_n / len(detail_rows) if detail_rows else 0.0,
        "avg_improvement": avg_imp,
        "worst_improvement": worst_imp,
        "avg_filtered_trade_count": avg_tc,
        "final_verdict": verdict,
    }

    csv_path = OUT_DIR / f"final_strategy_positive_filter_{stamp}.csv"
    all_rows = detail_rows + [cons_row]
    keys = sorted({k for row in all_rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in keys})

    md_path = OUT_DIR / f"final_strategy_positive_filter_{stamp}.md"
    md = [
        "# Final strategy: Positive Regime Filter (P7)",
        "",
        "## 목적",
        "`allow_ema_below_or_strong_up` 적용 시 baseline 대비 전략 성능 최종 확인.",
        "",
        "## 설정",
        f"- filter: `{FILTER_MODE}`",
        f"- lookbacks: {lookbacks}",
        f"- cooldown_bars={args.cooldown_bars}, max_holding_bars={args.max_holding_bars}, profit_lock=OFF",
        "",
        "## 기간별 결과",
        "",
        "| period | baseline_ret | filtered_ret | improvement | wr_diff | trade_diff | MDD_diff |",
        "|--------|-------------|--------------|-------------|---------|------------|----------|",
    ]
    for r in detail_rows:
        md.append(
            f"| {r['lookback_days']}d | {r['baseline_total_return']} | {r['filtered_total_return']} | "
            f"{r['improvement_vs_baseline']} | {r['win_rate_diff']} | {r['trade_diff']} | {r['MDD_diff']} |"
        )
    md.extend(
        [
            "",
            "## Trade set",
            "",
            "| period | common_n | removed_n | removed_wr | removed_tp | new_n | new_wr |",
            "|--------|----------|-----------|------------|------------|-------|--------|",
        ]
    )
    for r in detail_rows:
        md.append(
            f"| {r['lookback_days']}d | {r.get('common_count')} | {r.get('removed_count')} | "
            f"{r.get('removed_win_rate')} | {r.get('removed_total_profit')} | {r.get('new_count')} | "
            f"{r.get('new_win_rate')} |"
        )
    md.extend(
        [
            "",
            "## 리스크",
            "",
            "| period | base_MDD | filt_MDD | base_sharpe | filt_sharpe | base_calmar | filt_calmar | worst_bl | worst_f |",
            "|--------|----------|----------|-------------|-------------|-------------|-------------|----------|---------|",
        ]
    )
    for r in detail_rows:
        md.append(
            f"| {r['lookback_days']}d | {r['baseline_max_drawdown']} | {r['filtered_max_drawdown']} | "
            f"{r['baseline_sharpe_simple']} | {r['filtered_sharpe_simple']} | {r['baseline_calmar']} | "
            f"{r['filtered_calmar']} | {r['baseline_worst_trade']} | {r['filtered_worst_trade']} |"
        )
    md.extend(
        [
            "",
            "## Consistency",
            "",
            f"- improved_periods: **{improved_n} / {len(detail_rows)}**",
            f"- avg_improvement: **{avg_imp}**",
            f"- worst_improvement: **{worst_imp}**",
            f"- avg filtered trades: **{avg_tc}**",
            "",
            f"## 판정: **{verdict}**",
            "",
        ]
    )
    md_path.write_text("\n".join(md), encoding="utf-8")

    print("[FINAL STRATEGY SUMMARY]")
    print("")
    for r in detail_rows:
        print(
            f"{r['lookback_days']}d: baseline_return={r['baseline_total_return']} "
            f"filtered_return={r['filtered_total_return']} improvement={r['improvement_vs_baseline']}"
        )
    print("")
    print("overall:")
    print(f"- improved_periods: {improved_n} / {len(detail_rows)}")
    print(f"- avg_improvement: {avg_imp}")
    print(f"- avg_trade_count (filtered): {avg_tc}")
    print("")
    print(f"final verdict: {verdict}")
    print("")
    print(f"CSV: {csv_path}")
    print(f"MD: {md_path}")


if __name__ == "__main__":
    main()

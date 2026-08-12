"""
Profit Lock TP/SL 미니 스윕 (엔진 기준, 진단 전용).

실행:
  python -m scripts.diagnostics.sweep_profit_lock_tpsl --lookback-days 14
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "diagnostics" / "exit_validation"


def _mask_last_days(df: pd.DataFrame, lookback_days: int) -> np.ndarray:
    ts = pd.to_datetime(df["timestamp"])
    end = ts.max()
    start = end - pd.Timedelta(days=int(lookback_days))
    return (ts >= start).values


def _get(res: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return res.get(key, default) if isinstance(res, dict) else getattr(res, key, default)


def _trade_profits(result: Mapping[str, Any]) -> List[float]:
    trades = _get(result, "trades") or []
    out: List[float] = []
    for t in trades:
        if isinstance(t, dict):
            p = t.get("profit")
        else:
            p = getattr(t, "profit", None)
        if p is not None:
            try:
                out.append(float(p))
            except (TypeError, ValueError):
                pass
    return out


def _exit_reason_buckets(trade_events: Sequence[Mapping[str, Any]]) -> Tuple[int, int, int, int]:
    """profit_lock_tp, profit_lock_sl, max_holding, other"""
    tp_c = sl_c = mh_c = other = 0
    for e in trade_events:
        ev = str(e.get("event", ""))
        if not ev.startswith("EXIT"):
            continue
        r = str(e.get("exit_reason") or "unknown")
        if r == "profit_lock_tp":
            tp_c += 1
        elif r == "profit_lock_sl":
            sl_c += 1
        elif r == "max_holding":
            mh_c += 1
        else:
            other += 1
    return tp_c, sl_c, mh_c, other


def _parse_float_list(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


def _profit_factor_gross(trades_profit: Sequence[float]) -> Tuple[Optional[float], float, float]:
    wins = [p for p in trades_profit if p > 0]
    losses = [p for p in trades_profit if p < 0]
    gp = float(sum(wins))
    gl = float(sum(losses))
    if gl >= 0:
        return (None, gp, gl)
    pf = gp / abs(gl) if abs(gl) > 1e-18 else None
    return (pf, gp, gl)


def _payoff_ratio(trades_profit: Sequence[float]) -> Optional[float]:
    wins = [p for p in trades_profit if p > 0]
    losses = [p for p in trades_profit if p < 0]
    if not wins or not losses:
        return None
    return float(np.mean(wins) / abs(np.mean(losses)))


def _expectancy(trades_profit: Sequence[float]) -> float:
    if not trades_profit:
        return float("nan")
    return float(np.mean(trades_profit))


def run_single(
    engine,
    *,
    common: Dict[str, Any],
    enable_pl: bool,
    tp: Optional[float],
    sl: Optional[float],
) -> Mapping[str, Any]:
    kw = {
        **common,
        "enable_profit_lock_exit": enable_pl,
        "profit_lock_take_profit": tp if tp is not None else 0.0015,
        "profit_lock_stop_loss": sl if sl is not None else -0.0030,
        "profit_lock_max_bars": common.get("profit_lock_max_bars", 12),
        "profit_lock_conservative_intrabar": True,
    }
    if not enable_pl:
        kw["profit_lock_take_profit"] = 0.0015
        kw["profit_lock_stop_loss"] = -0.0030
    return engine.run_backtest(**kw)


def build_row(
    case_name: str,
    lookback_days: int,
    cooldown_bars: int,
    profit_lock_max_bars: int,
    tp: Optional[float],
    sl: Optional[float],
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    profits = _trade_profits(result)
    pf, gp, gl = _profit_factor_gross(profits)
    tr = float(_get(result, "total_return") or 0.0)
    mdd = float(_get(result, "max_drawdown") or 0.0)
    events = _get(result, "trade_events") or []
    pl_tp_n, pl_sl_n, mh_n, oth_n = _exit_reason_buckets(events)
    pl_tp_ex = int(_get(result, "profit_lock_tp_exits") or 0)
    pl_sl_ex = int(_get(result, "profit_lock_sl_exits") or 0)
    total_trades = int(_get(result, "total_trades") or 0)
    tp_sl_ratio = (pl_tp_ex / pl_sl_ex) if pl_sl_ex > 0 else float("nan")

    score = tr - mdd

    return {
        "case_name": case_name,
        "lookback_days": lookback_days,
        "cooldown_bars": cooldown_bars,
        "profit_lock_max_bars": profit_lock_max_bars,
        "tp": tp if tp is not None else "",
        "sl": sl if sl is not None else "",
        "total_return": tr,
        "win_rate": float(_get(result, "win_rate") or 0.0),
        "total_trades": total_trades,
        "avg_profit": _get(result, "avg_profit"),
        "median_profit": _get(result, "median_profit"),
        "max_drawdown": mdd,
        "profit_factor": pf,
        "expectancy": _expectancy(profits),
        "trades_per_day": total_trades / lookback_days if lookback_days > 0 else float("nan"),
        "exit_profit_lock_tp_count": pl_tp_ex,
        "exit_profit_lock_sl_count": pl_sl_ex,
        "exit_max_holding_count": mh_n,
        "exit_other_count": oth_n,
        "tp_sl_ratio": tp_sl_ratio,
        "score": score,
        "avg_win": _get(result, "avg_win"),
        "avg_loss": _get(result, "avg_loss"),
        "payoff_ratio": _payoff_ratio(profits),
        "gross_profit": gp,
        "gross_loss": gl,
    }


def _sort_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
    return (-float(row["score"]), -float(row["total_return"]), float(row["max_drawdown"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Profit lock TP/SL sweep (engine)")
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--cooldown-bars", type=int, default=12)
    parser.add_argument("--profit-lock-max-bars", type=int, default=12)
    parser.add_argument(
        "--tp-values",
        type=str,
        default="0.0015,0.0020,0.0025,0.0030,0.0035,0.0040",
    )
    parser.add_argument(
        "--sl-values",
        type=str,
        default="-0.0020,-0.0025,-0.0030,-0.0035,-0.0040,-0.0050",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-holding-bars", type=int, default=12)
    args = parser.parse_args()

    tp_list = _parse_float_list(args.tp_values)
    sl_list = _parse_float_list(args.sl_values)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine

    engine = get_ml_backtest_engine("ml_lstm_attn", "BTCUSDT", "5m")
    proba_long, proba_short, df = engine.load_predictions()
    mask = _mask_last_days(df, args.lookback_days)
    if int(mask.sum()) <= 0:
        raise RuntimeError("빈 mask")

    common = dict(
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
        df_with_proba=df,
        index_mask=mask,
        emit_trade_log=True,
        max_holding_bars=args.max_holding_bars,
        profit_lock_max_bars=args.profit_lock_max_bars,
    )

    rows: List[Dict[str, Any]] = []

    res_bl = run_single(engine, common=common, enable_pl=False, tp=None, sl=None)
    rows.append(
        build_row(
            "baseline",
            args.lookback_days,
            args.cooldown_bars,
            args.profit_lock_max_bars,
            None,
            None,
            res_bl,
        )
    )

    for tp, sl in product(tp_list, sl_list):
        case = f"grid_tp{tp:.6f}_sl{sl:.6f}"
        res = run_single(engine, common=common, enable_pl=True, tp=tp, sl=sl)
        rows.append(
            build_row(
                case,
                args.lookback_days,
                args.cooldown_bars,
                args.profit_lock_max_bars,
                tp,
                sl,
                res,
            )
        )

    baseline_row = rows[0]
    pl_a_row = next(
        (
            r
            for r in rows
            if r["case_name"].startswith("grid_")
            and abs(float(r["tp"]) - 0.0015) < 1e-9
            and abs(float(r["sl"]) - (-0.0030)) < 1e-9
        ),
        None,
    )

    grid_rows = [r for r in rows if r["case_name"].startswith("grid_")]
    sorted_grid = sorted(grid_rows, key=_sort_key)

    csv_path = OUT_DIR / f"profit_lock_tpsl_sweep_{stamp}.csv"
    if rows:
        keys = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    top_n = sorted(grid_rows, key=_sort_key)[: max(1, int(args.top_n))]

    bl_tr = float(baseline_row["total_return"])
    bl_ap = float(baseline_row["avg_profit"] or 0.0)
    bl_mdd = float(baseline_row["max_drawdown"] or 0.0)
    best = sorted_grid[0] if sorted_grid else None

    pa_tr = float(pl_a_row["total_return"]) if pl_a_row else float("nan")
    pa_ap = float(pl_a_row["avg_profit"] or 0.0) if pl_a_row else float("nan")

    best_beats_bl = best is not None and float(best["total_return"]) > bl_tr
    best_beats_pa = best is not None and float(best["total_return"]) > pa_tr
    best_ap_ok = best is not None and (
        float(best["avg_profit"] or 0) >= 0 or float(best["avg_profit"] or 0) >= bl_ap - 1e-12
    )
    mdd_ok = (
        best is not None and float(best["max_drawdown"] or 0) <= bl_mdd * 1.35 + 1e-12
    )

    best_ap_ge_bl = best is not None and float(best["avg_profit"] or 0) >= bl_ap - 1e-12
    grid_vs_bl = best is not None and float(best["total_return"]) > bl_tr

    if best is None or not grid_vs_bl:
        verdict = "C. Profit lock 단독 보류"
    elif (
        best_beats_bl
        and best_beats_pa
        and best_ap_ge_bl
        and mdd_ok
    ):
        verdict = "A. 적용 후보 발견"
    else:
        verdict = "B. 추가 스윕 필요"

    md_path = OUT_DIR / f"profit_lock_tpsl_sweep_{stamp}.md"

    md_lines = [
        "# Can_bit Profit Lock TP/SL Sweep Report",
        "",
        "## 1. 목적",
        "",
        "- 엔진 기준으로 TP/SL 강도 조합을 비교해 수수료·슬리피지·스케일·재진입 후에도 edge가 남는 조합을 탐색한다.",
        "",
        "## 2. 실험 조건",
        "",
        f"- lookback_days: **{args.lookback_days}**",
        f"- cooldown_bars: **{args.cooldown_bars}**",
        f"- profit_lock_max_bars: **{args.profit_lock_max_bars}**",
        f"- max_holding_bars: **{args.max_holding_bars}**",
        f"- tp_values: `{args.tp_values}`",
        f"- sl_values: `{args.sl_values}`",
        "- baseline: `enable_profit_lock_exit=False`",
        "- grid: `enable_profit_lock_exit=True` × (tp, sl)",
        "",
        "## 3. Baseline 결과",
        "",
        f"- total_return: **{float(baseline_row['total_return']):.8f}**",
        f"- win_rate: **{float(baseline_row['win_rate']):.6f}**",
        f"- trades: **{int(baseline_row['total_trades'])}**",
        f"- avg_profit: **{baseline_row['avg_profit']}**",
        f"- MDD: **{float(baseline_row['max_drawdown']):.8f}**",
        "",
        "## 4. 기존 PL A (TP 0.0015 / SL -0.0030)",
        "",
    ]
    if pl_a_row:
        md_lines.extend(
            [
                f"- total_return: **{float(pl_a_row['total_return']):.8f}**",
                f"- win_rate: **{float(pl_a_row['win_rate']):.6f}**",
                f"- trades: **{int(pl_a_row['total_trades'])}**",
                f"- avg_profit: **{pl_a_row['avg_profit']}**",
                f"- MDD: **{float(pl_a_row['max_drawdown']):.8f}**",
                "",
            ]
        )
    else:
        md_lines.append("- 그리드에 해당 조합 없음 (tp/sl 목록 확인)\n")

    md_lines.extend(
        [
            "## 5. 전체 Top N (score = total_return - max_drawdown)",
            "",
            "| rank | tp | sl | total_return | win_rate | trades | avg_profit | MDD | profit_factor | score | tp_count | sl_count | max_holding_count |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for i, r in enumerate(top_n, start=1):
        pf_s = f"{r['profit_factor']:.4f}" if r.get("profit_factor") is not None else "계산 불가"
        md_lines.append(
            f"| {i} | {r['tp']} | {r['sl']} | {float(r['total_return']):.8f} | "
            f"{float(r['win_rate']):.6f} | {int(r['total_trades'])} | "
            f"{float(r['avg_profit'] or 0):.8f} | {float(r['max_drawdown']):.8f} | "
            f"{pf_s} | {float(r['score']):.8f} | "
            f"{int(r['exit_profit_lock_tp_count'])} | {int(r['exit_profit_lock_sl_count'])} | "
            f"{int(r['exit_max_holding_count'])} |"
        )

    md_lines.extend(
        [
            "",
            "## 6. 탈락 패턴 (요약)",
            "",
            "- **TP가 낮은 조합**: 소폭 익절만 반복하면 승률은 높아도 수수료·재진입으로 total_return이 깎일 수 있음.",
            "- **SL이 과도하게 큰 조합**: 손실 폭이 커져 gross_loss·MDD 악화 가능.",
            "- **SL이 너무 좁은 조합**: profit_lock_sl 빈도 증가로 churn.",
            "- **거래 수 폭증**: 동일 lookback에서 total_trades가 baseline 대비 과도하게 크면 재진입·비용 부담 검토.",
            "- **win_rate만 높음**: avg_profit·total_return이 음수면 기대값 관점에서 부적합.",
            "",
            "## 7. 최종 판정",
            "",
            f"- **{verdict}**",
            "",
            "## 8. 다음 액션",
            "",
            "- 후보가 있으면 lookback 30/60/90일 동일 그리드 재검증.",
            "- 후보가 없으면 entry/regime 쪽과 병행 검토.",
            "",
    ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n[PROFIT LOCK TP/SL SWEEP SUMMARY]\n")
    print("baseline:")
    print(f"  total_return: {float(baseline_row['total_return']):.8f}")
    print(f"  win_rate: {float(baseline_row['win_rate']):.6f}")
    print(f"  trades: {int(baseline_row['total_trades'])}")
    print(f"  avg_profit: {baseline_row['avg_profit']}")
    print(f"  MDD: {float(baseline_row['max_drawdown']):.8f}")
    print("")
    print("current PL A:")
    if pl_a_row:
        print(f"  total_return: {float(pl_a_row['total_return']):.8f}")
        print(f"  win_rate: {float(pl_a_row['win_rate']):.6f}")
        print(f"  trades: {int(pl_a_row['total_trades'])}")
        print(f"  avg_profit: {pl_a_row['avg_profit']}")
        print(f"  MDD: {float(pl_a_row['max_drawdown']):.8f}")
    else:
        print("  (없음)")
    print("")
    print("best candidate:")
    if best:
        print(f"  tp: {best['tp']}")
        print(f"  sl: {best['sl']}")
        print(f"  total_return: {float(best['total_return']):.8f}")
        print(f"  win_rate: {float(best['win_rate']):.6f}")
        print(f"  trades: {int(best['total_trades'])}")
        print(f"  avg_profit: {best['avg_profit']}")
        print(f"  MDD: {float(best['max_drawdown']):.8f}")
        print(f"  score: {float(best['score']):.8f}")
    else:
        print("  (없음)")
    print("")
    print(f"final verdict:\n  {verdict}")
    print("")
    print("created:")
    print(f"  csv: {csv_path}")
    print(f"  markdown: {md_path}")
    print("")


if __name__ == "__main__":
    main()

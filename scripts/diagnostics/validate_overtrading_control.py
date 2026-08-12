"""
Profit lock + 과매매(재진입) 억제 조합 엔진 검증.

실행:
  python -m scripts.diagnostics.validate_overtrading_control
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "diagnostics" / "overtrading"


def _mask_last_days(df: pd.DataFrame, lookback_days: int) -> np.ndarray:
    ts = pd.to_datetime(df["timestamp"])
    end = ts.max()
    start = end - pd.Timedelta(days=int(lookback_days))
    return (ts >= start).values


def _get(result: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _avg_bars_between_entries(trade_events: List[Dict[str, Any]]) -> Optional[float]:
    idxs = sorted(
        int(e["idx"])
        for e in trade_events
        if str(e.get("event", "")).startswith("ENTRY_") and e.get("idx") is not None
    )
    if len(idxs) < 2:
        return None
    gaps = [idxs[j] - idxs[j - 1] for j in range(1, len(idxs))]
    return float(sum(gaps) / len(gaps))


def _pick_best(rows: List[Dict[str, Any]]) -> Optional[str]:
    """baseline 제외 행 중 total_return 최대."""
    pool = [r for r in rows if r["case_id"] != "baseline"]
    if not pool:
        return None
    best = max(pool, key=lambda x: float(x["total_return"] or -1e18))
    return str(best["case_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--max-holding-bars", type=int, default=12)
    args = parser.parse_args()

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
        # 진단 전용: paper(12)보다 짧게 두어 enable_reentry_cooldown(12) 등이 기존 cooldown과 차별화되게 함.
        cooldown_bars=6,
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
        profit_lock_take_profit=0.0015,
        profit_lock_stop_loss=-0.0030,
        profit_lock_max_bars=12,
        profit_lock_conservative_intrabar=True,
    )

    runs = [
        ("baseline", dict(enable_profit_lock_exit=False)),
        ("case1_pl_only", dict(enable_profit_lock_exit=True)),
        (
            "case2_pl_global_cd",
            dict(
                enable_profit_lock_exit=True,
                enable_reentry_cooldown=True,
                reentry_cooldown_bars=12,
            ),
        ),
        (
            "case3_pl_post_tp",
            dict(
                enable_profit_lock_exit=True,
                enable_post_tp_block=True,
                post_tp_block_bars=6,
            ),
        ),
        (
            "case4_pl_post_sl",
            dict(
                enable_profit_lock_exit=True,
                enable_post_sl_block=True,
                post_sl_block_bars=12,
            ),
        ),
        (
            "case5_pl_tp_sl_block",
            dict(
                enable_profit_lock_exit=True,
                enable_post_tp_block=True,
                post_tp_block_bars=6,
                enable_post_sl_block=True,
                post_sl_block_bars=12,
            ),
        ),
    ]

    rows: List[Dict[str, Any]] = []
    for case_id, kw in runs:
        kw_full = {**common, **kw}
        res = engine.run_backtest(**kw_full)
        ev = _get(res, "trade_events") or []
        n_tr = int(_get(res, "total_trades") or 0)
        row = {
            "case_id": case_id,
            "total_return": _get(res, "total_return"),
            "win_rate": _get(res, "win_rate"),
            "total_trades": n_tr,
            "avg_profit": _get(res, "avg_profit"),
            "max_drawdown": _get(res, "max_drawdown"),
            "trade_per_day": (
                float(n_tr) / float(args.lookback_days) if args.lookback_days > 0 else None
            ),
            "avg_bars_between_entries": _avg_bars_between_entries(ev),
            "profit_lock_tp_exits": _get(res, "profit_lock_tp_exits"),
            "profit_lock_sl_exits": _get(res, "profit_lock_sl_exits"),
            "ot_blocked_global": _get(res, "overtrading_control", {})
            .get("blocked_entry_global", 0)
            if isinstance(_get(res, "overtrading_control"), dict)
            else None,
            "ot_blocked_post_tp": (
                _get(res, "overtrading_control", {}).get("blocked_entry_post_tp", 0)
                if isinstance(_get(res, "overtrading_control"), dict)
                else None
            ),
            "ot_blocked_post_sl": (
                _get(res, "overtrading_control", {}).get("blocked_entry_post_sl", 0)
                if isinstance(_get(res, "overtrading_control"), dict)
                else None
            ),
        }
        rows.append(row)

    csv_path = OUT_DIR / f"overtrading_validation_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 판정
    bl = next(r for r in rows if r["case_id"] == "baseline")
    pl_only = next(r for r in rows if r["case_id"] == "case1_pl_only")
    improved_vs_pl = [
        r
        for r in rows
        if r["case_id"].startswith("case") and r["case_id"] != "case1_pl_only"
    ]

    def better(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        try:
            return float(a["total_return"] or 0) > float(b["total_return"] or 0)
        except (TypeError, ValueError):
            return False

    def lower_dd(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        try:
            return float(a["max_drawdown"] or 0) < float(b["max_drawdown"] or 0)
        except (TypeError, ValueError):
            return False

    success = []
    partial = []
    fail = []
    for r in improved_vs_pl:
        tr_down = float(r["total_trades"] or 0) < float(pl_only["total_trades"] or 0)
        ret_up = better(r, pl_only)
        wr_ok = float(r["win_rate"] or 0) >= float(pl_only["win_rate"] or 0) - 1e-9
        dd_ok = lower_dd(r, pl_only)
        if tr_down and wr_ok and ret_up and dd_ok:
            success.append(r["case_id"])
        elif tr_down and (ret_up or abs(float(r["total_return"] or 0) - float(pl_only["total_return"] or 0)) < 1e-6):
            partial.append(r["case_id"])
        elif tr_down and not ret_up:
            fail.append(r["case_id"])

    if success:
        verdict = "A. 성공"
        verdict_detail = f"후보: {', '.join(success)}"
    elif partial:
        verdict = "B. 부분 성공"
        verdict_detail = f"후보: {', '.join(partial)}"
    else:
        verdict = "C. 실패"
        verdict_detail = "재진입 억제만으로는 total_return 회복 미흡"

    best_id = _pick_best(rows)

    md_path = OUT_DIR / f"overtrading_validation_{stamp}.md"
    md_lines = [
        "# Overtrading Control Validation",
        "",
        f"- lookback_days={args.lookback_days}, max_holding_bars={args.max_holding_bars}, cooldown_bars=6 (진단 전용)",
        "",
        "## 결과표",
        "",
        "| case | trades | win_rate | total_return | MDD | trade/day | avg_bars_between_entries |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        abe = r["avg_bars_between_entries"]
        abe_s = f"{abe:.2f}" if abe is not None else "계산 불가"
        md_lines.append(
            f"| {r['case_id']} | {r['total_trades']} | {float(r['win_rate'] or 0):.6f} | "
            f"{float(r['total_return'] or 0):.8f} | {float(r['max_drawdown'] or 0):.8f} | "
            f"{float(r['trade_per_day'] or 0):.4f} | {abe_s} |"
        )
    md_lines.extend(
        [
            "",
            "## 판정",
            "",
            f"- **{verdict}**",
            f"- {verdict_detail}",
            f"- **best_case (total_return 기준)**: `{best_id}`",
            "",
            "## 결론 요약",
            "",
            "1. profit lock + cooldown 조합 유효성: 위 표 및 판정 참고.",
            "2. 가장 균형 잡힌 케이스: `best_case` 행.",
            "3. 다음 단계: TP/SL 블록·global cooldown 튜닝 또는 더 긴 룩백 재검증.",
            "",
            "## 주의 (실험 설정)",
            "",
            "- 본 스크립트는 과매매 필터 효과를 분리하기 위해 **cooldown_bars=6** 을 사용한다.",
            "- 운영 `run_paper_shadow` 의 cooldown_bars=12 와 동일하지 않으며, 수치 비교는 진단 목적 전용이다.",
            "",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n[OVERTRADING CONTROL SUMMARY]\n")
    for r in rows:
        print(
            f"{r['case_id']}: trades={r['total_trades']} win_rate={float(r['win_rate'] or 0):.4f} "
            f"total_return={float(r['total_return'] or 0):.8f} MDD={float(r['max_drawdown'] or 0):.8f}"
        )
    print(f"\nbest_case (heuristic): {best_id}")
    print(f"verdict: {verdict}")
    print(f"\ncsv: {csv_path}")
    print(f"md:  {md_path}\n")


if __name__ == "__main__":
    main()

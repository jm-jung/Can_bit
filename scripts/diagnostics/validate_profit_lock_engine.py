"""
엔진 기준 profit lock exit 활성화 전후 비교 (진단 전용).

실행 (저장소 루트):
  python -m scripts.diagnostics.validate_profit_lock_engine

출력:
  data/diagnostics/exit_validation/profit_lock_engine_validation_<stamp>.md
  data/diagnostics/exit_validation/profit_lock_engine_validation_<stamp>.csv
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
OUT_DIR = REPO_ROOT / "data" / "diagnostics" / "exit_validation"


def _mask_last_days(df: pd.DataFrame, lookback_days: int) -> np.ndarray:
    ts = pd.to_datetime(df["timestamp"])
    end = ts.max()
    start = end - pd.Timedelta(days=int(lookback_days))
    return (ts >= start).values


def _pick_result_scalar(result: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _exit_reason_hist(result: Mapping[str, Any]) -> Dict[str, int]:
    evs = _pick_result_scalar(result, "trade_events") or []
    counts: Dict[str, int] = {}
    for e in evs:
        ev = str(e.get("event", ""))
        if not ev.startswith("EXIT"):
            continue
        r = str(e.get("exit_reason") or "unknown")
        counts[r] = counts.get(r, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))


def _row(
    label: str,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "variant": label,
        "total_return": _pick_result_scalar(result, "total_return"),
        "win_rate": _pick_result_scalar(result, "win_rate"),
        "total_trades": _pick_result_scalar(result, "total_trades"),
        "avg_profit": _pick_result_scalar(result, "avg_profit"),
        "max_drawdown": _pick_result_scalar(result, "max_drawdown"),
        "profit_lock_tp_exits": _pick_result_scalar(result, "profit_lock_tp_exits", 0),
        "profit_lock_sl_exits": _pick_result_scalar(result, "profit_lock_sl_exits", 0),
        "profit_lock_enabled": _pick_result_scalar(result, "profit_lock_enabled", False),
    }


def _fmt_float(x: Any) -> str:
    if x is None:
        return "계산 불가"
    try:
        return f"{float(x):.8f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profit lock engine validation")
    parser.add_argument("--lookback-days", type=int, default=14, help="평가 구간(마지막 N일)")
    parser.add_argument("--max-holding-bars", type=int, default=12, help="max_holding_bars (paper와 유사하게 고정 청산 폭 제한)")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine

    engine = get_ml_backtest_engine("ml_lstm_attn", "BTCUSDT", "5m")
    proba_long, proba_short, df = engine.load_predictions()
    mask = _mask_last_days(df, args.lookback_days)
    if int(mask.sum()) <= 0:
        raise RuntimeError("index_mask가 비었습니다. lookback_days를 늘리거나 데이터를 확인하세요.")

    common = dict(
        long_threshold=None,
        short_threshold=None,
        use_optimized_threshold=True,
        long_only=True,
        signal_confirmation_bars=1,
        min_hold_bars=12,
        cooldown_bars=12,
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
    )

    bl1 = engine.run_backtest(
        enable_profit_lock_exit=False,
        **common,
    )
    bl2 = engine.run_backtest(
        enable_profit_lock_exit=False,
        **common,
    )
    pl = engine.run_backtest(
        enable_profit_lock_exit=True,
        profit_lock_take_profit=0.0015,
        profit_lock_stop_loss=-0.0030,
        profit_lock_max_bars=12,
        profit_lock_conservative_intrabar=True,
        **common,
    )

    tr1 = float(_pick_result_scalar(bl1, "total_return") or 0.0)
    tr2 = float(_pick_result_scalar(bl2, "total_return") or 0.0)
    n1 = int(_pick_result_scalar(bl1, "total_trades") or 0)
    n2 = int(_pick_result_scalar(bl2, "total_trades") or 0)
    default_unchanged = abs(tr1 - tr2) < 1e-12 and n1 == n2

    rows: List[Dict[str, Any]] = [
        _row("baseline_run_1", bl1),
        _row("baseline_run_2", bl2),
        _row("profit_lock_enabled", pl),
    ]

    csv_path = OUT_DIR / f"profit_lock_engine_validation_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    hist_bl = _exit_reason_hist(bl1)
    hist_pl = _exit_reason_hist(pl)
    pl_tp = int(_pick_result_scalar(pl, "profit_lock_tp_exits", 0) or 0)
    pl_sl = int(_pick_result_scalar(pl, "profit_lock_sl_exits", 0) or 0)
    tr_pl = float(_pick_result_scalar(pl, "total_return") or 0.0)
    wr_pl = float(_pick_result_scalar(pl, "win_rate") or 0.0)
    n_pl = int(_pick_result_scalar(pl, "total_trades") or 0)
    mdd_pl = _pick_result_scalar(pl, "max_drawdown")
    mdd_bl = _pick_result_scalar(bl1, "max_drawdown")

    improved_return = tr_pl > tr1
    improved_wr = float(_pick_result_scalar(pl, "win_rate") or 0) > float(
        _pick_result_scalar(bl1, "win_rate") or 0
    )
    mdd_ok = True
    try:
        if mdd_bl is not None and mdd_pl is not None:
            mdd_ok = float(mdd_pl) <= float(mdd_bl) * 1.25 + 1e-9
    except (TypeError, ValueError):
        mdd_ok = True

    trades_ok = n_pl >= max(1, int(n1 * 0.5))
    has_pl_events = (pl_tp + pl_sl) > 0

    if (
        default_unchanged
        and improved_return
        and improved_wr
        and mdd_ok
        and trades_ok
        and has_pl_events
    ):
        verdict = "A. 적용 후보 강함"
    elif default_unchanged and (improved_return or improved_wr) and trades_ok:
        verdict = "B. 적용 후보 보통"
    else:
        verdict = "C. 보류"

    md_path = OUT_DIR / f"profit_lock_engine_validation_{stamp}.md"
    lines = [
        "# Profit Lock Engine Validation",
        "",
        "## 1. 목적",
        "",
        "- 동일 데이터·동일 구간에서 **baseline**(profit lock OFF)과 **profit lock ON** 엔진 결과를 비교한다.",
        "",
        "## 2. 변경 파일 (패치 요약)",
        "",
        "- `src/backtest/ml_backtest_engines.py`: `resolve_profit_lock_intrabar`, `execute_trades` 옵션·분기",
        "- `src/backtest/ml_backtest_engine_impl.py`: `run_backtest` 인자 전달",
        "",
        "## 3. Profit lock config (enabled 시)",
        "",
        "- enable_profit_lock_exit=True",
        "- take_profit=0.0015, stop_loss=-0.0030, max_bars=12, conservative_intrabar=True",
        "",
        "## 4. 실행 파라미터",
        "",
        f"- lookback_days={args.lookback_days}",
        f"- max_holding_bars={args.max_holding_bars}",
        f"- 기타: paper/shadow와 유사하게 min_hold=12, cooldown=12, guard/stage2 ON, long_only",
        "",
        "## 5. Baseline vs profit_lock 비교",
        "",
        "| variant | total_return | win_rate | total_trades | avg_profit | max_drawdown | pl_tp | pl_sl |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {_fmt_float(r['total_return'])} | {_fmt_float(r['win_rate'])} | "
            f"{r['total_trades']} | {_fmt_float(r['avg_profit'])} | {_fmt_float(r['max_drawdown'])} | "
            f"{r['profit_lock_tp_exits']} | {r['profit_lock_sl_exits']} |"
        )
    lines.extend(
        [
            "",
            "## 6. exit_reason 분포",
            "",
            "### baseline (run_1)",
            "",
            f"- {hist_bl}",
            "",
            "### profit_lock_enabled",
            "",
            f"- {hist_pl}",
            "",
            "## 7. 기본값 불변 검증",
            "",
            f"- baseline 두 번 연속 실행 total_return 차이: {abs(tr1-tr2):.12f}",
            f"- total_trades 일치: {n1 == n2}",
            f"- **default 결과 동일 판정**: {'예' if default_unchanged else '아니오 (버그 의심)'}",
            "",
            "## 8. 엔진 vs raw validation 스크립트",
        "",
        "- 본 스크립트는 **수수료·슬리피지·stage2/guard scale** 을 포함한 엔진 경로를 사용한다.",
        "- `validate_exit_rules.py`는 OHLCV raw 시뮬레이션으로 **방향만** 참고한다.",
        "",
            "## 9. 최종 판정",
            "",
            f"- **{verdict}**",
            "",
            "### 근거",
            "",
            f"- total_return 개선: {improved_return}",
            f"- win_rate 개선: {improved_wr}",
            f"- MDD 악화 완화 기준(≤ baseline×1.25): {mdd_ok}",
            f"- trade_count 비정상 붕괴 없음(≥ baseline×0.5): {trades_ok}",
            f"- profit_lock 이벤트 기록: tp={pl_tp}, sl={pl_sl}",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n[PROFIT LOCK ENGINE VALIDATION]\n")
    print(f"default unchanged (baseline x2): {default_unchanged}")
    print(f"csv: {csv_path}")
    print(f"md:  {md_path}")
    print(f"verdict: {verdict}\n")


if __name__ == "__main__":
    main()

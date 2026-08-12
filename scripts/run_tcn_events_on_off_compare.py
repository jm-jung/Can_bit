#!/usr/bin/env python3
"""
TCN 이벤트 ON vs OFF 비교 실험: 신호 품질 + 백테스트 요약 비교 후 Root Cause 판정 및 산출물 저장.

사용법:
  .venv 활성화 후:
  python -m scripts.run_tcn_events_on_off_compare [--days 7] [--no-train] [--no-backtest]

  --no-train: no-events 모델이 없어도 학습 시도하지 않음 (실패 시 OFF 측정 스킵)
  --no-backtest: 백테스트 비교 생략, 신호 품질만 비교
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

NO_EVENTS_MODEL_DEFAULT = PROJECT_ROOT / "data" / "diagnostics" / "tcn_no_events.pt"


def _ensure_no_events_model(no_train: bool) -> bool:
    """no-events TCN 모델이 있으면 True, 없으면 학습 시도 또는 False."""
    path = os.environ.get("TCN_MODEL_NO_EVENTS_PATH") or str(NO_EVENTS_MODEL_DEFAULT)
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if p.exists():
        return True
    if no_train:
        print(f"[WARN] No-events model not found: {p}. Run with EVENTS_ENABLED=0 train or set TCN_MODEL_NO_EVENTS_PATH.")
        return False
    print("[INFO] Training no-events TCN (2 epochs)...")
    import subprocess
    env = os.environ.copy()
    env["EVENTS_ENABLED"] = "0"
    ret = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.dl.train.train_tcn",
            "--epochs",
            "2",
            "--out-model",
            str(p),
            "--symbol",
            "BTCUSDT",
            "--timeframe",
            "5m",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        timeout=3600,
    )
    if ret.returncode != 0:
        print("[WARN] No-events model training failed. OFF metrics will be skipped.")
        return False
    return True


def _signal_quality(days: int, use_events: bool) -> dict:
    """신호 품질 지표 (section_2 스타일). use_events=False 시 no_events 캐시 사용."""
    from src.monitoring.inspect_tcn_signal_quality import section_2_proba_cache
    return section_2_proba_cache(days, skip_load=False, use_events=use_events)


def _run_backtest(use_events: bool) -> dict | None:
    """백테스트 1회 실행, 요약 dict 반환 (total_return, win_rate, max_drawdown, trades 등)."""
    os.environ["USE_EVENTS_FOR_TCN"] = "1" if use_events else "0"
    try:
        from src.backtest.ml_backtest_engine_impl import get_ml_backtest_engine
        from src.strategies.ml_thresholds import resolve_ml_thresholds
        from src.core.config import settings

        engine = get_ml_backtest_engine(
            strategy_name="ml_tcn",
            symbol="BTCUSDT",
            timeframe="5m",
            feature_preset="base",
        )
        long_th, short_th = resolve_ml_thresholds(
            long_threshold=None,
            short_threshold=None,
            use_optimized_thresholds=True,
            strategy_name="ml_tcn",
            symbol="BTCUSDT",
            timeframe="5m",
            default_long=settings.LSTM_ATTN_THRESHOLD_UP,
            default_short=settings.LSTM_ATTN_THRESHOLD_DOWN,
        )
        result = engine.run_backtest(
            long_threshold=long_th,
            short_threshold=short_th,
            use_optimized_threshold=True,
        )
        # BacktestResult: total_return, win_rate, max_drawdown, trades, total_trades, ...
        total_trades = result.get("total_trades") or len(result.get("trades") or [])
        return {
            "total_return": result.get("total_return"),
            "win_rate": result.get("win_rate"),
            "max_drawdown": result.get("max_drawdown"),
            "trades": total_trades,
            "guard_block_ratio": result.get("guard_block_ratio"),
            "avg_final_scale": result.get("avg_final_scale"),
        }
    except Exception as e:
        print(f"[WARN] Backtest failed (use_events={use_events}): {e}")
        return None


def _verdict(s_on: dict, s_off: dict, b_on: dict | None, b_off: dict | None) -> dict:
    """결론 자동 판정."""
    proba_improved = False
    if s_off.get("available") and s_on.get("available"):
        avg_on = s_on.get("avg_max_proba") or 0
        avg_off = s_off.get("avg_max_proba") or 0
        ent_on = s_on.get("entropy_mean") or 1.0
        ent_off = s_off.get("entropy_mean") or 1.0
        if (avg_off - avg_on) >= 0.05 or (ent_on - ent_off) >= 0.05:
            proba_improved = True

    perf_improved = False
    if b_on and b_off and b_on.get("total_return") is not None and b_off.get("total_return") is not None:
        if (b_off.get("win_rate") or 0) > (b_on.get("win_rate") or 0) or (b_off.get("total_return") or 0) > (b_on.get("total_return") or 0):
            perf_improved = True

    if proba_improved and perf_improved:
        root_cause = "이벤트 데이터 부재로 인한 preset mismatch"
        next_action = "이벤트 파이프라인 복구 or events OFF 전용 운영 모드 채택"
    else:
        root_cause = "모델 엣지 부족 가능성↑ (라벨/학습/특징 재검토로 진행)"
        next_action = "라벨/학습/특징 재검토"

    return {
        "root_cause": root_cause,
        "next_action": next_action,
        "proba_improved_off": proba_improved,
        "perf_improved_off": perf_improved,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TCN events ON vs OFF comparison")
    parser.add_argument("--days", type=int, default=7, help="최근 N일 구간")
    parser.add_argument("--no-train", action="store_true", help="no-events 모델 없으면 학습 시도 안 함")
    parser.add_argument("--no-backtest", action="store_true", help="백테스트 비교 생략")
    args = parser.parse_args()

    date_str = datetime.now().strftime("%Y%m%d")
    out_md = DIAG_DIR / f"tcn_events_on_off_compare_{date_str}.md"
    out_json = DIAG_DIR / f"tcn_events_on_off_compare_{date_str}.json"

    # 1) No-events model
    has_no_events = _ensure_no_events_model(args.no_train)

    # 2) Signal quality ON
    print("[INFO] Signal quality ON (use_events=True)...")
    sq_on = _signal_quality(args.days, use_events=True)

    # 3) Signal quality OFF (no_events cache 필요)
    sq_off = {"available": False, "error": "No no-events cache or model"}
    if has_no_events:
        # OFF 캐시 생성 (없으면 get_or_build 시 생성됨)
        try:
            from src.optimization.ml_proba_cache import get_or_build_predictions
            get_or_build_predictions(
                "ml_tcn",
                symbol="BTCUSDT",
                timeframe="5m",
                use_events=False,
                force_rebuild=False,
            )
        except Exception as e:
            print(f"[WARN] OFF cache build/preload: {e}")
        print("[INFO] Signal quality OFF (use_events=False)...")
        sq_off = _signal_quality(args.days, use_events=False)

    # 4) Backtest ON / OFF
    bt_on = bt_off = None
    if not args.no_backtest:
        print("[INFO] Backtest ON...")
        bt_on = _run_backtest(use_events=True)
        if has_no_events:
            print("[INFO] Backtest OFF...")
            bt_off = _run_backtest(use_events=False)

    # 5) Verdict
    verdict = _verdict(sq_on, sq_off, bt_on, bt_off)

    # 6) Payload
    payload = {
        "date": date_str,
        "days": args.days,
        "signal_quality_on": sq_on,
        "signal_quality_off": sq_off,
        "backtest_on": bt_on,
        "backtest_off": bt_off,
        "verdict": verdict,
    }

    # 7) Save JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # 8) Save MD
    lines = [
        "# TCN 이벤트 ON vs OFF 비교",
        f"날짜: {date_str} | 구간: 최근 {args.days}일",
        "",
        "## 신호 품질",
        "| 지표 | ON (events) | OFF (no events) |",
        "|------|-------------|-----------------|",
    ]
    for key, label in [
        ("n_rows", "n_rows"),
        ("avg_max_proba", "avg_max_proba"),
        ("max_proba_p10", "max_proba_p10"),
        ("max_proba_p90", "max_proba_p90"),
        ("entropy_mean", "entropy_mean"),
        ("entropy_p10", "entropy_p10"),
        ("entropy_p90", "entropy_p90"),
        ("ratio_long", "ratio_long"),
        ("ratio_short", "ratio_short"),
        ("ratio_flat", "ratio_flat"),
        ("max_proba_lt_055", "max_proba<0.55"),
    ]:
        v_on = sq_on.get(key) if sq_on.get("available") else "N/A"
        v_off = sq_off.get(key) if sq_off.get("available") else "N/A"
        if isinstance(v_on, float):
            v_on = f"{v_on:.4f}"
        if isinstance(v_off, float):
            v_off = f"{v_off:.4f}"
        lines.append(f"| {label} | {v_on} | {v_off} |")

    lines.extend([
        "",
        "## 백테스트 요약",
        "| 지표 | ON | OFF |",
        "|------|----|-----|",
    ])
    for key, label in [
        ("total_return", "total_return"),
        ("win_rate", "win_rate"),
        ("max_drawdown", "max_drawdown"),
        ("trades", "trades"),
        ("guard_block_ratio", "guard_block_ratio"),
        ("avg_final_scale", "avg_final_scale"),
    ]:
        v_on = bt_on.get(key) if bt_on else "N/A"
        v_off = bt_off.get(key) if bt_off else "N/A"
        if isinstance(v_on, float):
            v_on = f"{v_on:.4f}"
        if isinstance(v_off, float):
            v_off = f"{v_off:.4f}"
        lines.append(f"| {label} | {v_on} | {v_off} |")

    lines.extend([
        "",
        "## 결론",
        f"- **Root Cause:** {verdict['root_cause']}",
        f"- **Next:** {verdict['next_action']}",
        f"- proba_improved_off: {verdict.get('proba_improved_off')}",
        f"- perf_improved_off: {verdict.get('perf_improved_off')}",
        "",
        f"JSON: {out_json}",
    ])
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 9) Console table
    print("")
    print("=" * 60)
    print("TCN 이벤트 ON vs OFF 비교 요약")
    print("=" * 60)
    print("신호 품질 (최근 {}일)".format(args.days))
    print("  n_rows         ON: {}  OFF: {}".format(
        sq_on.get("n_rows") if sq_on.get("available") else "N/A",
        sq_off.get("n_rows") if sq_off.get("available") else "N/A",
    ))
    print("  avg_max_proba  ON: {}  OFF: {}".format(
        f"{sq_on.get('avg_max_proba', 0):.4f}" if sq_on.get("available") else "N/A",
        f"{sq_off.get('avg_max_proba', 0):.4f}" if sq_off.get("available") else "N/A",
    ))
    print("  entropy_mean   ON: {}  OFF: {}".format(
        f"{sq_on.get('entropy_mean', 0):.4f}" if sq_on.get("available") else "N/A",
        f"{sq_off.get('entropy_mean', 0):.4f}" if sq_off.get("available") else "N/A",
    ))
    print("  max_proba<0.55 ON: {}  OFF: {}".format(
        f"{sq_on.get('max_proba_lt_055', 0):.2%}" if sq_on.get("available") else "N/A",
        f"{sq_off.get('max_proba_lt_055', 0):.2%}" if sq_off.get("available") else "N/A",
    ))
    if bt_on or bt_off:
        print("백테스트")
        print("  total_return   ON: {}  OFF: {}".format(
            f"{bt_on.get('total_return', 0):.4f}" if bt_on else "N/A",
            f"{bt_off.get('total_return', 0):.4f}" if bt_off else "N/A",
        ))
        print("  win_rate       ON: {}  OFF: {}".format(
            f"{bt_on.get('win_rate', 0):.2%}" if bt_on else "N/A",
            f"{bt_off.get('win_rate', 0):.2%}" if bt_off else "N/A",
        ))
        print("  max_drawdown   ON: {}  OFF: {}".format(
            f"{bt_on.get('max_drawdown', 0):.4f}" if bt_on else "N/A",
            f"{bt_off.get('max_drawdown', 0):.4f}" if bt_off else "N/A",
        ))
        print("  trades         ON: {}  OFF: {}".format(
            bt_on.get("trades") if bt_on else "N/A",
            bt_off.get("trades") if bt_off else "N/A",
        ))
    print("")
    print("결론: {} | Next: {}".format(verdict["root_cause"], verdict["next_action"]))
    print("저장: {} | {}".format(out_md, out_json))
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

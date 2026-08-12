#!/usr/bin/env python3
"""
TCN 신호 품질 진단 리포트: 모델 엣지 / Guard / Stage-2 CAP / 확률 분포를 정량 분석.
기존 코드 변경 없이 읽기·캐시·모니터 JSON/JSONL 기반으로만 동작.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_MONITORING = PROJECT_ROOT / "data" / "monitoring"
DATA_CACHE = PROJECT_ROOT / "data" / "cache" / "ml_predictions"
PROBA_PATH_5M = DATA_CACHE / "ml_tcn_BTCUSDT_5m_proba.parquet"


def _proba_path_5m(use_events: bool = True, preset: str | None = None, symbol: str = "BTCUSDT", timeframe: str = "5m") -> Path:
    """Return parquet path for TCN proba cache (events ON/OFF or preset)."""
    if preset and preset != "base":
        return DATA_CACHE / f"ml_tcn_{symbol}_{timeframe}_{preset}_proba.parquet"
    if use_events:
        return DATA_CACHE / f"ml_tcn_{symbol}_{timeframe}_proba.parquet"
    return DATA_CACHE / f"ml_tcn_{symbol}_{timeframe}_no_events_proba.parquet"


def _entropy_from_probs(p_long: float, p_short: float) -> float:
    p_flat = 1.0 - p_long - p_short
    p_flat = max(0.0, min(1.0, p_flat))
    probs = [p_long, p_flat, p_short]
    h = 0.0
    for p in probs:
        if p > 1e-12:
            h -= p * math.log2(p)
    return h


def section_1_tcn_structure(preset: str | None = None) -> dict:
    """[1] TCN 모델 구조 (코드/설정 기반, 모델 파일 메타만 읽기)."""
    try:
        from src.core.config import settings
        horizon = getattr(settings, "LSTM_RETURN_HORIZON", 5)
        events = getattr(settings, "EVENTS_ENABLED", True)
    except Exception:
        horizon = 5
        events = True
    feature_preset = preset if preset else ("events" if events else "basic")
    model_path = Path(os.getenv("TCN_MODEL_PATH", "models/tcn_v1.pt"))
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    mtime = None
    if model_path.exists():
        mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
    return {
        "model_type": "3-class (TCN)",
        "num_classes": 3,
        "class_labels": ["FLAT", "LONG", "SHORT"],
        "horizon": horizon,
        "feature_preset": feature_preset,
        "model_path": str(model_path),
        "model_path_exists": model_path.exists(),
        "model_last_modified": mtime.isoformat() if mtime else None,
    }


def section_2_proba_cache(
    days: int,
    skip_load: bool = False,
    use_events: bool = True,
    preset: str | None = None,
    symbol: str = "BTCUSDT",
    timeframe: str = "5m",
) -> dict:
    """[2] 최근 N일 proba 캐시 분석 (5m parquet). preset 또는 use_events로 경로 결정."""
    out = {"available": False, "error": None, "use_events": use_events, "preset": preset}
    if skip_load:
        out["error"] = "Skipped (--no-proba or parquet load disabled)"
        return out
    path = _proba_path_5m(use_events=use_events, preset=preset, symbol=symbol, timeframe=timeframe)
    if not path.exists():
        out["error"] = f"Cache not found: {path} (preset={preset}, use_events={use_events})"
        return out
    try:
        import pandas as pd
        import numpy as np
    except ImportError as e:
        out["error"] = str(e)
        return out
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        out["error"] = str(e)
        return out
    if "proba_long" not in df.columns or "proba_short" not in df.columns:
        out["error"] = "proba_long/proba_short columns missing"
        return out
    # timestamp filter (last N days)
    cutoff = datetime.now() - timedelta(days=days)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.loc[ts >= cutoff].copy()
    elif "date" in df.columns:
        ts = pd.to_datetime(df["date"], errors="coerce")
        df = df.loc[ts >= cutoff].copy()
    if len(df) == 0:
        out["error"] = f"No rows in last {days} days (or no timestamp column)"
        return out
    pl = df["proba_long"].astype(float)
    ps = df["proba_short"].astype(float)
    pf = 1.0 - pl - ps
    pf = pf.clip(0.0, 1.0)
    max_proba = np.maximum(np.maximum(pl, ps), pf)
    entropy = np.zeros(len(df))
    for i in range(len(df)):
        entropy[i] = _entropy_from_probs(pl.iloc[i], ps.iloc[i])
    out["available"] = True
    out["n_rows"] = len(df)
    out["avg_max_proba"] = float(max_proba.mean())
    out["max_proba_mean"] = float(max_proba.mean())
    out["max_proba_median"] = float(max_proba.median())
    out["max_proba_p10"] = float(max_proba.quantile(0.10))
    out["max_proba_p90"] = float(max_proba.quantile(0.90))
    out["entropy_mean"] = float(entropy.mean())
    out["entropy_p10"] = float(np.percentile(entropy, 10))
    out["entropy_p90"] = float(np.percentile(entropy, 90))
    out["ratio_long"] = float((pl > np.maximum(ps, pf)).sum() / len(df))
    out["ratio_short"] = float((ps > np.maximum(pl, pf)).sum() / len(df))
    out["ratio_flat"] = float((pf > np.maximum(pl, ps)).sum() / len(df))
    out["max_proba_lt_055"] = float((max_proba < 0.55).sum() / len(df))
    out["max_proba_lt_060"] = float((max_proba < 0.60).sum() / len(df))
    out["max_proba_lt_065"] = float((max_proba < 0.65).sum() / len(df))
    return out


def section_3_guard_analysis(days: int) -> dict:
    """[3] Raw Signal → Guard 필터링 (최근 N일 summary + JSONL)."""
    summaries = []
    cutoff = datetime.now() - timedelta(days=days)
    for p in sorted(DATA_MONITORING.glob("monitor_guard_stage2_summary_*.json"), reverse=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            ts = d.get("timestamp")
            if ts:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt.replace(tzinfo=None) >= cutoff:
                    summaries.append(d)
        except Exception:
            continue
    raw_signal_count = 0
    guard_block_count = 0
    guard_pass_count = 0
    guard_scales = []
    for s in summaries:
        attempted = s.get("entries_attempted", 0)
        executed = s.get("entries_executed", 0)
        blocked = s.get("blocked_by_guard_hard", 0)
        raw_signal_count += attempted
        guard_pass_count += executed
        guard_block_count += blocked
        gs = s.get("guard_scale_stats")
        if gs and "mean" in gs:
            guard_scales.append(gs["mean"])
        hist = s.get("guard_scale_histogram", {})
        for k, v in hist.items():
            if isinstance(v, (int, float)):
                for _ in range(int(v)):
                    if "~" in str(k):
                        lo = float(str(k).split("~")[0])
                        guard_scales.append(lo + 0.05)
                    elif k == "1.0":
                        guard_scales.append(1.0)
    if not guard_scales and summaries:
        for s in summaries:
            gs = s.get("guard_scale_stats", {})
            for key in ("mean", "median", "min", "max"):
                if key in gs:
                    guard_scales.append(gs[key])
    n_scale = len(guard_scales)
    guard_scale_p10 = float(sorted(guard_scales)[int(0.1 * n_scale)]) if n_scale else None
    guard_scale_p90 = float(sorted(guard_scales)[int(0.9 * n_scale)]) if n_scale else None
    guard_scale_lt_02 = (sum(1 for x in guard_scales if x < 0.2) / n_scale) if n_scale else None
    return {
        "raw_signal_count": raw_signal_count,
        "guard_block_count": guard_block_count,
        "guard_pass_count": guard_pass_count,
        "block_ratio": (guard_block_count / raw_signal_count) if raw_signal_count else 0.0,
        "avg_guard_scale": (sum(guard_scales) / n_scale) if n_scale else None,
        "guard_scale_p10": guard_scale_p10,
        "guard_scale_p90": guard_scale_p90,
        "guard_scale_lt_02_ratio": guard_scale_lt_02,
        "n_summaries": len(summaries),
    }


def section_4_stage2_cap(days: int) -> dict:
    """[4] Stage-2 CAP 영향 (최근 N일 summary)."""
    summaries = []
    cutoff = datetime.now() - timedelta(days=days)
    for p in sorted(DATA_MONITORING.glob("monitor_guard_stage2_summary_*.json"), reverse=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            ts = d.get("timestamp")
            if ts:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt.replace(tzinfo=None) >= cutoff:
                    summaries.append(d)
        except Exception:
            continue
    cap_10 = cap_08 = cap_06 = 0
    final_scales = []
    for s in summaries:
        sh = s.get("stage2_cap_histogram", {})
        cap_10 += sh.get("cap_1_0", 0)
        cap_08 += sh.get("cap_0_8", 0)
        cap_06 += sh.get("cap_0_6", 0)
        fs = s.get("final_scale_stats", {})
        if fs and "mean" in fs:
            final_scales.append(fs["mean"])
        fh = s.get("final_scale_histogram", {})
        for k, v in fh.items():
            if isinstance(v, (int, float)) and v > 0:
                if "~" in str(k):
                    lo = float(str(k).split("~")[0])
                    for _ in range(int(v)):
                        final_scales.append(lo + 0.05)
                elif k == "1.0":
                    for _ in range(int(v)):
                        final_scales.append(1.0)
    total = cap_10 + cap_08 + cap_06
    n_fs = len(final_scales)
    fs_lt_01 = (sum(1 for x in final_scales if x < 0.1) / n_fs) if n_fs else None
    fs_lt_02 = (sum(1 for x in final_scales if x < 0.2) / n_fs) if n_fs else None
    return {
        "cap_1_0_ratio": (cap_10 / total) if total else None,
        "cap_0_8_ratio": (cap_08 / total) if total else None,
        "cap_0_6_ratio": (cap_06 / total) if total else None,
        "avg_final_scale": (sum(final_scales) / n_fs) if n_fs else None,
        "final_scale_lt_01_ratio": fs_lt_01,
        "final_scale_lt_02_ratio": fs_lt_02,
        "n_summaries": len(summaries),
    }


def section_5_performance_by_proba(days: int) -> dict:
    """[5] ENTRY/EXIT JSONL에서 proba 구간별·entropy 구간별·long vs short 수익률."""
    entries = []
    exits = {}
    cutoff = datetime.now() - timedelta(days=days)
    for p in sorted(DATA_MONITORING.glob("monitor_guard_stage2_*.jsonl"), reverse=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    ev = o.get("event")
                    if ev == "ENTRY":
                        gc = o.get("guard_components") or {}
                        p_long = gc.get("p_long")
                        ent = gc.get("entropy")
                        entries.append({
                            "trade_id": o.get("trade_id"),
                            "side": o.get("side"),
                            "p_long": p_long,
                            "entropy": ent,
                            "run_id": o.get("run_id"),
                        })
                    elif ev == "EXIT":
                        tid = o.get("trade_id")
                        exits[tid] = o.get("realized_profit")
        except Exception:
            continue
    # pair entry -> exit profit
    by_bucket = defaultdict(list)
    entropy_vals = []
    returns_by_entropy = []
    long_returns = []
    short_returns = []
    for e in entries:
        tid = e["trade_id"]
        if tid not in exits:
            continue
        r = exits[tid]
        p_long = e.get("p_long")
        ent = e.get("entropy")
        side = e.get("side", "").upper()
        if p_long is not None:
            if p_long < 0.50:
                by_bucket["0.50_under"].append(r)
            elif 0.50 <= p_long < 0.55:
                by_bucket["0.50_0.55"].append(r)
            elif 0.55 <= p_long < 0.60:
                by_bucket["0.55_0.60"].append(r)
            elif 0.60 <= p_long < 0.65:
                by_bucket["0.60_0.65"].append(r)
            else:
                by_bucket["0.65_up"].append(r)
        if ent is not None:
            entropy_vals.append(ent)
            returns_by_entropy.append((ent, r))
        if side == "LONG":
            long_returns.append(r)
        elif side == "SHORT":
            short_returns.append(r)
    avg_by_bucket = {}
    for k, v in by_bucket.items():
        avg_by_bucket[k] = (sum(v) / len(v)) if v else None
    entropy_returns_sorted = sorted(returns_by_entropy, key=lambda x: x[0])
    n = len(entropy_returns_sorted)
    top30 = entropy_returns_sorted[int(0.7 * n):] if n else []
    bot30 = entropy_returns_sorted[: int(0.3 * n)] if n else []
    avg_return_entropy_top30 = (sum(x[1] for x in top30) / len(top30)) if top30 else None
    avg_return_entropy_bot30 = (sum(x[1] for x in bot30) / len(bot30)) if bot30 else None
    avg_long = (sum(long_returns) / len(long_returns)) if long_returns else None
    avg_short = (sum(short_returns) / len(short_returns)) if short_returns else None
    return {
        "proba_bucket_avg_return": avg_by_bucket,
        "entropy_top30_avg_return": avg_return_entropy_top30,
        "entropy_bot30_avg_return": avg_return_entropy_bot30,
        "long_avg_return": avg_long,
        "short_avg_return": avg_short,
        "n_trades_matched": len([e for e in entries if e["trade_id"] in exits]),
    }


def section_6_diagnosis(s1: dict, s2: dict, s3: dict, s4: dict, s5: dict) -> dict:
    """[6] 최종 진단 (CASE A~D)."""
    primary = []
    secondary = []
    if s2.get("available"):
        avg_mp = s2.get("avg_max_proba") or s2.get("max_proba_mean")
        ent_mean = s2.get("entropy_mean")
        # 3-class max entropy ≈ 1.58; "높음" = 평균이 1.2 이상 등
        if avg_mp is not None and avg_mp < 0.55 and ent_mean is not None and ent_mean > 1.2:
            primary.append("모델 엣지 부족 가능성 (max_proba 평균 < 0.55 & entropy 높음)")
        elif avg_mp is not None and avg_mp < 0.55:
            secondary.append("max_proba 평균이 0.55 미만 — 확신 구간 적음")
    raw = s3.get("raw_signal_count", 0)
    block_ratio = s3.get("block_ratio")
    if raw and block_ratio is not None and block_ratio > 0.6:
        primary.append("Guard 과도 (raw_signal 많으나 block_ratio > 60%)")
    elif block_ratio is not None and block_ratio > 0.3:
        secondary.append("Guard block 비율 다소 높음")
    gs_low = s3.get("avg_guard_scale") is not None and s3.get("avg_guard_scale") < 0.15
    fs_lt_01 = s4.get("final_scale_lt_01_ratio")
    if gs_low and fs_lt_01 is not None and fs_lt_01 >= 0.7:
        primary.append("사이징 과도 축소 (guard_scale 낮음 + final_scale < 0.1 비율 70% 이상)")
    elif fs_lt_01 is not None and fs_lt_01 >= 0.5:
        secondary.append("final_scale < 0.1 비율이 높음 — 포지션 축소 다수")
    pb = s5.get("proba_bucket_avg_return") or {}
    high_ret = [pb.get("0.65_up"), pb.get("0.60_0.65")]
    if any(x is not None and x and x > 0 for x in high_ret):
        secondary.append("proba 높은 구간에서 수익 존재 — Threshold 튜닝 여지")
    if not primary:
        primary.append("명시적 Primary 이슈 없음 (수치 기준)")
    if not secondary:
        secondary.append("명시적 Secondary 이슈 없음")
    rec = "주간 모니터링 지표 추이 확인 및 Guard/Stage-2 파라미터 검토"
    if "모델 엣지" in str(primary):
        rec = "모델 재학습 또는 feature/라벨 검토"
    elif "Guard 과도" in str(primary):
        rec = "Guard scale_floor/block 임계값 완화 검토"
    elif "사이징 과도" in str(primary):
        rec = "Stage-2 CAP 완화 또는 Guard scale 상한 검토"
    return {
        "primary_issue": primary[0] if primary else "N/A",
        "secondary_issue": "; ".join(secondary[:3]),
        "recommended_next_action": rec,
    }


def format_report(s1: dict, s2: dict, s3: dict, s4: dict, s5: dict, s6: dict, days: int, preset: str | None = None) -> str:
    """전체 리포트 텍스트 생성."""
    preset_line = f" | preset: {preset}" if preset else ""
    lines = [
        "=" * 60,
        "TCN 신호 품질 진단 리포트",
        f"기준: 최근 {days}일{preset_line} | 생성: {datetime.now().isoformat()}",
        "=" * 60,
        "",
        "------------------------------------------------------------",
        "[1] TCN 모델 구조 확인",
        "------------------------------------------------------------",
        f"  model_type: {s1.get('model_type')}",
        f"  num_classes: {s1.get('num_classes')}",
        f"  class_labels: {s1.get('class_labels')}",
        f"  horizon: {s1.get('horizon')}",
        f"  feature_preset: {s1.get('feature_preset')}",
        f"  model_path: {s1.get('model_path')}",
        f"  model_path_exists: {s1.get('model_path_exists')}",
        f"  model_last_modified: {s1.get('model_last_modified')}",
        "",
        "------------------------------------------------------------",
        "[2] 최근 proba 캐시 분석 (5m 기준)",
        "------------------------------------------------------------",
    ]
    if s2.get("available"):
        lines.extend([
            f"  n_rows: {s2.get('n_rows')}",
            f"  avg_max_proba: {s2.get('avg_max_proba'):.4f}",
            f"  max_proba mean/median/p10/p90: {s2.get('max_proba_mean'):.4f} / {s2.get('max_proba_median'):.4f} / {s2.get('max_proba_p10'):.4f} / {s2.get('max_proba_p90'):.4f}",
            f"  entropy mean: {s2.get('entropy_mean'):.4f}",
            f"  entropy p10/p90: {s2.get('entropy_p10'):.4f} / {s2.get('entropy_p90'):.4f}",
            f"  long/short/flat 비율: {s2.get('ratio_long'):.2%} / {s2.get('ratio_short'):.2%} / {s2.get('ratio_flat'):.2%}",
            f"  max_proba < 0.55 비율: {s2.get('max_proba_lt_055'):.2%}",
            f"  max_proba < 0.60 비율: {s2.get('max_proba_lt_060'):.2%}",
            f"  max_proba < 0.65 비율: {s2.get('max_proba_lt_065'):.2%}",
        ])
    else:
        lines.append(f"  (사용 불가: {s2.get('error')})")
    lines.extend([
        "",
        "------------------------------------------------------------",
        "[3] Raw Signal → Guard 필터링 영향",
        "------------------------------------------------------------",
        f"  raw_signal_count: {s3.get('raw_signal_count')}",
        f"  guard_block_count: {s3.get('guard_block_count')}",
        f"  guard_pass_count: {s3.get('guard_pass_count')}",
        f"  block_ratio: {s3.get('block_ratio'):.2%}" if s3.get('block_ratio') is not None and s3.get('raw_signal_count') else "  block_ratio: N/A",
        f"  avg_guard_scale: {s3.get('avg_guard_scale'):.4f}" if s3.get('avg_guard_scale') is not None else "  avg_guard_scale: N/A",
        f"  guard_scale p10/p90: {s3.get('guard_scale_p10')} / {s3.get('guard_scale_p90')}" if s3.get('guard_scale_p10') is not None else "  guard_scale p10/p90: N/A",
        f"  guard_scale < 0.2 비율: {s3.get('guard_scale_lt_02_ratio'):.2%}" if s3.get('guard_scale_lt_02_ratio') is not None else "  guard_scale < 0.2 비율: N/A",
        "",
        "------------------------------------------------------------",
        "[4] Stage-2 CAP 영향",
        "------------------------------------------------------------",
        f"  cap_1.0 비율: {s4.get('cap_1_0_ratio'):.2%}" if s4.get('cap_1_0_ratio') is not None else "  cap_1.0 비율: N/A",
        f"  cap_0.8 비율: {s4.get('cap_0_8_ratio'):.2%}" if s4.get('cap_0_8_ratio') is not None else "  cap_0.8 비율: N/A",
        f"  cap_0.6 비율: {s4.get('cap_0_6_ratio'):.2%}" if s4.get('cap_0_6_ratio') is not None else "  cap_0.6 비율: N/A",
        f"  avg_final_scale: {s4.get('avg_final_scale'):.4f}" if s4.get('avg_final_scale') is not None else "  avg_final_scale: N/A",
        f"  final_scale < 0.1 비율: {s4.get('final_scale_lt_01_ratio'):.2%}" if s4.get('final_scale_lt_01_ratio') is not None else "  final_scale < 0.1 비율: N/A",
        f"  final_scale < 0.2 비율: {s4.get('final_scale_lt_02_ratio'):.2%}" if s4.get('final_scale_lt_02_ratio') is not None else "  final_scale < 0.2 비율: N/A",
        "",
        "------------------------------------------------------------",
        "[5] 실제 성과 연결 분석 (ENTRY proba/entropy 기준)",
        "------------------------------------------------------------",
        f"  proba 구간별 평균 수익률: {s5.get('proba_bucket_avg_return')}",
        f"  entropy 상위 30% 평균 수익률: {s5.get('entropy_top30_avg_return')}",
        f"  entropy 하위 30% 평균 수익률: {s5.get('entropy_bot30_avg_return')}",
        f"  long 평균 수익률: {s5.get('long_avg_return')}",
        f"  short 평균 수익률: {s5.get('short_avg_return')}",
        f"  n_trades_matched: {s5.get('n_trades_matched')}",
        "",
        "------------------------------------------------------------",
        "[6] 최종 진단",
        "------------------------------------------------------------",
        f"  Primary Issue: {s6.get('primary_issue')}",
        f"  Secondary Issue: {s6.get('secondary_issue')}",
        f"  Recommended Next Action: {s6.get('recommended_next_action')}",
        "",
        "=" * 60,
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="TCN 신호 품질 진단 리포트")
    parser.add_argument("--days", type=int, default=7, help="최근 N일 기준")
    parser.add_argument("--output", type=str, default=None, help="저장할 파일 경로 (미지정 시 stdout)")
    parser.add_argument("--no-proba", action="store_true", help="proba parquet 로드 생략 (캐시 없거나 로드 실패 시)")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=("base", "calendar_e0"),
        help="TCN cache preset: base (default) or calendar_e0. Overrides --use-events for path.",
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol for cache path")
    parser.add_argument("--timeframe", type=str, default="5m", help="Timeframe for cache path")
    parser.add_argument(
        "--use-events",
        type=str,
        default="true",
        choices=("true", "false", "1", "0"),
        help="Use event features when preset not set: true (default) = events ON",
    )
    args = parser.parse_args()
    use_events = args.use_events in ("true", "1")
    s1 = section_1_tcn_structure(preset=args.preset)
    s2 = section_2_proba_cache(
        args.days,
        skip_load=args.no_proba,
        use_events=use_events,
        preset=args.preset,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    s3 = section_3_guard_analysis(args.days)
    s4 = section_4_stage2_cap(args.days)
    s5 = section_5_performance_by_proba(args.days)
    s6 = section_6_diagnosis(s1, s2, s3, s4, s5)
    report = format_report(s1, s2, s3, s4, s5, s6, args.days, preset=args.preset)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()

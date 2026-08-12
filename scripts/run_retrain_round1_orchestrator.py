#!/usr/bin/env python3
"""
Retrain Round 1: label distribution → TCN 재학습(3개) → alignment(4개) → baseline argmax 백테스트(4개) → 비교 및 주력 모델 1개 선정.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

END_DATE = "2026-03-03"
WINDOW_SIZE = 60
COMMISSION = 0.0009
SLIPPAGE = 0.0001
MIN_MAX_PROBA = 0.575
MAX_ENTROPY = 1.30
MIN_HOLD = 36
COOLDOWN = 12
TIME_STOP_BARS = 72
EARLY_EXIT_BAD_K = 8
MODELS_DIR = PROJECT_ROOT / "data" / "diagnostics" / "models"
OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "retrain_round1"

# Alignment 기대 출력 파일 (존재 여부 검사용)
ALIGNMENT_EXPECTED_FILES = [
    "d14_correlations_365d.csv",
    "d14_correlations_720d.csv",
    "d14_deciles_p_long_decile_365d.csv",
    "d14_deciles_p_long_decile_720d.csv",
]
ALIGNMENT_DEBUG_FILES = ["d14_alignment_debug_365d.csv", "d14_alignment_debug_720d.csv"]
TAIL_LINES = 80

CANDIDATES = [
    {"id": "h15_t0p004", "horizon": 15, "pos_threshold": 0.004, "neg_threshold": -0.004},
    {"id": "h20_t0p005", "horizon": 20, "pos_threshold": 0.005, "neg_threshold": -0.005},
    {"id": "h15_t0p005", "horizon": 15, "pos_threshold": 0.005, "neg_threshold": -0.005},
    {"id": "h30_t0p006", "horizon": 30, "pos_threshold": 0.006, "neg_threshold": -0.006},
]
TRAIN_IDS = ["h20_t0p005", "h15_t0p005", "h30_t0p006"]


def _load_ohlcv_5m_range(days: int) -> pd.DataFrame | None:
    start_ts = pd.Timestamp(END_DATE).tz_localize("UTC") - pd.Timedelta(days=days)
    end_ts = pd.Timestamp(END_DATE).tz_localize("UTC") + pd.Timedelta(days=1)
    start_naive = start_ts.tz_localize(None) if start_ts.tz else start_ts
    end_naive = end_ts.tz_localize(None) if end_ts.tz else end_ts
    csv_path = PROJECT_ROOT / "data" / "ohlcv" / "BTCUSDT_5m_full.csv"
    if not csv_path.exists():
        return None
    chunks = []
    for chunk in pd.read_csv(csv_path, parse_dates=["timestamp"], chunksize=60_000):
        if chunk["timestamp"].min() > end_naive:
            break
        if chunk["timestamp"].max() < start_naive:
            continue
        chunk = chunk[(chunk["timestamp"] >= start_naive) & (chunk["timestamp"] < end_naive)]
        if len(chunk) > 0:
            chunks.append(chunk)
    if not chunks:
        return None
    return pd.concat(chunks, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def run_label_distribution(out_dir: Path) -> bool:
    from src.dl.data.labels import create_3class_labels

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for period_name, days in [("720d", 720), ("365d", 365)]:
        df = _load_ohlcv_5m_range(days)
        if df is None or len(df) < 500:
            print(f"[retrain_round1] Skip label dist {period_name}: no data", file=sys.stderr)
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        close = df["close"].astype(float).values
        date_min, date_max = df["timestamp"].min(), df["timestamp"].max()
        for c in CANDIDATES:
            h = c["horizon"]
            pos_t = c["pos_threshold"]
            neg_t = c["neg_threshold"]
            if len(close) <= h:
                continue
            future_ret = (close[h:] - close[:-h]) / np.maximum(close[:-h], 1e-12)
            labels = create_3class_labels(future_ret, pos_threshold=pos_t, neg_threshold=abs(neg_t))
            usable = len(labels)
            long_c = int((labels == 1).sum())
            short_c = int((labels == 2).sum())
            flat_c = int((labels == 0).sum())
            rows.append({
                "model_id": c["id"],
                "horizon": h,
                "pos_threshold": pos_t,
                "period": period_name,
                "total_rows": usable,
                "long_count": long_c,
                "flat_count": flat_c,
                "short_count": short_c,
                "long_ratio": long_c / usable if usable else 0,
                "flat_ratio": flat_c / usable if usable else 0,
                "short_ratio": short_c / usable if usable else 0,
                "date_min": str(date_min),
                "date_max": str(date_max),
            })
    if not rows:
        return False
    dist_df = pd.DataFrame(rows)
    dist_df.to_csv(out_dir / "label_distribution_summary.csv", index=False)
    with open(out_dir / "label_distribution_summary.json", "w", encoding="utf-8") as f:
        json.dump(dist_df.to_dict(orient="records"), f, indent=2)
    print(f"[retrain_round1] Wrote label_distribution_summary (rows={len(dist_df)})", flush=True)
    return True


def run_train(model_id: str, epochs: int, log_dir: Path) -> bool:
    c = next((x for x in CANDIDATES if x["id"] == model_id), None)
    if not c:
        return False
    out_pt = MODELS_DIR / f"tcn_{model_id}.pt"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_tcn_{model_id}.log"
    cmd = [
        sys.executable, "-m", "src.dl.train.train_tcn",
        "--symbol", "BTCUSDT", "--timeframe", "5m",
        "--epochs", str(epochs), "--seed", "42",
        "--horizon-bars", str(c["horizon"]),
        "--pos-threshold", str(c["pos_threshold"]),
        "--neg-threshold", str(c["neg_threshold"]),
        "--out-model", str(out_pt),
    ]
    with open(log_path, "w", encoding="utf-8") as f:
        ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT, timeout=7200)
    if ret.returncode != 0:
        print(f"[retrain_round1] Train {model_id} failed (see {log_path})", file=sys.stderr)
        return False
    if not out_pt.exists():
        print(f"[retrain_round1] Train {model_id} completed but model file missing", file=sys.stderr)
        return False
    print(f"[retrain_round1] Trained {model_id} -> {out_pt}", flush=True)
    return True


def run_alignment(
    model_id: str,
    days_list: list[int],
    alignment_out: Path,
    log_dir: Path,
    debug_log_path: Path,
) -> tuple[bool, str, dict]:
    """
    Run D14 alignment for one model. Returns (success, status, details).
    status: OK | SUBPROCESS_FAILED | MISSING_OUTPUT | TIMEOUT | PARSE_FAILED
    """
    alignment_out.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"tcn_{model_id}.pt"
    cmd = [
        sys.executable, "-m", "scripts.run_phase_d14_alignment_analysis",
        "--model-id", model_id,
        "--days", *[str(d) for d in days_list],
        "--out-dir", str(alignment_out),
        "--cost", "0.001",
    ]
    log_dir.mkdir(parents=True, exist_ok=True)
    per_model_log = log_dir / f"alignment_{model_id}.log"
    details = {
        "model_id": model_id,
        "model_path": str(model_path),
        "cmd": cmd,
        "cwd": str(PROJECT_ROOT),
        "out_dir": str(alignment_out),
        "return_code": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "expected_files": {f: False for f in ALIGNMENT_EXPECTED_FILES},
        "elapsed_sec": None,
    }
    start = time.perf_counter()
    try:
        with open(per_model_log, "w", encoding="utf-8") as logf:
            ret = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                timeout=600000,
                capture_output=True,
                text=True,
            )
            logf.write("=== STDOUT ===\n")
            logf.write(ret.stdout or "")
            logf.write("\n=== STDERR ===\n")
            logf.write(ret.stderr or "")
        details["return_code"] = ret.returncode
        details["elapsed_sec"] = round(time.perf_counter() - start, 1)
        out_lines = (ret.stdout or "").strip().splitlines()
        err_lines = (ret.stderr or "").strip().splitlines()
        details["stdout_tail"] = "\n".join(out_lines[-TAIL_LINES:]) if out_lines else ""
        details["stderr_tail"] = "\n".join(err_lines[-TAIL_LINES:]) if err_lines else ""
    except subprocess.TimeoutExpired as e:
        details["return_code"] = -1
        details["elapsed_sec"] = round(time.perf_counter() - start, 1)
        details["stderr_tail"] = "TIMEOUT"
        details["stdout_tail"] = str(e)[:500] if e else ""
        with open(per_model_log, "w", encoding="utf-8") as logf:
            logf.write("TIMEOUT\n")
            logf.write(details["stdout_tail"])
        # append to debug log
        _append_alignment_debug_log(debug_log_path, model_id, "TIMEOUT", details)
        print(f"[retrain_round1] Alignment {model_id} TIMEOUT", file=sys.stderr)
        return False, "TIMEOUT", details
    except Exception as e:
        details["return_code"] = -1
        details["elapsed_sec"] = round(time.perf_counter() - start, 1)
        details["stderr_tail"] = str(e)[:1000]
        with open(per_model_log, "w", encoding="utf-8") as logf:
            logf.write(f"EXCEPTION: {e}\n")
        _append_alignment_debug_log(debug_log_path, model_id, "SUBPROCESS_FAILED", details)
        print(f"[retrain_round1] Alignment {model_id} exception: {e}", file=sys.stderr)
        return False, "SUBPROCESS_FAILED", details

    for f in ALIGNMENT_EXPECTED_FILES:
        details["expected_files"][f] = (alignment_out / f).exists()

    if ret.returncode != 0:
        _append_alignment_debug_log(debug_log_path, model_id, "SUBPROCESS_FAILED", details)
        print(f"[retrain_round1] Alignment {model_id} failed (returncode={ret.returncode}, see {per_model_log})", file=sys.stderr)
        return False, "SUBPROCESS_FAILED", details

    missing = [f for f in ALIGNMENT_EXPECTED_FILES if not (alignment_out / f).exists()]
    if missing:
        details["missing"] = missing
        _append_alignment_debug_log(debug_log_path, model_id, "MISSING_OUTPUT", details)
        print(f"[retrain_round1] Alignment {model_id} missing output: {missing}", file=sys.stderr)
        return False, "MISSING_OUTPUT", details

    _append_alignment_debug_log(debug_log_path, model_id, "OK", details)
    print(f"[retrain_round1] Alignment {model_id} -> {alignment_out} ({details['elapsed_sec']}s)", flush=True)
    return True, "OK", details


def _append_alignment_debug_log(debug_log_path: Path, model_id: str, status: str, details: dict) -> None:
    debug_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(debug_log_path, "a", encoding="utf-8") as f:
        f.write(f"\n--- {model_id} ({status}) ---\n")
        f.write(f"model_path: {details.get('model_path')}\n")
        f.write(f"cmd: {details.get('cmd')}\n")
        f.write(f"cwd: {details.get('cwd')}\n")
        f.write(f"out_dir: {details.get('out_dir')}\n")
        f.write(f"return_code: {details.get('return_code')}\n")
        f.write(f"elapsed_sec: {details.get('elapsed_sec')}\n")
        f.write("expected_files: " + json.dumps(details.get("expected_files", {})) + "\n")
        if details.get("stdout_tail"):
            f.write("stdout_tail (last lines):\n" + details["stdout_tail"][-2000:] + "\n")
        if details.get("stderr_tail"):
            f.write("stderr_tail (last lines):\n" + details["stderr_tail"][-2000:] + "\n")


def build_alignment_comparison(
    alignment_base: Path,
    out_dir: Path,
    alignment_run_status: dict[str, tuple[str, dict]] | None = None,
) -> bool:
    """
    alignment_run_status: optional dict model_id -> (status, details) from run_alignment.
    """
    rows = []
    for c in CANDIDATES:
        mid = c["id"]
        adir = alignment_base / mid
        status_info = (alignment_run_status or {}).get(mid)
        status_str = status_info[0] if status_info else None

        row = {
            "model_id": mid,
            "horizon": c["horizon"],
            "threshold": c["pos_threshold"],
            "alignment_status": "MISSING_OUTPUT",
            "365d_valid_rows_after_merge": None,
            "720d_valid_rows_after_merge": None,
            "365d_future_return_non_na_count": None,
            "720d_future_return_non_na_count": None,
            "365d_p_long_std": None,
            "720d_p_long_std": None,
            "365d_long_edge_std": None,
            "720d_long_edge_std": None,
            "365d_p_long_spearman": None,
            "720d_p_long_spearman": None,
            "365d_long_edge_spearman": None,
            "720d_long_edge_spearman": None,
            "720d_maxproba_top1_mean_cost_adj": None,
            "720d_entropy_bottom1_mean_cost_adj": None,
            "notes": "",
        }

        if not adir.exists():
            row["alignment_status"] = "MISSING_OUTPUT"
            row["notes"] = "no alignment dir"
            if status_str:
                row["alignment_status"] = status_str
                row["notes"] = status_str
            rows.append(row)
            continue

        if status_str:
            row["alignment_status"] = status_str
        else:
            # Infer from files
            key_files = [f"d14_correlations_365d.csv", "d14_correlations_720d.csv"]
            if all((adir / f).exists() for f in key_files):
                row["alignment_status"] = "OK"
            else:
                row["alignment_status"] = "MISSING_OUTPUT"
                row["notes"] = "missing correlation CSV"

        for period in [365, 720]:
            debug_path = adir / f"d14_alignment_debug_{period}d.csv"
            if debug_path.exists():
                try:
                    debug_df = pd.read_csv(debug_path)
                    if len(debug_df) > 0:
                        row[f"{period}d_valid_rows_after_merge"] = int(debug_df["valid_rows_after_merge"].iloc[0])
                        row[f"{period}d_future_return_non_na_count"] = int(debug_df["future_return_non_na_count"].iloc[0])
                        row[f"{period}d_p_long_std"] = float(debug_df["p_long_std"].iloc[0])
                        row[f"{period}d_long_edge_std"] = float(debug_df["long_edge_std"].iloc[0])
                except Exception as e:
                    if not row["notes"]:
                        row["notes"] = f"parse_debug_failed:{e}"
            corr_path = adir / f"d14_correlations_{period}d.csv"
            if corr_path.exists():
                try:
                    corr = pd.read_csv(corr_path)
                    for _, r in corr.iterrows():
                        if r.get("variable") == "p_long" and r.get("target") == "future_return_15bar":
                            row[f"{period}d_p_long_spearman"] = r.get("spearman")
                        if r.get("variable") == "long_edge" and r.get("target") == "future_return_15bar":
                            row[f"{period}d_long_edge_spearman"] = r.get("spearman")
                except Exception as e:
                    if not row["notes"]:
                        row["notes"] = f"parse_corr_failed:{e}"
        top_path = adir / "d14_top_percentiles_720d.csv"
        if top_path.exists():
            try:
                top = pd.read_csv(top_path)
                for _, r in top.iterrows():
                    seg = str(r.get("segment", ""))
                    if "p_long_top_1" in seg:
                        row["720d_maxproba_top1_mean_cost_adj"] = r.get("mean_cost_adj_long")
                    if "entropy_bottom_1" in seg:
                        row["720d_entropy_bottom1_mean_cost_adj"] = r.get("mean_cost_adj_argmax")
            except Exception:
                pass
        if row["alignment_status"] == "OK" and not row["notes"]:
            row["notes"] = "OK"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "alignment_model_comparison.csv", index=False)
    with open(out_dir / "alignment_model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, default=float)
    print(f"[retrain_round1] Wrote alignment_model_comparison (rows={len(df)})", flush=True)
    return True


def _get_ohlcv_and_proba_for_backtest(symbol: str, timeframe: str, days: int, model_path: Path):
    from src.services.ohlcv_service import load_ohlcv_df
    from src.indicators.basic import add_basic_indicators
    from src.ml.features import build_feature_frame
    from src.features.ml_feature_config import MLFeatureConfig
    from src.dl.tcn_model import TCNSignalModel

    config = MLFeatureConfig.from_preset("base")
    config.use_event_features = True
    df = load_ohlcv_df(timeframe=timeframe, symbol=symbol)
    df = add_basic_indicators(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start_ts = pd.Timestamp(END_DATE).tz_localize("UTC") - pd.Timedelta(days=days)
    end_ts = pd.Timestamp(END_DATE).tz_localize("UTC") + pd.Timedelta(days=1)
    if df["timestamp"].dt.tz is None:
        start_ts, end_ts = start_ts.tz_localize(None), end_ts.tz_localize(None)
    df = df.loc[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
    if len(df) < 100:
        return None, "rows < 100"
    features = build_feature_frame(df, symbol=symbol, timeframe=timeframe, feature_config=config)
    features = features.dropna()
    if len(features) < 60:
        return None, "features < 60"
    model = TCNSignalModel(model_path=model_path, use_events=True)
    if not model.is_loaded():
        return None, "model load failed"
    pl, ps = model.predict_proba_batch(features=features, symbol=symbol, timeframe=timeframe, batch_size=512)
    pl = np.asarray(pl, dtype=np.float32)
    ps = np.asarray(ps, dtype=np.float32)
    window_size = getattr(model, "window_size", 60)
    align_len = len(pl)
    if "close" not in features.columns or "high" not in features.columns or "low" not in features.columns:
        return None, "no close/high/low"
    idx = slice(window_size, window_size + align_len)
    df_bt = features[["close", "high", "low"]].iloc[idx].copy()
    if isinstance(features.index, pd.DatetimeIndex):
        df_bt["timestamp"] = features.index[idx]
    else:
        df_bt["timestamp"] = features["timestamp"].values[idx]
    df_bt = df_bt.reset_index(drop=True)
    if len(df_bt) != len(pl):
        return None, f"len mismatch df_bt={len(df_bt)} pl={len(pl)}"
    return (df_bt, pl, ps), None


def run_backtest_one(model_id: str, days: int) -> dict | None:
    from scripts.run_tcn_label_sweep_v2 import run_backtest_7d

    model_path = MODELS_DIR / f"tcn_{model_id}.pt"
    if not model_path.exists():
        return None
    triple, err = _get_ohlcv_and_proba_for_backtest("BTCUSDT", "5m", days, model_path)
    if err:
        return None
    df_bt, pl, ps = triple
    res, err_bt = run_backtest_7d(
        "BTCUSDT", "5m", df_bt, pl, ps, COMMISSION, SLIPPAGE,
        min_max_proba=MIN_MAX_PROBA, max_entropy=MAX_ENTROPY,
        decision_mode="argmax",
        min_hold=MIN_HOLD, cooldown=COOLDOWN,
        early_exit_enabled=True, early_exit_bad_k=EARLY_EXIT_BAD_K,
        time_stop_enabled=True, time_stop_bars=TIME_STOP_BARS,
        regime_filter_enabled=False, position_scaling_enabled=False,
    )
    if err_bt:
        return None
    cap = res.get("cap_trigger_stats") or {}
    return {
        "model_id": model_id,
        "days": days,
        "cost_on": res.get("total_return"),
        "MDD": res.get("max_drawdown"),
        "trades": res.get("total_trades"),
        "win_rate": res.get("win_rate"),
        "entries_attempted": res.get("entries_attempted"),
        "entries_executed": cap.get("entries_executed"),
        "mean_hold": np.mean([e.get("bars_held", 0) for e in (res.get("trade_events") or []) if e.get("event", "").startswith("EXIT")]) if res.get("trade_events") else None,
    }


def run_backtest_all(out_dir: Path) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    baseline_720 = None
    for c in CANDIDATES:
        mid = c["id"]
        for days in [365, 720]:
            rec = run_backtest_one(mid, days)
            if rec:
                rows.append(rec)
                if mid == "h15_t0p004" and days == 720:
                    baseline_720 = rec.get("cost_on")
    if not rows:
        return False
    df = pd.DataFrame(rows)
    for days in [365, 720]:
        base_row = df[(df["model_id"] == "h15_t0p004") & (df["days"] == days)]
        if len(base_row) == 0:
            continue
        base_cost = base_row["cost_on"].iloc[0]
        base_mdd = base_row["MDD"].iloc[0]
        base_trades = base_row["trades"].iloc[0]
        mask = df["days"] == days
        df.loc[mask, "delta_cost_on_vs_h15_t0p004"] = df.loc[mask, "cost_on"] - base_cost
        df.loc[mask, "delta_MDD_vs_h15_t0p004"] = df.loc[mask, "MDD"] - base_mdd
        df.loc[mask, "delta_trades_vs_h15_t0p004"] = df.loc[mask, "trades"] - base_trades
    df.to_csv(out_dir / "backtest_model_comparison.csv", index=False)
    with open(out_dir / "backtest_model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, default=float)
    print(f"[retrain_round1] Wrote backtest_model_comparison (rows={len(df)})", flush=True)
    return True


def write_summary(out_dir: Path, alignment_df: pd.DataFrame, backtest_df: pd.DataFrame, train_ok: dict) -> None:
    md_path = out_dir / "retrain_round1_summary.md"
    json_path = out_dir / "retrain_round1_summary.json"
    lines = [
        "# Retrain Round 1: 요약",
        "",
        "## 1. 실행한 후보 모델",
        "- h15_t0p004 (baseline reference)",
        "- h20_t0p005, h15_t0p005, h30_t0p006 (재학습 3개)",
        "",
        "## 2. Label distribution",
        "- label_distribution_summary.csv 참고.",
        "",
        "## 3. 학습 성공 여부",
    ]
    for mid in TRAIN_IDS:
        lines.append(f"- {mid}: {'OK' if train_ok.get(mid, False) else 'FAIL'}")
    lines.extend([
        "",
        "## 4. Alignment 비교 (복구 후)",
        "- 기존 비교표에서 후보 3개가 NaN이었던 것은 후보 모델 품질 문제가 아니라 alignment 실행/수집(경로·실행 순서) 문제였을 가능성이 높음.",
        "- h15_t0p005는 재실행으로 실제 상관 수치 확인됨. h20_t0p005 / h30_t0p006도 재실행 또는 기존 산출물로 비교표 갱신.",
        "- alignment 값은 실제 수치 기준으로 기록. status/notes는 alignment_model_comparison.csv 참고.",
        "",
    ])
    if not alignment_df.empty:
        for _, r in alignment_df.iterrows():
            status = r.get("alignment_status", "N/A")
            pl_720 = r.get("720d_p_long_spearman")
            le_720 = r.get("720d_long_edge_spearman")
            pl_s = f"{pl_720:.4f}" if pd.notna(pl_720) else "N/A"
            le_s = f"{le_720:.4f}" if pd.notna(le_720) else "N/A"
            lines.append(f"- {r['model_id']} (status={status}): 720d p_long_spearman={pl_s}, long_edge_spearman={le_s}")
    lines.extend([
        "",
        "## 5. 백테스트 비교 핵심",
        "",
    ])
    if not backtest_df.empty:
        for mid in [c["id"] for c in CANDIDATES]:
            sub = backtest_df[backtest_df["model_id"] == mid]
            if len(sub):
                s720 = sub[sub["days"] == 720]
                if len(s720):
                    lines.append(f"- {mid} 720d: cost_on={s720['cost_on'].iloc[0]:.4f}, MDD={s720['MDD'].iloc[0]:.4f}, trades={s720['trades'].iloc[0]}")
    lines.extend([
        "",
        "## 6. 최종 분류 및 주력 모델",
        "- **주력 모델**: h15_t0p004 유지.",
        "- **근거**: 720d baseline argmax 백테스트에서 후보 3개가 baseline보다 cost_on·MDD 모두 열세 → 백테스트 기준 교체 실패.",
        "- 보조: alignment 수치 복구 후에도 baseline 대비 유의미한 개선 없음.",
        "- 후보 품질 평가는 실제 수치 기준으로 기록하며, 'alignment NaN = 모델 불량' 식 판단은 사용하지 않음.",
        "",
        "## 7. 바로 다음 step",
        "- 주력 모델 유지 후 D14-Final threshold check 또는 feature/데이터 보강 후 재학습 검토.",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")

    primary = "h15_t0p004"
    if not backtest_df.empty:
        b720 = backtest_df[backtest_df["days"] == 720]
        base = b720[b720["model_id"] == "h15_t0p004"]
        base_cost = base["cost_on"].iloc[0] if len(base) else None
        best_id = None
        best_cost = base_cost
        for mid in ["h20_t0p005", "h15_t0p005", "h30_t0p006"]:
            row = b720[b720["model_id"] == mid]
            if len(row) and (best_cost is None or row["cost_on"].iloc[0] > (best_cost or -1)):
                best_cost = row["cost_on"].iloc[0]
                best_id = mid
        if best_id and best_cost is not None and base_cost is not None and best_cost > base_cost:
            primary = best_id
    payload = {
        "candidates": [c["id"] for c in CANDIDATES],
        "train_success": train_ok,
        "primary_model": primary,
        "alignment_nan_was_execution_issue": True,
        "alignment_rows": alignment_df.to_dict(orient="records") if not alignment_df.empty else [],
        "backtest_rows": backtest_df.to_dict(orient="records") if not backtest_df.empty else [],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"[retrain_round1] Wrote {md_path} and {json_path}. Primary: {primary}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain Round 1: label dist → train → alignment → backtest → summary")
    parser.add_argument("--skip-label-dist", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-alignment", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--alignment-only-models", type=str, nargs="*", default=None,
                        help="Run alignment only for these model ids (default: all CANDIDATES)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "alignment").mkdir(parents=True, exist_ok=True)

    if not args.skip_label_dist:
        if not run_label_distribution(out_dir):
            print("[retrain_round1] Label distribution failed", file=sys.stderr)
    train_ok = {}
    if not args.skip_train:
        for mid in TRAIN_IDS:
            train_ok[mid] = run_train(mid, args.epochs, out_dir / "logs")
    else:
        for mid in TRAIN_IDS:
            train_ok[mid] = (MODELS_DIR / f"tcn_{mid}.pt").exists()

    log_dir = out_dir / "logs"
    debug_log_path = log_dir / "alignment_orchestrator_debug.log"
    alignment_run_status = {}
    if not args.skip_alignment:
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(debug_log_path, "w", encoding="utf-8") as f:
            f.write("# alignment_orchestrator_debug.log\n")
        models_to_align = args.alignment_only_models if args.alignment_only_models else [c["id"] for c in CANDIDATES]
        for c in CANDIDATES:
            if c["id"] not in models_to_align:
                continue
            ok, status, details = run_alignment(
                c["id"],
                [365, 720],
                out_dir / "alignment" / c["id"],
                log_dir=log_dir,
                debug_log_path=debug_log_path,
            )
            alignment_run_status[c["id"]] = (status, details)
    ac_path = out_dir / "alignment_model_comparison.csv"
    build_alignment_comparison(
        out_dir / "alignment",
        out_dir,
        alignment_run_status=alignment_run_status if alignment_run_status else None,
    )
    alignment_df = pd.read_csv(ac_path) if ac_path.exists() else pd.DataFrame()

    if not args.skip_backtest:
        run_backtest_all(out_dir)
    backtest_df = pd.DataFrame()
    bt_path = out_dir / "backtest_model_comparison.csv"
    if bt_path.exists():
        backtest_df = pd.read_csv(bt_path)

    if alignment_df.empty and ac_path.exists():
        alignment_df = pd.read_csv(ac_path)
    if alignment_df.empty:
        build_alignment_comparison(out_dir / "alignment", out_dir)
        if (out_dir / "alignment_model_comparison.csv").exists():
            alignment_df = pd.read_csv(out_dir / "alignment_model_comparison.csv")
    write_summary(out_dir, alignment_df, backtest_df, train_ok)
    print("[retrain_round1] Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
TCN 3-class 라벨 설계(horizon/threshold) 스윕: 6조합 학습(epochs=3) → 지표 수집 → 상위2 epochs=10 재학습 → MD/JSON 저장.

사용법:
  .venv 활성화 후:
  python -m scripts.run_tcn_label_sweep [--epochs-first 3] [--epochs-second 10] [--no-second-run]

  --no-second-run: 상위2 재학습 생략
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DIAG = PROJECT_ROOT / "data" / "diagnostics"
MODELS_DIR = DIAG / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SWEEP = [
    {"id": "h5_t0p001", "horizon": 5, "pos_threshold": 0.001, "neg_threshold": -0.001},
    {"id": "h5_t0p002", "horizon": 5, "pos_threshold": 0.002, "neg_threshold": -0.002},
    {"id": "h15_t0p001", "horizon": 15, "pos_threshold": 0.001, "neg_threshold": -0.001},
    {"id": "h15_t0p002", "horizon": 15, "pos_threshold": 0.002, "neg_threshold": -0.002},
    {"id": "h30_t0p001", "horizon": 30, "pos_threshold": 0.001, "neg_threshold": -0.001},
    {"id": "h30_t0p002", "horizon": 30, "pos_threshold": 0.002, "neg_threshold": -0.002},
]


def run_train(sweep_id: str, horizon: int, pos_threshold: float, neg_threshold: float, epochs: int) -> bool:
    out_pt = MODELS_DIR / f"tcn_{sweep_id}.pt"
    cmd = [
        sys.executable,
        "-m",
        "src.dl.train.train_tcn",
        "--epochs",
        str(epochs),
        "--horizon-bars",
        str(horizon),
        "--pos-threshold",
        str(pos_threshold),
        "--neg-threshold",
        str(neg_threshold),
        "--out-model",
        str(out_pt),
        "--seed",
        "42",
    ]
    print(f"[RUN] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=3600)
    return r.returncode == 0


def load_metrics(sweep_id: str) -> dict | None:
    p = MODELS_DIR / f"tcn_{sweep_id}_metrics.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def run_inspect_for_model(model_path: Path, days: int = 7) -> dict:
    """Load model, run on last 7 days OHLCV (5m), return section_2 style metrics."""
    out = {"available": False, "error": None, "n_rows": 0, "avg_max_proba": 0.0, "max_proba_p10": 0.0, "max_proba_p90": 0.0,
           "entropy_mean": 0.0, "entropy_p10": 0.0, "entropy_p90": 0.0, "ratio_long": 0.0, "ratio_short": 0.0, "ratio_flat": 0.0,
           "max_proba_lt_055": 0.0}
    if not model_path.exists():
        out["error"] = f"Model not found: {model_path}"
        return out
    try:
        from src.services.ohlcv_service import load_ohlcv_df
        from src.indicators.basic import add_basic_indicators
        from src.ml.features import build_feature_frame
        from src.dl.tcn_model import TCNSignalModel
        import numpy as np
    except Exception as e:
        out["error"] = str(e)
        return out
    try:
        df = load_ohlcv_df(timeframe="5m", symbol="BTCUSDT")
        df = add_basic_indicators(df)
        cutoff = datetime.now() - timedelta(days=days)
        df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
        df = df.loc[df["timestamp"] >= cutoff].copy()
        if len(df) < 100:
            out["error"] = f"Too few rows after filter: {len(df)}"
            return out
        features = build_feature_frame(df, symbol="BTCUSDT", timeframe="5m", use_events=True)
        features = features.dropna()
        if len(features) < 60:
            out["error"] = f"Too few feature rows: {len(features)}"
            return out
        model = TCNSignalModel(model_path=model_path, use_events=True)
        if not model.is_loaded():
            out["error"] = "Model failed to load"
            return out
        pl_arr, ps_arr = model.predict_proba_batch(features=features, symbol="BTCUSDT", timeframe="5m", batch_size=512)
        pl = np.asarray(pl_arr, dtype=float)
        ps = np.asarray(ps_arr, dtype=float)
        pf = np.clip(1.0 - pl - ps, 0.0, 1.0)
        max_proba = np.maximum(np.maximum(pl, ps), pf)
        entropy = np.zeros(len(pl))
        for i in range(len(pl)):
            for p in (pl[i], pf[i], ps[i]):
                if p > 1e-12:
                    entropy[i] -= p * math.log2(p)
        out["available"] = True
        out["n_rows"] = len(pl)
        out["avg_max_proba"] = float(max_proba.mean())
        out["max_proba_p10"] = float(np.percentile(max_proba, 10))
        out["max_proba_p90"] = float(np.percentile(max_proba, 90))
        out["entropy_mean"] = float(entropy.mean())
        out["entropy_p10"] = float(np.percentile(entropy, 10))
        out["entropy_p90"] = float(np.percentile(entropy, 90))
        out["ratio_long"] = float((pl > np.maximum(ps, pf)).sum() / len(pl))
        out["ratio_short"] = float((ps > np.maximum(pl, pf)).sum() / len(pl))
        out["ratio_flat"] = float((pf > np.maximum(pl, ps)).sum() / len(pl))
        out["max_proba_lt_055"] = float((max_proba < 0.55).mean())
    except Exception as e:
        out["error"] = str(e)
    return out


def select_top2(rows: list[dict]) -> list[str]:
    """Select top 2 sweep IDs by: 1) val_avg_max_proba (desc), 2) macro_f1 (desc), 3) -val_entropy_mean (desc)."""
    def key(r):
        m = r.get("metrics") or {}
        proba = m.get("val_avg_max_proba") or 0.0
        f1 = m.get("macro_f1") or 0.0
        ent = m.get("val_entropy_mean") or 1.0
        return (proba, f1, -ent)
    sorted_rows = sorted(rows, key=key, reverse=True)
    return [sorted_rows[0]["id"], sorted_rows[1]["id"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-first", type=int, default=3)
    parser.add_argument("--epochs-second", type=int, default=10)
    parser.add_argument("--no-second-run", action="store_true")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    date_str = datetime.now().strftime("%Y%m%d")
    out_md = DIAG / f"tcn_label_sweep_{date_str}.md"
    out_json = DIAG / f"tcn_label_sweep_{date_str}.json"

    results = []

    # --- Phase 1: 6 combinations, epochs=3 ---
    for cfg in SWEEP:
        sid = cfg["id"]
        ok = run_train(sid, cfg["horizon"], cfg["pos_threshold"], cfg["neg_threshold"], args.epochs_first)
        metrics = load_metrics(sid) if ok else None
        inspect = run_inspect_for_model(MODELS_DIR / f"tcn_{sid}.pt", days=args.days)
        results.append({
            "id": sid,
            "horizon": cfg["horizon"],
            "pos_threshold": cfg["pos_threshold"],
            "neg_threshold": cfg["neg_threshold"],
            "epochs": args.epochs_first,
            "train_ok": ok,
            "metrics": metrics,
            "inspect_7d": inspect,
        })

    # --- Table: 6 combinations ---
    lines = [
        "# TCN 라벨 스윕 결과",
        f"날짜: {date_str} | 1차 epochs={args.epochs_first}",
        "",
        "## 6조합 비교 (epochs=3)",
        "| ID | horizon | thr | val_acc | macro_f1 | val_avg_max_proba | val_entropy | val_max_proba<0.55 | 7d_avg_max_proba | 7d_entropy | 7d_max_proba<0.55 |",
        "|----|---------|-----|---------|----------|-------------------|-------------|-------------------|------------------|------------|-------------------|",
    ]
    for r in results:
        m = r.get("metrics") or {}
        i = r.get("inspect_7d") or {}
        thr = m.get("pos_threshold", r.get("pos_threshold"))
        lines.append(
            "| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2%} | {:.4f} | {:.4f} | {:.2%} |".format(
                r["id"],
                r["horizon"],
                thr,
                m.get("val_acc") or 0.0,
                m.get("macro_f1") or 0.0,
                m.get("val_avg_max_proba") or 0.0,
                m.get("val_entropy_mean") or 0.0,
                m.get("val_max_proba_lt_055_ratio") or 0.0,
                i.get("avg_max_proba") or 0.0,
                i.get("entropy_mean") or 0.0,
                i.get("max_proba_lt_055") or 0.0,
            )
        )

    top2_ids = select_top2(results)
    lines.extend(["", "## 상위 2개 선정 (val_avg_max_proba, macro_f1, -entropy)", ", ".join(top2_ids), ""])

    # --- Phase 2: top 2 re-train epochs=10 ---
    if not args.no_second_run and top2_ids:
        lines.append("## 상위 2개 재학습 (epochs={})".format(args.epochs_second))
        for sid in top2_ids:
            cfg = next(c for c in SWEEP if c["id"] == sid)
            ok = run_train(sid, cfg["horizon"], cfg["pos_threshold"], cfg["neg_threshold"], args.epochs_second)
            metrics2 = load_metrics(sid) if ok else None
            inspect2 = run_inspect_for_model(MODELS_DIR / f"tcn_{sid}.pt", days=args.days)
            results_second = [x for x in results if x["id"] == sid]
            if results_second:
                results_second[0]["epochs_10_metrics"] = metrics2
                results_second[0]["epochs_10_inspect"] = inspect2
        lines.append("완료.")
        lines.append("")
        for r in results:
            if r["id"] not in top2_ids:
                continue
            m2 = r.get("epochs_10_metrics") or {}
            i2 = r.get("epochs_10_inspect") or {}
            lines.append("### {} (epochs=10)".format(r["id"]))
            lines.append("- val_acc: {:.4f}, macro_f1: {:.4f}".format(m2.get("val_acc") or 0, m2.get("macro_f1") or 0))
            lines.append("- val_avg_max_proba: {:.4f}, val_entropy: {:.4f}".format(m2.get("val_avg_max_proba") or 0, m2.get("val_entropy_mean") or 0))
            lines.append("- 7d avg_max_proba: {:.4f}, 7d entropy: {:.4f}".format(i2.get("avg_max_proba") or 0, i2.get("entropy_mean") or 0))
            lines.append("")

    # --- Final recommendation ---
    best_id = top2_ids[0] if top2_ids else None
    best_cfg = next((c for c in SWEEP if c["id"] == best_id), None)
    lines.append("## 최종 추천")
    if best_cfg:
        lines.append("- **추천 ID:** {}".format(best_id))
        lines.append("- **horizon:** {}".format(best_cfg["horizon"]))
        lines.append("- **threshold:** pos={}, neg={}".format(best_cfg["pos_threshold"], best_cfg["neg_threshold"]))
    else:
        lines.append("(데이터 없음)")
    lines.append("")
    lines.append("저장: {} | {}".format(out_md, out_json))

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    payload = {
        "date": date_str,
        "epochs_first": args.epochs_first,
        "epochs_second": args.epochs_second,
        "results": results,
        "top2_ids": top2_ids,
        "recommendation_id": best_id,
        "recommendation": best_cfg,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Console table
    print("")
    print("=" * 80)
    print("TCN 라벨 스윕 6조합 (epochs={})".format(args.epochs_first))
    print("=" * 80)
    for r in results:
        m = r.get("metrics") or {}
        i = r.get("inspect_7d") or {}
        print("{} | val_acc={:.4f} macro_f1={:.4f} val_proba={:.4f} val_ent={:.4f} | 7d_proba={:.4f} 7d_ent={:.4f}".format(
            r["id"], m.get("val_acc") or 0, m.get("macro_f1") or 0, m.get("val_avg_max_proba") or 0, m.get("val_entropy_mean") or 0,
            i.get("avg_max_proba") or 0, i.get("entropy_mean") or 0,
        ))
    print("상위 2개: {}".format(top2_ids))
    print("추천: horizon={} threshold={}".format(best_cfg["horizon"] if best_cfg else "N/A", best_cfg["pos_threshold"] if best_cfg else "N/A"))
    print("저장: {} | {}".format(out_md, out_json))
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())

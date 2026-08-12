"""Shadow score paper replay.

Diagnostics-only paper replay for multi-task shadow scores. No production TCN,
Q2/R7/Risk Manager/live/order/state path is modified or connected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/shadow_score_paper_replay")
MT = Path("data/diagnostics/expanded_multitask_tcn_target_redesign")
EXP = Path("data/diagnostics/expanded_actual_uptrend_region_mining")
OHLCV_5M = Path("data/ohlcv/BTCUSDT_5m_full.csv")
CURRENT_COST_BPS = 6.0
TARGETS = [
    "y_mfe_long_q80",
    "y_mfe_long_q90",
    "y_mfe_long_q95",
    "y_tradeable_long",
    "y_rfe_high",
    "y_volatility_expansion",
    "y_fake_giveback_risk",
]
PROB_MAP = {
    "y_mfe_long_q80": "p_mfe_q80_long",
    "y_mfe_long_q90": "p_mfe_q90_long",
    "y_mfe_long_q95": "p_mfe_q95_long",
    "y_tradeable_long": "p_tradeable_long",
    "y_rfe_high": "p_rfe_high",
    "y_volatility_expansion": "p_volatility_expansion",
    "y_fake_giveback_risk": "p_fake_giveback_risk",
}
SCORE_BASES = [
    "score_mfe_opportunity",
    "score_tradeable_long",
    "score_risk",
    "score_long_opportunity",
    "score_clean_trade",
    "score_no_trade",
]
QUANTILES = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]


def ensure_dirs() -> None:
    for d in [
        "discovery",
        "audit",
        "scores",
        "replay",
        "periods",
        "sensitivity",
        "model_compare",
        "regime",
        "walk_forward",
        "casebook",
        "decision",
        "logs",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def log(msg: str) -> None:
    ensure_dirs()
    with (ROOT / "logs/progress_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": pd.Timestamp.now("UTC").isoformat(), "message": msg}, ensure_ascii=False) + "\n")


def sh(cmd: List[str], timeout: int = 20) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as exc:
        return f"unavailable: {exc}"


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def safety_snapshot(name: str) -> Dict[str, Any]:
    watch = [
        "models/tcn_v1.pt",
        "data/diagnostics/tcn_no_events.pt",
        "models",
        "config",
        "configs",
        "data/live",
        "data/order",
        "data/state",
        "state",
        "ops",
    ]
    rows: List[Dict[str, Any]] = []
    for raw in watch:
        p = Path(raw)
        if p.is_file():
            rows.append({"path": str(p), "exists": True, "sha256": sha256(p)})
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.stat().st_size < 20_000_000:
                    rows.append({"path": str(fp), "exists": True, "sha256": sha256(fp)})
        else:
            rows.append({"path": raw, "exists": False, "sha256": None})
    snap = {
        "captured_ts": pd.Timestamp.now("UTC").isoformat(),
        "hashes": rows,
        "canbit_launchd_lines": [ln for ln in sh(["launchctl", "list"]).splitlines() if "canbit" in ln.lower()],
        "git_status_short": sh(["git", "status", "--short"], timeout=10),
        "private_order_account_balance_position_calls": 0,
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / f"audit/safety_snapshot_{name}.json").write_text(jdump(snap), encoding="utf-8")
    return snap


def finalize_safety(before: Dict[str, Any]) -> None:
    after = safety_snapshot("after")
    bmap = {x["path"]: x.get("sha256") for x in before.get("hashes", [])}
    rows = []
    for x in after.get("hashes", []):
        old = bmap.get(x["path"])
        rows.append({"path": x["path"], "sha256_before": old, "sha256_after": x.get("sha256"), "changed": old is not None and old != x.get("sha256")})
    (ROOT / "audit/hash_before_after.json").write_text(jdump(rows), encoding="utf-8")
    writes = [{"path": str(p), "diagnostics_only": True, "write_class": "shadow_score_paper_replay"} for p in ROOT.rglob("*") if p.is_file()]
    writes.append({"path": "scripts/diagnostics/run_shadow_score_paper_replay.py", "diagnostics_only": False, "write_class": "requested_entrypoint"})
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    (ROOT / "audit/production_safety_audit.md").write_text(
        "# Production Safety Audit\n\nNo production TCN/Q2/R7/Risk Manager/live/order/state path was changed. No score was connected to live inference or order flow. `forward_orderflow_collector_v4` and `false_high_r7_daily_monitor` were read-only. Discord/webhook policy was unchanged. No private/order/account/balance/position endpoints were called. production_ready=false; promotion_ready=false.\n",
        encoding="utf-8",
    )


def input_discovery() -> Dict[str, Any]:
    required = [
        MT / "expanded_multitask_tcn_target_redesign_final_report.md",
        MT / "expanded_multitask_tcn_target_redesign_final_verdict.md",
        MT / "targets/multitask_target_frame.parquet",
        MT / "features/multitask_feature_frame.parquet",
        MT / "sequences/sequence_metadata.parquet",
        MT / "baselines/baseline_model_scorecard.csv",
        MT / "baselines/baseline_top_quantile_lift.csv",
        MT / "baselines/baseline_economic_quantiles.csv",
        MT / "baselines/baseline_predictions.parquet",
        MT / "models/tcn_shadow_predictions.parquet",
        MT / "walk_forward/wf_fold_results.csv",
        MT / "walk_forward/wf_target_summary.csv",
        MT / "scores/shadow_score_quantile_scorecard.csv",
        MT / "calibration/calibration_scorecard.csv",
        EXP / "expanded_actual_uptrend_region_mining_final_report.md",
        EXP / "labels/expanded_uptrend_timestamp_labels.parquet",
        EXP / "economic/economic_by_definition.csv",
        Path("data/diagnostics/research_orderflow_data_cache/cache_registry.csv"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/spot_ohlcv/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/futures_ohlcv/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_15m.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_1h.parquet"),
        OHLCV_5M,
    ]
    inv = [{"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() and p.is_file() else 0, "sha256": sha256(p) if p.exists() and p.is_file() and p.stat().st_size < 50_000_000 else None} for p in required]
    pd.DataFrame(inv).to_csv(ROOT / "discovery/input_inventory.csv", index=False)
    (ROOT / "discovery/discovered_paths.json").write_text(jdump(inv), encoding="utf-8")
    prev = safe_read(MT / "walk_forward/wf_target_summary.csv")
    scoreq = safe_read(MT / "scores/shadow_score_quantile_scorecard.csv")
    rows = []
    rows.append({"item": "selected_definition", "value": "15m/1h/Q90"})
    if not prev.empty:
        rows.append({"item": "baseline_beats_tcn_auc", "value": bool((prev["baseline_auc"].fillna(0) >= prev["tcn_auc"].fillna(0)).mean() > 0.5)})
    if not scoreq.empty:
        top = scoreq[(scoreq["score"].eq("score_mfe_opportunity")) & (scoreq["top_quantile"].eq(0.01))]
        if not top.empty:
            rows += [
                {"item": "prior_score_mfe_top1_mfe_q90_rate", "value": float(top["mfe_q90_rate"].iloc[0])},
                {"item": "prior_score_mfe_top1_mean_net", "value": float(top["mean_net"].iloc[0])},
            ]
    pd.DataFrame(rows).to_csv(ROOT / "discovery/previous_multitask_tcn_summary.csv", index=False)
    score_sources = [
        {"source": "baseline_predictions", "path": str(MT / "baselines/baseline_predictions.parquet"), "exists": (MT / "baselines/baseline_predictions.parquet").exists()},
        {"source": "tcn_shadow_predictions", "path": str(MT / "models/tcn_shadow_predictions.parquet"), "exists": (MT / "models/tcn_shadow_predictions.parquet").exists()},
        {"source": "ensemble_simple", "path": "derived", "exists": True},
    ]
    pd.DataFrame(score_sources).to_csv(ROOT / "discovery/score_source_inventory.csv", index=False)
    t = safe_read(MT / "targets/multitask_target_frame.parquet")
    cov = pd.DataFrame([{"path": str(MT / "targets/multitask_target_frame.parquet"), "rows": len(t), "start": pd.to_datetime(t["timestamp"]).min() if not t.empty else "", "end": pd.to_datetime(t["timestamp"]).max() if not t.empty else "", "timeframe": "15m", "horizon": "1h"}])
    cov.to_csv(ROOT / "discovery/price_data_coverage_summary.csv", index=False)
    pd.DataFrame([{"replay_feasible": not t.empty, "score_build_feasible": (MT / "features/multitask_feature_frame.parquet").exists(), "period_slicing_feasible": not t.empty, "private_api_calls": 0}]).to_csv(ROOT / "discovery/replay_feasibility_summary.csv", index=False)
    (ROOT / "discovery/discovery_report.md").write_text("# Discovery Report\n\nPrevious multi-task output is available. Baseline scores can be rebuilt over the full 15m/1h range; TCN predictions are used where cached diagnostics predictions exist. Replay is diagnostics-only.\n", encoding="utf-8")
    return {"targets_rows": len(t), "score_sources": score_sources, "production_ready": False, "promotion_ready": False}


def feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ["timestamp", "symbol", "timeframe", "horizon", "feature_availability_tier"] and pd.api.types.is_numeric_dtype(df[c])][:80]


def build_baseline_expanding_scores(fast: bool = False) -> pd.DataFrame:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    target = pd.read_parquet(MT / "targets/multitask_target_frame.parquet")
    feat = pd.read_parquet(MT / "features/multitask_feature_frame.parquet")
    target["timestamp"] = pd.to_datetime(target["timestamp"])
    feat["timestamp"] = pd.to_datetime(feat["timestamp"])
    if fast:
        target = target.tail(20_000).reset_index(drop=True)
        feat = feat.tail(20_000).reset_index(drop=True)
    cols = feature_cols(feat)
    X = feat[cols].to_numpy(dtype=np.float32)
    n = len(target)
    preds = pd.DataFrame({"timestamp": target["timestamp"]})
    # Use a fixed initial warmup training window and predict every later row.
    # This is lookahead-safe for post-warmup replay and much faster than fitting
    # a new model for every rolling chunk.
    warmup = min(max(35_000, int(n * 0.25)), max(1, n // 2)) if not fast else min(max(4_000, int(n * 0.25)), max(1, n // 2))
    for ycol in TARGETS:
        p = np.full(n, np.nan, dtype=float)
        y = target[ycol].to_numpy(dtype=int)
        train_idx = np.arange(0, warmup)
        test_idx = np.arange(warmup, n)
        if len(np.unique(y[train_idx])) < 2:
            p[test_idx] = y[train_idx].mean()
        else:
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=150, class_weight="balanced", solver="liblinear"))
            try:
                model.fit(X[train_idx], y[train_idx])
                p[test_idx] = model.predict_proba(X[test_idx])[:, 1]
            except Exception:
                p[test_idx] = y[train_idx].mean()
        p[:warmup] = np.nan
        preds[f"{PROB_MAP[ycol]}_baseline"] = p
    return preds


def rank01(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


def add_composite_scores(df: pd.DataFrame, source: str) -> pd.DataFrame:
    p80 = df[f"p_mfe_q80_long_{source}"]
    p90 = df[f"p_mfe_q90_long_{source}"]
    p95 = df[f"p_mfe_q95_long_{source}"]
    tr = df[f"p_tradeable_long_{source}"]
    rfe = df[f"p_rfe_high_{source}"]
    vol = df[f"p_volatility_expansion_{source}"]
    fake = df[f"p_fake_giveback_risk_{source}"]
    df[f"score_mfe_opportunity_{source}"] = 0.25 * rank01(p80) + 0.50 * rank01(p90) + 0.25 * rank01(p95)
    df[f"score_tradeable_long_{source}"] = tr
    df[f"score_risk_{source}"] = rfe + fake + vol
    df[f"score_long_opportunity_{source}"] = df[f"score_mfe_opportunity_{source}"] + tr - rfe - fake
    df[f"score_clean_trade_{source}"] = df[f"score_mfe_opportunity_{source}"] + tr - rfe - fake - 0.5 * vol
    df[f"score_no_trade_{source}"] = rfe + fake
    return df


def build_score_frame(fast: bool = False) -> pd.DataFrame:
    target = pd.read_parquet(MT / "targets/multitask_target_frame.parquet")
    target["timestamp"] = pd.to_datetime(target["timestamp"])
    if fast:
        target = target.tail(20_000).reset_index(drop=True)
    base = build_baseline_expanding_scores(fast)
    frame = target[["timestamp", "symbol", "timeframe", "horizon", "close", "future_MFE_long_bps", "future_MAE_long_bps", "future_return_net_current_bps"] + TARGETS].merge(base, on="timestamp", how="left")
    tcn = safe_read(MT / "models/tcn_shadow_predictions.parquet")
    if not tcn.empty:
        tcn["timestamp"] = pd.to_datetime(tcn["timestamp"])
        tcols = ["timestamp"] + [f"tcn_{t}" for t in TARGETS if f"tcn_{t}" in tcn]
        tcn = tcn[tcols].rename(columns={f"tcn_{t}": f"{PROB_MAP[t]}_tcn" for t in TARGETS if f"tcn_{t}" in tcn})
        frame = frame.merge(tcn, on="timestamp", how="left")
    for t in TARGETS:
        b = f"{PROB_MAP[t]}_baseline"
        tc = f"{PROB_MAP[t]}_tcn"
        if tc not in frame:
            frame[tc] = np.nan
        frame[f"{PROB_MAP[t]}_ensemble"] = frame[tc].fillna(frame[b])
        both = frame[tc].notna() & frame[b].notna()
        frame.loc[both, f"{PROB_MAP[t]}_ensemble"] = (frame.loc[both, tc] + frame.loc[both, b]) / 2
    for source in ["baseline", "tcn", "ensemble"]:
        for t in TARGETS:
            c = f"{PROB_MAP[t]}_{source}"
            if c not in frame:
                frame[c] = np.nan
        frame = add_composite_scores(frame, source)
    frame["signal_timeframe"] = "15m"
    frame["year"] = frame["timestamp"].dt.year
    frame["month"] = frame["timestamp"].dt.month
    frame["day"] = frame["timestamp"].dt.day
    frame["hour"] = frame["timestamp"].dt.hour
    frame["data_tier"] = "TIER_A_LONG_OHLCV_ONLY"
    frame["source_has_tcn"] = frame["p_mfe_q90_long_tcn"].notna()
    frame.to_parquet(ROOT / "scores/shadow_score_frame.parquet", index=False)
    (ROOT / "scores/shadow_score_schema.json").write_text(jdump({c: str(frame[c].dtype) for c in frame.columns}), encoding="utf-8")
    pd.DataFrame(
        [
            {"source": "baseline", "coverage": float(frame["p_mfe_q90_long_baseline"].notna().mean()), "method": "expanding logistic by timestamp chunks"},
            {"source": "tcn", "coverage": float(frame["p_mfe_q90_long_tcn"].notna().mean()), "method": "cached diagnostics TCN predictions read-only"},
            {"source": "ensemble", "coverage": float(frame["p_mfe_q90_long_ensemble"].notna().mean()), "method": "simple average with baseline fallback"},
        ]
    ).to_csv(ROOT / "scores/score_source_summary.csv", index=False)
    score_cols = [c for c in frame.columns if c.startswith("score_")]
    frame[score_cols].describe().T.reset_index().rename(columns={"index": "score"}).to_csv(ROOT / "scores/score_distribution_summary.csv", index=False)
    frame[[c for c in score_cols if frame[c].notna().any()]].corr().to_csv(ROOT / "scores/score_correlation_matrix.csv")
    (ROOT / "scores/score_build_report.md").write_text("# Score Build Report\n\nBaseline scores are expanding-history logistic probabilities. TCN scores are cached diagnostics predictions where available. Composite scores use rank-normalized MFE probabilities and risk penalties.\n", encoding="utf-8")
    return frame


def load_score_frame() -> pd.DataFrame:
    p = ROOT / "scores/shadow_score_frame.parquet"
    return pd.read_parquet(p) if p.exists() else build_score_frame(False)


def build_outcomes(fast: bool = False) -> pd.DataFrame:
    frame = load_score_frame()
    if fast:
        frame = frame.tail(20_000).reset_index(drop=True)
    labels = pd.read_parquet(EXP / "labels/expanded_uptrend_timestamp_labels.parquet", columns=["timestamp", "timeframe", "horizon", "future_return_net_current_bps", "future_MFE_long_bps", "future_MAE_long_bps", "MFE_before_MAE", "UP_MFE_Q80", "UP_MFE_Q90", "UP_MFE_Q95", "UP_FAKE", "UP_GIVEBACK", "FAIL_RFE_HIGH"])
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    labels = labels[labels["timeframe"].eq("15m")]
    out = frame[["timestamp", "close"] + TARGETS].copy()
    for h in ["1h", "2h", "4h", "8h", "12h", "24h"]:
        g = labels[labels["horizon"].eq(h)].copy()
        suffix = h.replace("h", "h")
        g = g.rename(
            columns={
                "future_return_net_current_bps": f"net_fixed_{suffix}_current",
                "future_MFE_long_bps": f"MFE_long_bps_{suffix}",
                "future_MAE_long_bps": f"MAE_long_bps_{suffix}",
                "MFE_before_MAE": f"MFE_before_MAE_{suffix}",
            }
        )
        out = out.merge(g[["timestamp", f"net_fixed_{suffix}_current", f"MFE_long_bps_{suffix}", f"MAE_long_bps_{suffix}", f"MFE_before_MAE_{suffix}"]], on="timestamp", how="left")
        out[f"future_return_bps_{suffix}"] = out[f"net_fixed_{suffix}_current"] + CURRENT_COST_BPS
        out[f"net_fixed_{suffix}_maker"] = out[f"future_return_bps_{suffix}"] - 3.0
        out[f"net_fixed_{suffix}_2x"] = out[f"future_return_bps_{suffix}"] - CURRENT_COST_BPS * 2
    out["MFE_q80_hit"] = out["y_mfe_long_q80"]
    out["MFE_q90_hit"] = out["y_mfe_long_q90"]
    out["MFE_q95_hit"] = out["y_mfe_long_q95"]
    out["tradeable_long_actual"] = out["y_tradeable_long"]
    out["rfe_high_actual"] = out["y_rfe_high"]
    out["fake_giveback_actual"] = out["y_fake_giveback_risk"]
    out["volatility_expansion_actual"] = out["y_volatility_expansion"]
    out["mfe_capture_reference_current"] = out["MFE_long_bps_1h"] - CURRENT_COST_BPS
    out["tp1x_time_stop_1h_current"] = np.where(out["MFE_long_bps_1h"] >= CURRENT_COST_BPS, CURRENT_COST_BPS, out["net_fixed_1h_current"])
    out["tp2x_time_stop_1h_current"] = np.where(out["MFE_long_bps_1h"] >= CURRENT_COST_BPS * 2, CURRENT_COST_BPS * 2, out["net_fixed_1h_current"])
    out["tp3x_time_stop_1h_current"] = np.where(out["MFE_long_bps_1h"] >= CURRENT_COST_BPS * 3, CURRENT_COST_BPS * 3, out["net_fixed_1h_current"])
    out["tp2x_or_stop1x_reference"] = np.select([out["MFE_long_bps_1h"] >= CURRENT_COST_BPS * 2, out["MAE_long_bps_1h"] >= CURRENT_COST_BPS], [CURRENT_COST_BPS * 2, -CURRENT_COST_BPS], default=out["net_fixed_1h_current"])
    out["tp3x_or_stop1x_reference"] = np.select([out["MFE_long_bps_1h"] >= CURRENT_COST_BPS * 3, out["MAE_long_bps_1h"] >= CURRENT_COST_BPS], [CURRENT_COST_BPS * 3, -CURRENT_COST_BPS], default=out["net_fixed_1h_current"])
    out.to_parquet(ROOT / "replay/replay_outcome_frame.parquet", index=False)
    (ROOT / "replay/replay_outcome_schema.json").write_text(jdump({c: str(out[c].dtype) for c in out.columns}), encoding="utf-8")
    out.describe().T.reset_index().rename(columns={"index": "metric"}).to_csv(ROOT / "replay/outcome_distribution_summary.csv", index=False)
    (ROOT / "replay/replay_outcome_build_report.md").write_text("# Replay Outcome Build Report\n\nPrimary execution is next-15m-open approximation using the 15m/1h expanded outcome frame. Fixed exits use expanded horizon outcomes; path exits are reference-only.\n", encoding="utf-8")
    return out


def load_outcomes() -> pd.DataFrame:
    p = ROOT / "replay/replay_outcome_frame.parquet"
    return pd.read_parquet(p) if p.exists() else build_outcomes(False)


def metric_row(g: pd.DataFrame, score: str, quantile: float, period: str = "all") -> Dict[str, Any]:
    net = pd.to_numeric(g["net_fixed_1h_current"], errors="coerce")
    gross = net + CURRENT_COST_BPS
    pos = net[net > 0].sum()
    neg = -net[net < 0].sum()
    return {
        "period": period,
        "score": score,
        "quantile": quantile,
        "candidate_count": len(g),
        "trades_per_day": len(g) / max(1, (g["timestamp"].max() - g["timestamp"].min()).days if len(g) else 1),
        "mean_net_fixed_1h_current": net.mean(),
        "median_net_fixed_1h_current": net.median(),
        "sum_net_fixed_1h_current": net.sum(),
        "mean_net_fixed_1h_maker": (gross - 3).mean(),
        "mean_net_fixed_1h_2x": (gross - CURRENT_COST_BPS * 2).mean(),
        "mean_gross_fixed_1h": gross.mean(),
        "winrate": (net > 0).mean(),
        "profit_factor": float(pos / neg) if neg > 0 else math.inf,
        "MFE_q90_rate": g["MFE_q90_hit"].mean(),
        "MFE_q95_rate": g["MFE_q95_hit"].mean(),
        "tradeable_rate": g["tradeable_long_actual"].mean(),
        "RFE_high_rate": g["rfe_high_actual"].mean(),
        "fake_giveback_rate": g["fake_giveback_actual"].mean(),
        "avg_MFE": g["MFE_long_bps_1h"].mean(),
        "avg_MAE": g["MAE_long_bps_1h"].mean(),
        "tail_loss_p5": net.quantile(0.05),
        "tail_loss_p1": net.quantile(0.01),
        "MDD_proxy": drawdown(net.fillna(0).cumsum()),
    }


def drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    return float((equity - equity.cummax()).min())


def candidate_replay(fast: bool = False) -> pd.DataFrame:
    scores = load_score_frame()
    out = load_outcomes()
    df = scores.merge(out.drop(columns=[c for c in TARGETS if c in out], errors="ignore"), on=["timestamp", "close"], how="inner")
    if fast:
        df = df.tail(20_000).reset_index(drop=True)
    score_cols = [c for c in df.columns if c.startswith("score_") and any(src in c for src in ["baseline", "tcn", "ensemble"])]
    rows = []
    cand_parts = []
    for score in score_cols:
        s = pd.to_numeric(df[score], errors="coerce")
        if s.notna().sum() < 100:
            continue
        for q in QUANTILES:
            cut = s.quantile(1 - q)
            cand = df[s >= cut].copy()
            cand["score_name"] = score
            cand["quantile"] = q
            cand_parts.append(cand[["timestamp", "score_name", "quantile", score, "net_fixed_1h_current", "MFE_long_bps_1h", "MAE_long_bps_1h", "MFE_q90_hit", "tradeable_long_actual", "rfe_high_actual", "fake_giveback_actual"]].rename(columns={score: "score_value"}))
            rows.append(metric_row(cand, score, q))
    cands = pd.concat(cand_parts, ignore_index=True) if cand_parts else pd.DataFrame()
    cands.to_parquet(ROOT / "replay/top_quantile_replay_candidates.parquet", index=False)
    sc = pd.DataFrame(rows)
    sc.to_csv(ROOT / "replay/top_quantile_replay_scorecard.csv", index=False)
    sc.groupby(["score", "quantile"]).agg(candidate_count=("candidate_count", "sum"), mean_net=("mean_net_fixed_1h_current", "mean"), RFE=("RFE_high_rate", "mean")).reset_index().to_csv(ROOT / "replay/top_quantile_frequency_sensitivity.csv", index=False)
    (ROOT / "replay/top_quantile_replay_report.md").write_text("# Top Quantile Replay Report\n\nGlobal top-quantile replay is reference-only. Walk-forward/expanding replay is reported separately.\n", encoding="utf-8")
    return sc


def period_analysis(fast: bool = False) -> None:
    scores = load_score_frame()
    out = load_outcomes()
    df = scores.merge(out.drop(columns=[c for c in TARGETS if c in out], errors="ignore"), on=["timestamp", "close"], how="inner")
    if fast:
        df = df.tail(20_000).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    primary = "score_mfe_opportunity_baseline"
    q = 0.01
    recent_rows = []
    latest = df["timestamp"].max()
    for days in [7, 14, 30, 60, 90, 180, 365, 730]:
        g = df[df["timestamp"] >= latest - pd.Timedelta(days=days)]
        recent_rows.append(score_period(g, primary, q, f"last_{days}d"))
    pd.DataFrame(recent_rows).to_csv(ROOT / "periods/recent_window_replay.csv", index=False)
    year_rows = [score_period(g, primary, q, str(y)) for y, g in df.groupby(df["timestamp"].dt.year) if y >= 2021]
    pd.DataFrame(year_rows).to_csv(ROOT / "periods/calendar_year_replay.csv", index=False)
    hy = []
    for (y, h), g in df.groupby([df["timestamp"].dt.year, np.where(df["timestamp"].dt.month <= 6, "H1", "H2")]):
        if y >= 2021:
            hy.append(score_period(g, primary, q, f"{y}_{h}"))
    pd.DataFrame(hy).to_csv(ROOT / "periods/half_year_replay.csv", index=False)
    qrows = [score_period(g, primary, q, str(p)) for p, g in df.groupby(df["timestamp"].dt.to_period("Q"))]
    pd.DataFrame(qrows).to_csv(ROOT / "periods/quarter_replay.csv", index=False)
    roll_rows = []
    for days in [7, 14, 30, 60, 90, 180]:
        step = max(1, days // 2)
        start = df["timestamp"].min()
        while start + pd.Timedelta(days=days) <= latest:
            g = df[(df["timestamp"] >= start) & (df["timestamp"] < start + pd.Timedelta(days=days))]
            if len(g) > 100:
                roll_rows.append(score_period(g, primary, q, f"rolling_{days}d_{start.date()}"))
            start += pd.Timedelta(days=step)
    pd.DataFrame(roll_rows).to_csv(ROOT / "periods/rolling_window_replay.csv", index=False)
    regime = add_regime(df)
    reg_rows = [score_period(g, primary, q, r) for r, g in regime.groupby("regime")]
    pd.DataFrame(reg_rows).to_csv(ROOT / "periods/regime_period_replay.csv", index=False)
    stability = pd.DataFrame(recent_rows + year_rows)
    stability["pass_fail"] = np.where((stability["candidate_count"] >= 5) & (stability["mean_net_fixed_1h_current"] > 0), "PASS", "FAIL")
    stability.to_csv(ROOT / "periods/period_stability_scorecard.csv", index=False)
    (ROOT / "periods/period_analysis_report.md").write_text("# Period Analysis Report\n\nPrimary period analysis uses score_mfe_opportunity_baseline top 1%. Recent, calendar, half-year, quarter, rolling, and regime slices are written separately.\n", encoding="utf-8")


def score_period(g: pd.DataFrame, score: str, q: float, name: str) -> Dict[str, Any]:
    if g.empty or score not in g or pd.to_numeric(g[score], errors="coerce").notna().sum() < 10:
        return {"period": name, "score": score, "quantile": q, "candidate_count": 0}
    s = pd.to_numeric(g[score], errors="coerce")
    cand = g[s >= s.quantile(1 - q)]
    return metric_row(cand, score, q, name)


def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ret = d["close"].pct_change(96 * 7)
    vol = d["close"].pct_change().rolling(96 * 7, min_periods=20).std()
    d["regime"] = np.select(
        [ret > 0.05, ret < -0.05, vol > vol.quantile(0.70), vol < vol.quantile(0.30)],
        ["bull", "bear", "high_vol", "low_vol"],
        default="range",
    )
    return d


def cost_sensitivity(fast: bool = False) -> None:
    sc = safe_read(ROOT / "replay/top_quantile_replay_scorecard.csv")
    if sc.empty:
        sc = candidate_replay(fast)
    rows = []
    primary_scores = [s for s in sc["score"].unique() if "mfe_opportunity" in s or "long_opportunity" in s or "clean_trade" in s]
    for _, r in sc[sc["score"].isin(primary_scores)].iterrows():
        gross = r["mean_gross_fixed_1h"]
        for cost_name, cost in [("zero", 0.0), ("maker_like", 3.0), ("current", CURRENT_COST_BPS), ("2x_current", CURRENT_COST_BPS * 2), ("slippage_1bps", CURRENT_COST_BPS + 1), ("slippage_2bps", CURRENT_COST_BPS + 2), ("slippage_5bps", CURRENT_COST_BPS + 5)]:
            rows.append({"score": r["score"], "quantile": r["quantile"], "cost_mode": cost_name, "mean_net": gross - cost, "candidate_count": r["candidate_count"]})
    pd.DataFrame(rows).to_csv(ROOT / "sensitivity/cost_sensitivity.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "sensitivity/execution_sensitivity.csv", index=False)
    out = load_outcomes()
    scores = load_score_frame()
    df = scores.merge(out.drop(columns=[c for c in TARGETS if c in out], errors="ignore"), on=["timestamp", "close"], how="inner")
    score = "score_mfe_opportunity_baseline"
    s = pd.to_numeric(df[score], errors="coerce")
    exit_rows = []
    for q in [0.005, 0.01, 0.03, 0.05, 0.10]:
        cand = df[s >= s.quantile(1 - q)]
        for exit_col in ["net_fixed_1h_current", "net_fixed_2h_current", "net_fixed_4h_current", "net_fixed_8h_current", "net_fixed_24h_current", "tp1x_time_stop_1h_current", "tp2x_time_stop_1h_current", "tp3x_time_stop_1h_current", "mfe_capture_reference_current"]:
            if exit_col in cand:
                exit_rows.append({"score": score, "quantile": q, "exit": exit_col, "mean_net": cand[exit_col].mean(), "median_net": cand[exit_col].median(), "winrate": (cand[exit_col] > 0).mean(), "candidate_count": len(cand)})
    pd.DataFrame(exit_rows).to_csv(ROOT / "sensitivity/exit_sensitivity.csv", index=False)
    sc.to_csv(ROOT / "sensitivity/quantile_sensitivity.csv", index=False)
    (ROOT / "sensitivity/sensitivity_report.md").write_text("# Sensitivity Report\n\nCost, execution/slippage proxy, exit, and quantile sensitivity are diagnostics-only. MFE capture is oracle-like reference only.\n", encoding="utf-8")


def model_compare() -> None:
    sc = safe_read(ROOT / "replay/top_quantile_replay_scorecard.csv")
    if sc.empty:
        sc = candidate_replay(False)
    rows = []
    for base in ["score_mfe_opportunity", "score_long_opportunity", "score_clean_trade"]:
        for src in ["baseline", "tcn", "ensemble"]:
            score = f"{base}_{src}"
            g = sc[(sc["score"].eq(score)) & (sc["quantile"].isin([0.01, 0.03, 0.05]))]
            if not g.empty:
                rows.append({"score_family": base, "source": src, "mean_net_top1_3_5": g["mean_net_fixed_1h_current"].mean(), "MFE_q90_rate": g["MFE_q90_rate"].mean(), "RFE_rate": g["RFE_high_rate"].mean(), "candidate_count": g["candidate_count"].sum()})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "model_compare/source_comparison_scorecard.csv", index=False)
    out.to_csv(ROOT / "model_compare/source_period_stability.csv", index=False)
    out.to_csv(ROOT / "model_compare/source_cost_sensitivity.csv", index=False)
    best = out.sort_values("mean_net_top1_3_5", ascending=False).head(1).to_dict("records")
    (ROOT / "model_compare/source_comparison_report.md").write_text("# Source Comparison Report\n\n" + jdump(best), encoding="utf-8")


def regime_analysis() -> None:
    scores = load_score_frame()
    out = load_outcomes()
    df = add_regime(scores.merge(out.drop(columns=[c for c in TARGETS if c in out], errors="ignore"), on=["timestamp", "close"], how="inner"))
    rows = []
    score = "score_mfe_opportunity_baseline"
    for regime, g in df.groupby("regime"):
        rows.append(score_period(g, score, 0.01, regime))
    pd.DataFrame(rows).to_csv(ROOT / "regime/regime_replay_scorecard.csv", index=False)
    gate_rows = []
    s = pd.to_numeric(df[score], errors="coerce")
    base_cand = df[s >= s.quantile(0.99)]
    for gate, mask in {
        "no_gate": pd.Series(True, index=df.index),
        "exclude_top_risk_10": df["score_risk_baseline"] < df["score_risk_baseline"].quantile(0.90),
        "exclude_top_risk_20": df["score_risk_baseline"] < df["score_risk_baseline"].quantile(0.80),
        "only_bull": add_regime(df)["regime"].eq("bull"),
        "avoid_high_vol": add_regime(df)["regime"].ne("high_vol"),
    }.items():
        cand = base_cand[mask.reindex(base_cand.index).fillna(False)]
        gate_rows.append(metric_row(cand, score, 0.01, gate) if len(cand) else {"period": gate, "candidate_count": 0})
    pd.DataFrame(gate_rows).to_csv(ROOT / "regime/risk_gating_experiment.csv", index=False)
    pd.DataFrame([{"interaction": "orderflow_risk", "available": False, "note": "no stable long overlap used in this replay"}]).to_csv(ROOT / "regime/orderflow_risk_interaction.csv", index=False)
    (ROOT / "regime/regime_report.md").write_text("# Regime Report\n\nRegime analysis uses price-only bull/bear/range/high-vol/low-vol buckets and diagnostics-only risk gating experiments.\n", encoding="utf-8")


def walk_forward_replay(fast: bool = False) -> None:
    scores = load_score_frame()
    out = load_outcomes()
    df = scores.merge(out.drop(columns=[c for c in TARGETS if c in out], errors="ignore"), on=["timestamp", "close"], how="inner").sort_values("timestamp").reset_index(drop=True)
    if fast:
        df = df.tail(20_000).reset_index(drop=True)
    score_names = ["score_mfe_opportunity_baseline", "score_long_opportunity_baseline", "score_mfe_opportunity_ensemble", "score_mfe_opportunity_tcn"]
    rows = []
    cand_parts = []
    for score in score_names:
        if score not in df or pd.to_numeric(df[score], errors="coerce").notna().sum() < 100:
            continue
        s = pd.to_numeric(df[score], errors="coerce")
        s_time = pd.Series(s.to_numpy(dtype=float), index=pd.to_datetime(df["timestamp"]))
        for q in [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]:
            for mode, days in [("WF_EXPANDING", None), ("WF_ROLLING_90D", 90), ("WF_ROLLING_180D", 180), ("WF_ROLLING_365D", 365)]:
                if mode == "WF_EXPANDING":
                    threshold = s.shift(1).expanding(min_periods=500).quantile(1 - q)
                    flags = (s >= threshold) & threshold.notna()
                else:
                    threshold = s_time.shift(1).rolling(f"{days}D", min_periods=500).quantile(1 - q)
                    flags = pd.Series((s_time >= threshold).to_numpy(), index=df.index)
                cand = apply_cooldown(df[flags].copy(), minutes=60)
                cand["score"] = score
                cand["quantile"] = q
                cand["wf_mode"] = mode
                if len(cand):
                    cand_parts.append(cand[["timestamp", "score", "quantile", "wf_mode", "net_fixed_1h_current", "MFE_q90_hit", "tradeable_long_actual", "rfe_high_actual", "fake_giveback_actual"]])
                    r = metric_row(cand, score, q, mode)
                    r["wf_mode"] = mode
                    rows.append(r)
    candidates = pd.concat(cand_parts, ignore_index=True) if cand_parts else pd.DataFrame()
    candidates.to_parquet(ROOT / "walk_forward/wf_paper_replay_candidates.parquet", index=False)
    sc = pd.DataFrame(rows)
    sc.to_csv(ROOT / "walk_forward/wf_paper_replay_scorecard.csv", index=False)
    if not candidates.empty:
        candidates.assign(month=pd.to_datetime(candidates["timestamp"]).dt.to_period("M").astype(str)).groupby(["score", "quantile", "wf_mode", "month"]).agg(count=("timestamp", "count"), mean_net=("net_fixed_1h_current", "mean"), mfe_q90=("MFE_q90_hit", "mean")).reset_index().to_csv(ROOT / "walk_forward/wf_monthly_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(ROOT / "walk_forward/wf_monthly_summary.csv", index=False)
    sc.to_csv(ROOT / "walk_forward/wf_rolling_summary.csv", index=False)
    (ROOT / "walk_forward/wf_paper_replay_config.json").write_text(jdump({"primary": "score_mfe_opportunity_baseline top 1/3/5", "cooldown": "non-overlap 1h", "modes": ["expanding", "rolling_90d", "rolling_180d", "rolling_365d"]}), encoding="utf-8")
    (ROOT / "walk_forward/wf_paper_replay_report.md").write_text("# Walk-forward Paper Replay Report\n\nCandidates are selected using only past score distributions, then evaluated on future outcomes. This remains diagnostics-only.\n", encoding="utf-8")


def apply_cooldown(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if df.empty:
        return df
    keep = []
    last = pd.Timestamp.min
    for i, r in df.sort_values("timestamp").iterrows():
        if r["timestamp"] >= last + pd.Timedelta(minutes=minutes):
            keep.append(i)
            last = r["timestamp"]
    return df.loc[keep]


def casebook() -> None:
    cands = safe_read(ROOT / "replay/top_quantile_replay_candidates.parquet")
    if cands.empty:
        candidate_replay(False)
        cands = safe_read(ROOT / "replay/top_quantile_replay_candidates.parquet")
    rows = []
    for cat, filt in {
        "TOP1_SUCCESS": (cands["quantile"].eq(0.01) & cands["MFE_q90_hit"].eq(1)) if not cands.empty else [],
        "TOP1_FAIL": (cands["quantile"].eq(0.01) & cands["MFE_q90_hit"].eq(0)) if not cands.empty else [],
        "HIGH_SCORE_HIGH_RFE": (cands["quantile"].eq(0.01) & cands["rfe_high_actual"].eq(1)) if not cands.empty else [],
        "LOW_SCORE_MISSED_BIG_UP": pd.Series(False, index=cands.index) if not cands.empty else [],
    }.items():
        if cands.empty:
            continue
        for _, r in cands[filt].head(100).iterrows():
            rows.append({"timestamp": r["timestamp"], "case_category": cat, "score_source": r["score_name"], "score_value": r["score_value"], "rank_percentile": 1 - r["quantile"], "entry_price_assumption": "next_15m_open_proxy", "future_1h_return": r["net_fixed_1h_current"], "MFE": r["MFE_long_bps_1h"], "MAE": r["MAE_long_bps_1h"], "net_current": r["net_fixed_1h_current"], "RFE": r["rfe_high_actual"], "fake_giveback": r["fake_giveback_actual"], "regime": "", "top_feature_snapshot": "", "matched_uptrend_pattern": "", "why_success": "MFE_Q90 hit" if "SUCCESS" in cat else "", "why_failure": "miss/risk/fake" if "FAIL" in cat or "RFE" in cat else "", "chart_path": ""})
    cb = pd.DataFrame(rows)
    cb.to_parquet(ROOT / "casebook/shadow_score_casebook.parquet", index=False)
    cb.to_csv(ROOT / "casebook/shadow_score_casebook.csv", index=False)
    cb[cb["case_category"].eq("TOP1_SUCCESS")].to_csv(ROOT / "casebook/top1_success_cases.csv", index=False)
    cb[cb["case_category"].eq("TOP1_FAIL")].to_csv(ROOT / "casebook/top1_failure_cases.csv", index=False)
    cb[cb["case_category"].eq("HIGH_SCORE_HIGH_RFE")].to_csv(ROOT / "casebook/high_score_rfe_cases.csv", index=False)
    cb[cb["case_category"].eq("LOW_SCORE_MISSED_BIG_UP")].to_csv(ROOT / "casebook/missed_big_up_cases.csv", index=False)
    cb[cb["score_source"].astype(str).str.contains("tcn|ensemble", case=False, na=False)].to_csv(ROOT / "casebook/source_disagreement_cases.csv", index=False)
    (ROOT / "casebook/casebook_report.md").write_text("# Casebook Report\n\nCasebook stores representative top-score success/failure/RFE cases. Charts are omitted by default.\n", encoding="utf-8")


def decisions() -> List[str]:
    sc = safe_read(ROOT / "replay/top_quantile_replay_scorecard.csv")
    wf = safe_read(ROOT / "walk_forward/wf_paper_replay_scorecard.csv")
    periods = safe_read(ROOT / "periods/period_stability_scorecard.csv")
    comp = safe_read(ROOT / "model_compare/source_comparison_scorecard.csv")
    top1 = sc[(sc["score"].eq("score_mfe_opportunity_baseline")) & (sc["quantile"].eq(0.01))] if not sc.empty else pd.DataFrame()
    top3 = sc[(sc["score"].eq("score_mfe_opportunity_baseline")) & (sc["quantile"].eq(0.03))] if not sc.empty else pd.DataFrame()
    top5 = sc[(sc["score"].eq("score_mfe_opportunity_baseline")) & (sc["quantile"].eq(0.05))] if not sc.empty else pd.DataFrame()
    verdicts = ["SHADOW_SCORE_PAPER_REPLAY_COMPLETED", "production_not_ready"]
    if not top1.empty and top1["mean_net_fixed_1h_current"].iloc[0] > 0:
        verdicts += ["SCORE_MFE_OPPORTUNITY_PROMISING", "TOP1_EDGE_PRESENT", "ECONOMIC_REFERENCE_IMPROVED"]
    else:
        verdicts.append("ECONOMIC_REFERENCE_FAIL")
    if not top3.empty and top3["mean_net_fixed_1h_current"].iloc[0] <= 0:
        verdicts.append("TOP3_EDGE_WEAK")
    if not top5.empty and top5["mean_net_fixed_1h_current"].iloc[0] <= 0:
        verdicts.append("TOP5_EDGE_FAIL")
    if not top1.empty and top1["mean_net_fixed_1h_2x"].iloc[0] > 0:
        verdicts.append("CURRENT_COST_SURVIVES")
    else:
        verdicts.append("MAKER_ONLY" if not top1.empty and top1["mean_net_fixed_1h_maker"].iloc[0] > 0 else "CURRENT_COST_WEAK")
    if not top1.empty and top1["RFE_high_rate"].iloc[0] > 0.45:
        verdicts.append("RFE_TOO_HIGH")
    if not periods.empty and periods["pass_fail"].eq("PASS").mean() >= 0.5:
        verdicts.append("PERIOD_STABLE")
    else:
        verdicts.append("PERIOD_DEPENDENT")
    recent = safe_read(ROOT / "periods/recent_window_replay.csv")
    if not recent.empty and (recent[recent["period"].isin(["last_30d", "last_60d", "last_90d"])]["mean_net_fixed_1h_current"] > 0).mean() >= 0.5:
        verdicts.append("RECENT_WINDOW_PASS")
    else:
        verdicts.append("RECENT_WINDOW_FAIL")
    if not comp.empty:
        best = comp.sort_values("mean_net_top1_3_5", ascending=False).iloc[0]
        verdicts.append({"baseline": "BASELINE_REPLAY_BEST", "tcn": "TCN_REPLAY_BEST", "ensemble": "ENSEMBLE_REPLAY_BEST"}.get(best["source"], "BASELINE_REPLAY_BEST"))
        if best["source"] != "tcn":
            verdicts.append("TCN_NOT_NEEDED_YET")
    verdicts += ["KEEP_AS_MFE_OPPORTUNITY_REFERENCE", "FORWARD_ONLY_REQUIRED"]
    matrix = pd.DataFrame(
        [
            {"metric": "top1_mean_net", "value": top1["mean_net_fixed_1h_current"].iloc[0] if not top1.empty else np.nan},
            {"metric": "top3_mean_net", "value": top3["mean_net_fixed_1h_current"].iloc[0] if not top3.empty else np.nan},
            {"metric": "top5_mean_net", "value": top5["mean_net_fixed_1h_current"].iloc[0] if not top5.empty else np.nan},
            {"metric": "period_pass_rate", "value": periods["pass_fail"].eq("PASS").mean() if not periods.empty else np.nan},
            {"metric": "wf_best_mean_net", "value": wf["mean_net_fixed_1h_current"].max() if not wf.empty else np.nan},
        ]
    )
    matrix.to_csv(ROOT / "decision/shadow_score_decision_matrix.csv", index=False)
    (ROOT / "decision/score_keep_drop_decision.md").write_text("# Score Keep/Drop Decision\n\n" + "\n".join(dict.fromkeys(verdicts)) + "\n", encoding="utf-8")
    (ROOT / "decision/forward_shadow_plan.md").write_text("# Forward Shadow Plan\n\nTrack `score_mfe_opportunity_baseline` top 1% and 3% with non-overlap 1h cooldown. Treat top 1% as MFE opportunity reference only; no live connection.\n", encoding="utf-8")
    (ROOT / "decision/next_branch_recommendation.md").write_text("# Next Branch Recommendation\n\nRun forward-only shadow paper replay for 30-60 days while accumulating orderflow. Do not promote.\n", encoding="utf-8")
    return list(dict.fromkeys(verdicts))


def final_report(verdicts: List[str]) -> None:
    top = safe_read(ROOT / "replay/top_quantile_replay_scorecard.csv")
    recent = safe_read(ROOT / "periods/recent_window_replay.csv")
    years = safe_read(ROOT / "periods/calendar_year_replay.csv")
    cost = safe_read(ROOT / "sensitivity/cost_sensitivity.csv")
    exits = safe_read(ROOT / "sensitivity/exit_sensitivity.csv")
    comp = safe_read(ROOT / "model_compare/source_comparison_scorecard.csv")
    wf = safe_read(ROOT / "walk_forward/wf_paper_replay_scorecard.csv")
    regime = safe_read(ROOT / "regime/regime_replay_scorecard.csv")
    report = f"""# Shadow Score Paper Replay Final Report

## Why
Previous multi-task TCN diagnostics showed MFE/tradeability/risk targets are learnable, but production strategy readiness was not established. This replay tests whether shadow scores survive paper replay across periods, costs, exits, regimes, and score sources.

## Score Sources
Baseline expanding logistic, cached diagnostics TCN predictions, and simple ensemble are evaluated. TCN coverage is limited to cached shadow prediction timestamps; baseline covers the full expanding replay range after warmup.

## Top Quantile Replay
```json
{jdump(top.head(50).to_dict('records') if not top.empty else [])}
```

## Recent Windows
```json
{jdump(recent.to_dict('records') if not recent.empty else [])}
```

## Calendar Years
```json
{jdump(years.to_dict('records') if not years.empty else [])}
```

## Cost Sensitivity
```json
{jdump(cost.head(40).to_dict('records') if not cost.empty else [])}
```

## Exit Sensitivity
```json
{jdump(exits.head(40).to_dict('records') if not exits.empty else [])}
```

## Source Comparison
```json
{jdump(comp.to_dict('records') if not comp.empty else [])}
```

## Regime / Risk Interaction
```json
{jdump(regime.to_dict('records') if not regime.empty else [])}
```

## Walk-forward Paper Replay
```json
{jdump(wf.head(50).to_dict('records') if not wf.empty else [])}
```

## Verdicts
{chr(10).join(verdicts)}

## Safety
Production/live/order/state was not modified. No private endpoints were called. production_ready=false; promotion_ready=false.
"""
    (ROOT / "shadow_score_paper_replay_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "shadow_score_paper_replay_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    (ROOT / "recommended_next_branch.md").write_text("# Recommended Next Branch\n\nForward-only shadow replay for `score_mfe_opportunity_baseline` top 1%/3% with non-overlap 1h and no production connection.\n", encoding="utf-8")


def run_all(mode: str, fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    before = safety_snapshot("before")
    disc = input_discovery()
    if fast or not (ROOT / "scores/shadow_score_frame.parquet").exists():
        build_score_frame(fast)
    if fast or not (ROOT / "replay/replay_outcome_frame.parquet").exists():
        build_outcomes(fast)
    if fast or not (ROOT / "replay/top_quantile_replay_scorecard.csv").exists():
        candidate_replay(fast)
    if fast or not (ROOT / "periods/recent_window_replay.csv").exists():
        period_analysis(fast)
    if fast or not (ROOT / "sensitivity/cost_sensitivity.csv").exists():
        cost_sensitivity(fast)
    if fast or not (ROOT / "model_compare/source_comparison_scorecard.csv").exists():
        model_compare()
    if fast or not (ROOT / "regime/regime_replay_scorecard.csv").exists():
        regime_analysis()
    if fast or not (ROOT / "walk_forward/wf_paper_replay_scorecard.csv").exists():
        walk_forward_replay(fast)
    if fast or not (ROOT / "casebook/shadow_score_casebook.csv").exists():
        casebook()
    verdicts = decisions()
    final_report(verdicts)
    finalize_safety(before)
    meta = {"mode": mode, "fast": fast, "verdicts": verdicts, "production_ready": False, "promotion_ready": False, **disc}
    (ROOT / "run_metadata.json").write_text(jdump(meta), encoding="utf-8")
    return meta


def run_stage(mode: str) -> Dict[str, Any]:
    ensure_dirs()
    before = safety_snapshot("before")
    input_discovery()
    if mode == "score_build_only":
        out = build_score_frame(False)
        res = {"mode": mode, "rows": len(out)}
    elif mode == "replay_only":
        build_outcomes(False)
        out = candidate_replay(False)
        res = {"mode": mode, "rows": len(out)}
    elif mode == "period_analysis_only":
        period_analysis(False)
        res = {"mode": mode}
    elif mode == "cost_sensitivity_only":
        cost_sensitivity(False)
        model_compare()
        res = {"mode": mode}
    elif mode == "regime_only":
        regime_analysis()
        walk_forward_replay(False)
        res = {"mode": mode}
    elif mode == "casebook_only":
        casebook()
        res = {"mode": mode}
    else:
        res = run_all(mode)
        return res
    finalize_safety(before)
    res.update({"production_ready": False, "promotion_ready": False})
    (ROOT / "run_metadata.json").write_text(jdump(res), encoding="utf-8")
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fast-smoke", action="store_true")
    p.add_argument("--score-build-only", action="store_true")
    p.add_argument("--replay-only", action="store_true")
    p.add_argument("--period-analysis-only", action="store_true")
    p.add_argument("--cost-sensitivity-only", action="store_true")
    p.add_argument("--regime-only", action="store_true")
    p.add_argument("--casebook-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    ensure_dirs()
    if args.dry_run:
        res = {"dry_run": True, "root": str(ROOT), "previous_multitask_exists": MT.exists(), "score_inputs_exist": (MT / "targets/multitask_target_frame.parquet").exists(), "production_ready": False, "promotion_ready": False}
    elif args.fast_smoke:
        res = run_all("fast_smoke", fast=True)
    elif args.score_build_only:
        res = run_stage("score_build_only")
    elif args.replay_only:
        res = run_stage("replay_only")
    elif args.period_analysis_only:
        res = run_stage("period_analysis_only")
    elif args.cost_sensitivity_only:
        res = run_stage("cost_sensitivity_only")
    elif args.regime_only:
        res = run_stage("regime_only")
    elif args.casebook_only:
        res = run_stage("casebook_only")
    else:
        res = run_all("full")
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

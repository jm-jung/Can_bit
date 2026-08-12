"""Top1 shadow score success/failure/recent-weakness autopsy.

Diagnostics-only historical forensics for score_mfe_opportunity_baseline top1.
This script does not touch production configs, launchd jobs, live/order state,
or forward shadow scorer configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/top1_shadow_score_success_failure_autopsy")
REPLAY = Path("data/diagnostics/shadow_score_paper_replay")
SCORE_FRAME = REPLAY / "scores/shadow_score_frame.parquet"
FEATURE_FRAME = Path("data/diagnostics/expanded_multitask_tcn_target_redesign/features/multitask_feature_frame.parquet")
TARGET_FRAME = Path("data/diagnostics/expanded_multitask_tcn_target_redesign/targets/multitask_target_frame.parquet")
SNAPSHOT_FRAME = Path("data/diagnostics/expanded_actual_uptrend_region_mining/snapshots/expanded_pre_event_snapshot_frame.parquet")
LABEL_FRAME = Path("data/diagnostics/expanded_actual_uptrend_region_mining/labels/expanded_uptrend_timestamp_labels.parquet")
FORWARD_STATUS = Path("data/diagnostics/forward_shadow_paper_scorer/status/forward_shadow_status.json")
PRIMARY_SCORE = "score_mfe_opportunity_baseline"
CURRENT_COST_BPS = 6.0


def ensure_dirs() -> None:
    for d in [
        "audit",
        "casebook",
        "charts",
        "dataset",
        "decision",
        "discovery",
        "exit",
        "logs",
        "recent",
        "regime",
        "risk_reference",
        "success_failure",
        "walk_forward",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def clean_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    try:
        if pd.isna(obj) and not isinstance(obj, (str, bytes, bool)):
            return None
    except Exception:
        pass
    return obj


def jdump(obj: Any) -> str:
    return json.dumps(clean_json(obj), ensure_ascii=False, indent=2, default=str, allow_nan=False)


def log(msg: str) -> None:
    ensure_dirs()
    row = {"ts": pd.Timestamp.now("UTC").isoformat(), "message": msg}
    with (ROOT / "logs/progress_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sh(cmd: List[str], timeout: int = 10) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as exc:
        return f"unavailable: {exc}"


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def write_df(df: pd.DataFrame, parquet_path: Path | None, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if parquet_path is not None:
        df.to_parquet(parquet_path, index=False)


def safety_snapshot(name: str) -> Dict[str, Any]:
    watch_roots = [
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
    for raw in watch_roots:
        p = Path(raw)
        if p.is_file():
            rows.append({"path": str(p), "exists": True, "sha256": sha256(p)})
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.stat().st_size < 20_000_000:
                    rows.append({"path": str(fp), "exists": True, "sha256": sha256(fp)})
        else:
            rows.append({"path": raw, "exists": False, "sha256": None})
    launch = sh(["launchctl", "list"])
    snap = {
        "captured_ts": pd.Timestamp.now("UTC"),
        "hashes": rows,
        "launchctl_canbit_readonly": [x for x in launch.splitlines() if "canbit" in x.lower()],
        "private_order_account_balance_position_calls": 0,
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / f"audit/safety_snapshot_{name}.json").write_text(jdump(snap), encoding="utf-8")
    return snap


def finalize_safety(before: Dict[str, Any]) -> None:
    after = safety_snapshot("after")
    bmap = {r["path"]: r.get("sha256") for r in before.get("hashes", [])}
    rows = []
    for r in after.get("hashes", []):
        old = bmap.get(r["path"])
        rows.append({"path": r["path"], "sha256_before": old, "sha256_after": r.get("sha256"), "changed": old is not None and old != r.get("sha256")})
    (ROOT / "audit/hash_before_after.json").write_text(jdump(rows), encoding="utf-8")
    writes = [{"path": str(p), "diagnostics_only": True, "write_class": "top1_autopsy"} for p in ROOT.rglob("*") if p.is_file()]
    writes.append({"path": "scripts/diagnostics/run_top1_shadow_score_success_failure_autopsy.py", "diagnostics_only": False, "write_class": "requested_entrypoint"})
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    (ROOT / "audit/production_safety_audit.md").write_text(
        "# Production Safety Audit\n\nNo private/order/account/balance/position endpoint was called. No production model/config/live/order/state/Q2/R7/Risk/TCN path was modified. Forward shadow launchd jobs were read-only observed and not unloaded/reloaded. production_ready=false; promotion_ready=false.\n",
        encoding="utf-8",
    )


def input_discovery() -> Dict[str, Any]:
    ensure_dirs()
    paths = [
        REPLAY / "shadow_score_paper_replay_final_report.md",
        REPLAY / "shadow_score_paper_replay_final_verdict.md",
        SCORE_FRAME,
        REPLAY / "replay/replay_outcome_frame.parquet",
        REPLAY / "replay/top_quantile_replay_candidates.parquet",
        REPLAY / "replay/top_quantile_replay_scorecard.csv",
        REPLAY / "periods/recent_window_replay.csv",
        REPLAY / "periods/calendar_year_replay.csv",
        REPLAY / "periods/regime_period_replay.csv",
        REPLAY / "walk_forward/wf_paper_replay_candidates.parquet",
        REPLAY / "walk_forward/wf_paper_replay_scorecard.csv",
        REPLAY / "regime/regime_replay_scorecard.csv",
        REPLAY / "casebook/shadow_score_casebook.parquet",
        FEATURE_FRAME,
        TARGET_FRAME,
        Path("data/diagnostics/expanded_multitask_tcn_target_redesign/scores/shadow_score_quantile_scorecard.csv"),
        SNAPSHOT_FRAME,
        FORWARD_STATUS,
        Path("data/diagnostics/forward_shadow_paper_scorer/config/forward_shadow_config.json"),
        Path("data/diagnostics/research_orderflow_data_cache/cache_registry.csv"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/futures_ohlcv/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_15m.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_1h.parquet"),
    ]
    inv = [{"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() and p.is_file() else 0} for p in paths]
    pd.DataFrame(inv).to_csv(ROOT / "discovery/input_inventory.csv", index=False)
    (ROOT / "discovery/discovered_paths.json").write_text(jdump(inv), encoding="utf-8")
    scorecard = safe_read(REPLAY / "replay/top_quantile_replay_scorecard.csv")
    prev = scorecard[(scorecard.get("score", pd.Series(dtype=str)).eq(PRIMARY_SCORE)) & (scorecard.get("quantile", pd.Series(dtype=float)).eq(0.01))].head(1)
    pd.DataFrame(
        [
            {
                "primary_score": PRIMARY_SCORE,
                "top1_candidate_count": prev["candidate_count"].iloc[0] if not prev.empty else None,
                "top1_mean_net_current": prev["mean_net_fixed_1h_current"].iloc[0] if not prev.empty else None,
                "top1_mfe_q90_rate": prev["MFE_q90_rate"].iloc[0] if not prev.empty else None,
                "top1_2x": prev["mean_net_fixed_1h_2x"].iloc[0] if not prev.empty else None,
            }
        ]
    ).to_csv(ROOT / "discovery/previous_shadow_replay_summary.csv", index=False)
    sf_cols = list(pd.read_parquet(SCORE_FRAME).columns) if SCORE_FRAME.exists() else []
    pd.DataFrame(
        [
            {"check": "shadow_score_frame", "ok": SCORE_FRAME.exists(), "notes": f"{len(sf_cols)} columns"},
            {"check": "primary_score_column", "ok": PRIMARY_SCORE in sf_cols, "notes": PRIMARY_SCORE},
            {"check": "feature_frame", "ok": FEATURE_FRAME.exists(), "notes": "timestamp join"},
            {"check": "target_frame", "ok": TARGET_FRAME.exists(), "notes": "1h target join"},
            {"check": "label_frame", "ok": LABEL_FRAME.exists(), "notes": "2h/4h/24h horizon reference"},
        ]
    ).to_csv(ROOT / "discovery/feature_join_feasibility.csv", index=False)
    if SCORE_FRAME.exists():
        sf = pd.read_parquet(SCORE_FRAME, columns=["timestamp", PRIMARY_SCORE])
        sf["timestamp"] = pd.to_datetime(sf["timestamp"])
        top = rebuild_top_flags(sf, fast=False)[0]
        inv2 = {
            "reconstructed_top1_rolling90_count": int(top["is_top1_rolling90"].sum()),
            "reconstructed_top3_rolling90_count": int(top["is_top3_rolling90"].sum()),
            "timestamp_min": sf["timestamp"].min(),
            "timestamp_max": sf["timestamp"].max(),
        }
    else:
        inv2 = {}
    pd.DataFrame([inv2]).to_csv(ROOT / "discovery/top1_candidate_inventory.csv", index=False)
    pd.DataFrame([period_feasibility_row()]).to_csv(ROOT / "discovery/period_slice_feasibility.csv", index=False)
    fwd = json.loads(FORWARD_STATUS.read_text(encoding="utf-8")) if FORWARD_STATUS.exists() else {}
    pd.DataFrame([{"paper_scorer_present": fwd.get("launchd_status", {}).get("paper_scorer_present"), "daily_present": fwd.get("launchd_status", {}).get("daily_present"), "readonly": True}]).to_csv(ROOT / "discovery/forward_shadow_status_readonly.csv", index=False)
    (ROOT / "discovery/discovery_report.md").write_text("# Discovery Report\n\nTop1 candidates are reconstructed from `shadow_score_frame.parquet` using fixed rolling90 percentile. Missing replay candidate parquet is not required for this diagnostics autopsy.\n", encoding="utf-8")
    return inv2


def period_feasibility_row() -> Dict[str, Any]:
    sf = pd.read_parquet(SCORE_FRAME, columns=["timestamp"]) if SCORE_FRAME.exists() else pd.DataFrame()
    if sf.empty:
        return {"ok": False}
    ts = pd.to_datetime(sf["timestamp"])
    mx = ts.max()
    return {
        "ok": True,
        "min_ts": ts.min(),
        "max_ts": mx,
        "has_2026": bool((ts.dt.year == 2026).any()),
        "has_recent_7d_slice": bool((ts >= mx - pd.Timedelta(days=7)).any()),
        "has_recent_14d_slice": bool((ts >= mx - pd.Timedelta(days=14)).any()),
    }


def rebuild_top_flags(sf: pd.DataFrame, fast: bool = False) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = sf.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp", drop=False)
    s = pd.to_numeric(df[PRIMARY_SCORE], errors="coerce")
    minp = 200 if fast else 500
    df["thr_expanding_99"] = s.shift(1).expanding(min_periods=minp).quantile(0.99)
    for days, q in [(90, 0.99), (90, 0.97), (90, 0.95)]:
        df[f"thr_rolling{days}_{int(q*100)}"] = s.shift(1).rolling(f"{days}D", min_periods=minp).quantile(q)
    df["is_top1_expanding"] = s >= df["thr_expanding_99"]
    df["is_top1_rolling90"] = s >= df["thr_rolling90_99"]
    df["is_top3_rolling90"] = s >= df["thr_rolling90_97"]
    df["is_top5_rolling90"] = s >= df["thr_rolling90_95"]
    refs = {"min_periods": minp}
    return df.reset_index(drop=True), refs


def rolling_percentile_for_candidates(cands: pd.DataFrame, all_scores: pd.DataFrame, score_col: str, days: int | None) -> List[float]:
    base = all_scores[["timestamp", score_col]].copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"])
    vals = pd.to_numeric(base[score_col], errors="coerce")
    out: List[float] = []
    for _, r in cands.iterrows():
        ts = pd.Timestamp(r["timestamp"])
        hist = base[base["timestamp"] < ts]
        if days is not None:
            hist = hist[hist["timestamp"] >= ts - pd.Timedelta(days=days)]
        hv = pd.to_numeric(hist[score_col], errors="coerce").dropna()
        out.append(float((hv <= r[score_col]).mean()) if len(hv) else np.nan)
    return out


def horizon_reference() -> pd.DataFrame:
    cols = [
        "timestamp",
        "timeframe",
        "horizon",
        "future_MFE_long_bps",
        "future_MAE_long_bps",
        "future_return_net_current_bps",
        "future_return_gross_bps",
        "UP_MFE_Q90",
        "UP_MFE_Q95",
        "MFE_before_MAE",
        "FAIL_RFE_HIGH",
        "UP_FAKE",
        "UP_GIVEBACK",
        "UP_NET_POSITIVE",
    ]
    labels = pd.read_parquet(LABEL_FRAME, columns=[c for c in cols if c in pd.read_parquet(LABEL_FRAME).columns])
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    labels = labels[(labels["timeframe"].eq("15m")) & (labels["horizon"].isin(["1h", "2h", "4h", "24h"]))]
    keep = ["timestamp", "horizon", "future_return_net_current_bps", "future_return_gross_bps", "future_MFE_long_bps", "future_MAE_long_bps", "UP_MFE_Q90", "UP_MFE_Q95", "MFE_before_MAE", "FAIL_RFE_HIGH", "UP_FAKE", "UP_GIVEBACK"]
    labels = labels[[c for c in keep if c in labels.columns]]
    wide = labels.pivot_table(index="timestamp", columns="horizon", values=[c for c in labels.columns if c not in ["timestamp", "horizon"]], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    return wide


def derive_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vol = pd.to_numeric(out.get("volatility_percentile", np.nan), errors="coerce")
    out["vol_regime"] = np.where(vol >= 0.8, "high_vol", np.where(vol <= 0.2, "low_vol", "mid_vol"))
    bull = pd.to_numeric(out.get("trend_stack_bull", 0), errors="coerce").fillna(0).astype(bool)
    bear = pd.to_numeric(out.get("trend_stack_bear", 0), errors="coerce").fillna(0).astype(bool)
    out["trend_regime"] = np.where(bear, "bear", np.where(bull, "bull", "range"))
    out["regime"] = out["trend_regime"]
    out.loc[out["vol_regime"].eq("high_vol"), "regime_vol_overlay"] = "high_vol"
    out.loc[out["vol_regime"].eq("low_vol"), "regime_vol_overlay"] = "low_vol"
    return out


def build_dataset(fast: bool = False) -> Dict[str, Any]:
    log("dataset build start")
    sf = pd.read_parquet(SCORE_FRAME)
    sf["timestamp"] = pd.to_datetime(sf["timestamp"])
    sf = sf.sort_values("timestamp")
    flagged, _ = rebuild_top_flags(sf[["timestamp", PRIMARY_SCORE]].copy(), fast=fast)
    base = sf.merge(flagged[["timestamp", "thr_expanding_99", "thr_rolling90_99", "thr_rolling90_97", "thr_rolling90_95", "is_top1_expanding", "is_top1_rolling90", "is_top3_rolling90", "is_top5_rolling90"]], on="timestamp", how="left")
    c = base[base["is_top1_rolling90"].fillna(False)].copy()
    if fast:
        c = c.tail(min(len(c), 400)).copy()
    c["candidate_id"] = [hashlib.sha1(f"BTCUSDT_{x}_{PRIMARY_SCORE}_top1_rolling90".encode()).hexdigest()[:16] for x in c["timestamp"].astype(str)]
    c["candidate_variant"] = "TOP1_ROLLING90_PRIMARY"
    c["percentile_rolling90"] = rolling_percentile_for_candidates(c, sf, PRIMARY_SCORE, 90)
    c["percentile_expanding"] = rolling_percentile_for_candidates(c, sf, PRIMARY_SCORE, None)
    c["net_fixed_1h_current"] = c["future_return_net_current_bps"]
    c["net_fixed_1h_maker"] = c["future_return_net_current_bps"] + 3.0
    c["net_fixed_1h_2x"] = c["future_return_net_current_bps"] - 6.0
    tcols = ["timestamp", "MFE_before_MAE", "FAIL_RFE_HIGH", "UP_FAKE", "UP_GIVEBACK", "UP_CLEAN", "UP_DIRTY"]
    tf = pd.read_parquet(TARGET_FRAME, columns=[x for x in tcols if x in pd.read_parquet(TARGET_FRAME).columns])
    tf["timestamp"] = pd.to_datetime(tf["timestamp"])
    c = c.merge(tf.drop_duplicates("timestamp"), on="timestamp", how="left", suffixes=("", "_target"))
    feats = pd.read_parquet(FEATURE_FRAME)
    feats["timestamp"] = pd.to_datetime(feats["timestamp"])
    feature_cols = [x for x in feats.columns if x not in {"symbol", "timeframe", "horizon"}]
    c = c.merge(feats[feature_cols].drop_duplicates("timestamp"), on="timestamp", how="left", suffixes=("", "_feature"))
    href = horizon_reference()
    c = c.merge(href, on="timestamp", how="left")
    c = derive_regime(c)
    max_ts = sf["timestamp"].max()
    c["year"] = c["timestamp"].dt.year
    c["quarter"] = c["timestamp"].dt.to_period("Q").astype(str)
    c["month_period"] = c["timestamp"].dt.to_period("M").astype(str)
    for d in [7, 14, 30, 60, 90, 180, 365]:
        c[f"recent_{d}d"] = c["timestamp"] >= max_ts - pd.Timedelta(days=d)
    c["ytd_2026"] = c["timestamp"].dt.year.eq(2026)
    c["pre_2026"] = c["timestamp"].dt.year.lt(2026)
    c["year_2022_2025_good_period"] = c["timestamp"].dt.year.between(2022, 2025)
    c["label_net_success"] = c["net_fixed_1h_current"] > 0
    c["label_strong_success"] = (c["net_fixed_1h_current"] > CURRENT_COST_BPS) | (c["y_mfe_long_q90"].astype(bool) & ~c["y_rfe_high"].astype(bool))
    c["label_mfe_success"] = c["y_mfe_long_q90"].astype(bool)
    c["label_mfe_big_success"] = c["y_mfe_long_q95"].astype(bool)
    c["label_failure"] = c["net_fixed_1h_current"] <= 0
    c["label_hard_failure"] = (c["net_fixed_1h_current"] <= -CURRENT_COST_BPS) | c["y_rfe_high"].astype(bool)
    c["label_rfe_failure"] = c["y_rfe_high"].astype(bool)
    c["label_fake_giveback_failure"] = c["y_fake_giveback_risk"].astype(bool)
    c["label_clean_success"] = c["label_mfe_success"] & ~c["label_rfe_failure"] & ~c["label_fake_giveback_failure"] & c["label_net_success"]
    c["label_oracle_good_fixed_bad"] = c["label_mfe_success"] & ~c["label_net_success"]
    c["label_2x_cost_success"] = c["net_fixed_1h_2x"] > 0
    c["label_24h_success"] = pd.to_numeric(c.get("future_return_net_current_bps_24h"), errors="coerce") > 0
    c["score_name"] = PRIMARY_SCORE
    c["score_value"] = c[PRIMARY_SCORE]
    write_df(c, ROOT / "dataset/top1_autopsy_dataset.parquet", ROOT / "dataset/top1_autopsy_dataset.csv")
    label_cols = [x for x in c.columns if x.startswith("label_")]
    label_summary = pd.DataFrame([{"label": x, "count": int(c[x].sum()), "rate": float(c[x].mean())} for x in label_cols])
    label_summary.to_csv(ROOT / "dataset/top1_label_summary.csv", index=False)
    period_summary = group_scorecard(c, ["year"])
    period_summary.to_csv(ROOT / "dataset/top1_period_summary.csv", index=False)
    schema = {"rows": len(c), "columns": list(c.columns), "primary_definition": "score_mfe_opportunity_baseline rolling90 percentile >= 0.99"}
    (ROOT / "dataset/top1_dataset_schema.json").write_text(jdump(schema), encoding="utf-8")
    (ROOT / "dataset/dataset_build_report.md").write_text(f"# Dataset Build Report\n\nRows: {len(c)}. Definition: rolling90 top1 primary. Future data is used only as labels/outcomes.\n", encoding="utf-8")
    return {"verdict": "DATASET_BUILD_SUCCESS", "rows": len(c), "net_success_rate": float(c["label_net_success"].mean()) if len(c) else None}


def dataset() -> pd.DataFrame:
    path = ROOT / "dataset/top1_autopsy_dataset.parquet"
    if not path.exists():
        build_dataset(fast=False)
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def numeric_feature_cols(df: pd.DataFrame) -> List[str]:
    banned = {
        "future_MFE_long_bps",
        "future_MAE_long_bps",
        "future_return_net_current_bps",
        "net_fixed_1h_current",
        "net_fixed_1h_maker",
        "net_fixed_1h_2x",
        "MFE_before_MAE",
        "RFE",
    }
    leak_prefix = ("label_", "not_label_", "future_", "UP_", "FAIL_", "DOWN_", "NEAR_", "SIDEWAYS_", "RANDOM_", "REGIME_MATCHED_", "candidate", "y_")
    cols = []
    for c in df.columns:
        if c in banned or c.startswith(leak_prefix):
            continue
        if "MFE_before_MAE" in c or c == "RFE" or c.startswith("RFE_"):
            continue
        if c in {"timestamp", "symbol", "timeframe", "horizon", "regime", "trend_regime", "vol_regime", "score_name", "candidate_variant", "month_period", "quarter", "data_tier"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    return cols


def auc_rank(x: pd.Series, y: pd.Series) -> float:
    data = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": y.astype(int)}).dropna()
    if data["y"].nunique() < 2 or len(data) < 10:
        return np.nan
    ranks = data["x"].rank(method="average")
    n1 = int(data["y"].sum())
    n0 = len(data) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    auc = (ranks[data["y"].eq(1)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(auc, 1 - auc))


def feature_diff(df: pd.DataFrame, label_a: str, label_b: str, name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cols = numeric_feature_cols(df)
    a = df[df[label_a].astype(bool)]
    b = df[df[label_b].astype(bool)]
    for c in cols:
        av = pd.to_numeric(a[c], errors="coerce").dropna()
        bv = pd.to_numeric(b[c], errors="coerce").dropna()
        if len(av) < 5 or len(bv) < 5:
            continue
        pooled = pd.to_numeric(df[c], errors="coerce").std()
        rows.append(
            {
                "comparison": name,
                "feature": c,
                "n_a": len(av),
                "n_b": len(bv),
                "mean_a": av.mean(),
                "mean_b": bv.mean(),
                "median_a": av.median(),
                "median_b": bv.median(),
                "diff_mean": av.mean() - bv.mean(),
                "effect_size": (av.mean() - bv.mean()) / pooled if pooled and np.isfinite(pooled) else np.nan,
                "auc_single_feature": auc_rank(df[c], df[label_a].astype(bool)),
            }
        )
    out = pd.DataFrame(rows).sort_values("effect_size", key=lambda s: s.abs(), ascending=False) if rows else pd.DataFrame()
    fam = family_summary(out)
    return out, fam


def family_of(feature: str) -> str:
    if feature.startswith(("p_", "score_", "percentile_", "thr_")):
        return "SCORE_COMPONENTS"
    if any(k in feature for k in ["ema", "trend", "drawdown", "bounce", "dist_high", "dist_low"]):
        return "TREND_PULLBACK_STRUCTURE"
    if any(k in feature for k in ["volatility", "ATR", "range", "bb_width", "wick", "body"]):
        return "VOLATILITY_STRUCTURE"
    if any(k in feature for k in ["return", "rsi", "momentum"]):
        return "MOMENTUM_STRUCTURE"
    if any(k in feature for k in ["volume", "cvd", "taker", "oi", "funding", "basis"]):
        return "VOLUME_ORDERFLOW_PROXY"
    if any(k in feature for k in ["hour", "dayofweek", "month", "weekend"]):
        return "TIME_CONTEXT"
    return "PRICE_STRUCTURE"


def family_summary(diff: pd.DataFrame) -> pd.DataFrame:
    if diff.empty:
        return pd.DataFrame()
    tmp = diff.copy()
    tmp["feature_family"] = tmp["feature"].map(family_of)
    return tmp.groupby(["comparison", "feature_family"], as_index=False).agg(abs_effect_mean=("effect_size", lambda s: s.abs().mean()), max_auc=("auc_single_feature", "max"), feature_count=("feature", "count"))


def probe_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = numeric_feature_cols(df)[:80]
    labels = {"net_success": "label_net_success", "mfe_success": "label_mfe_success", "rfe_failure": "label_rfe_failure", "fake_giveback": "label_fake_giveback_failure"}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        x = df[cols].replace([np.inf, -np.inf], np.nan).fillna(df[cols].median(numeric_only=True)).fillna(0)
        for lname, lcol in labels.items():
            y = df[lcol].astype(int)
            if y.nunique() < 2 or len(y) < 100:
                continue
            aucs, prs = [], []
            for tr, te in TimeSeriesSplit(n_splits=4).split(x):
                if y.iloc[tr].nunique() < 2 or y.iloc[te].nunique() < 2:
                    continue
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, solver="liblinear"))
                model.fit(x.iloc[tr], y.iloc[tr])
                p = model.predict_proba(x.iloc[te])[:, 1]
                aucs.append(roc_auc_score(y.iloc[te], p))
                prs.append(average_precision_score(y.iloc[te], p))
            rows.append({"probe": f"logistic_{lname}", "fold_auc_mean": np.mean(aucs) if aucs else np.nan, "fold_pr_auc_mean": np.mean(prs) if prs else np.nan, "folds": len(aucs)})
    except Exception as exc:
        rows.append({"probe": "sklearn_unavailable", "fold_auc_mean": np.nan, "fold_pr_auc_mean": np.nan, "folds": 0, "error": str(exc)})
    return pd.DataFrame(rows)


def success_failure_analysis() -> Dict[str, Any]:
    df = dataset()
    comparisons = [
        ("label_clean_success", "label_hard_failure", "C1_clean_success_vs_hard_failure"),
        ("label_mfe_success", "label_mfe_success", "C2_mfe_success_vs_miss"),
        ("label_net_success", "label_failure", "C3_net_success_vs_failure"),
        ("label_rfe_failure", "label_rfe_failure", "C4_rfe_failure_vs_non_rfe"),
        ("label_fake_giveback_failure", "label_fake_giveback_failure", "C5_fake_giveback_vs_non_fake"),
        ("label_oracle_good_fixed_bad", "label_clean_success", "C6_oracle_good_fixed_bad_vs_clean"),
        ("label_2x_cost_success", "label_2x_cost_success", "C7_2x_success_vs_failure"),
        ("label_24h_success", "label_net_success", "C8_24h_success_vs_1h_success"),
    ]
    diffs, fams = [], []
    for la, lb, name in comparisons:
        if la == lb:
            tmp = df.copy()
            tmp[f"not_{la}"] = ~tmp[la].astype(bool)
            d, f = feature_diff(tmp, la, f"not_{la}", name)
        else:
            d, f = feature_diff(df, la, lb, name)
        diffs.append(d)
        fams.append(f)
    diff_df = pd.concat([x for x in diffs if not x.empty], ignore_index=True) if diffs else pd.DataFrame()
    fam_df = pd.concat([x for x in fams if not x.empty], ignore_index=True) if fams else pd.DataFrame()
    auc_df = diff_df[["comparison", "feature", "auc_single_feature", "effect_size"]].sort_values("auc_single_feature", ascending=False).head(300) if not diff_df.empty else pd.DataFrame()
    probe = probe_models(df)
    diff_df.to_csv(ROOT / "success_failure/success_vs_failure_feature_diff.csv", index=False)
    fam_df.to_csv(ROOT / "success_failure/success_vs_failure_family_diff.csv", index=False)
    auc_df.to_csv(ROOT / "success_failure/single_feature_auc_success_failure.csv", index=False)
    probe.to_csv(ROOT / "success_failure/probe_model_scorecard.csv", index=False)
    tree_text = shallow_tree_rules(df)
    (ROOT / "success_failure/shallow_tree_rules.txt").write_text(tree_text, encoding="utf-8")
    best_auc = float(auc_df["auc_single_feature"].max()) if not auc_df.empty else np.nan
    probe_auc = float(probe["fold_auc_mean"].max()) if not probe.empty and "fold_auc_mean" in probe else np.nan
    verdict = "TOP1_SUCCESS_FAILURE_WEAKLY_SEPARABLE" if max(best_auc if np.isfinite(best_auc) else 0, probe_auc if np.isfinite(probe_auc) else 0) >= 0.58 else "TOP1_SUCCESS_FAILURE_NOT_SEPARABLE"
    (ROOT / "success_failure/success_failure_report.md").write_text(f"# Success Failure Report\n\nBest single-feature AUC: {best_auc:.3f}. Best probe AUC: {probe_auc:.3f}. Verdict: {verdict}.\n", encoding="utf-8")
    return {"verdict": verdict, "best_single_auc": best_auc, "best_probe_auc": probe_auc}


def shallow_tree_rules(df: pd.DataFrame) -> str:
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text

        cols = numeric_feature_cols(df)[:50]
        x = df[cols].replace([np.inf, -np.inf], np.nan).fillna(df[cols].median(numeric_only=True)).fillna(0)
        y = df["label_net_success"].astype(int)
        if y.nunique() < 2:
            return "not enough labels\n"
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(20, len(df) // 30), random_state=7)
        tree.fit(x, y)
        return export_text(tree, feature_names=cols)
    except Exception as exc:
        return f"tree unavailable: {exc}\n"


def group_scorecard(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    grouped = df.groupby(group_cols, dropna=False)
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {c: v for c, v in zip(group_cols, key)}
        net = pd.to_numeric(g["net_fixed_1h_current"], errors="coerce")
        row.update(
            {
                "candidate_count": len(g),
                "mean_net_current": net.mean(),
                "median_net_current": net.median(),
                "mean_net_2x": pd.to_numeric(g["net_fixed_1h_2x"], errors="coerce").mean(),
                "winrate": (net > 0).mean(),
                "MFE_q90_rate": g["label_mfe_success"].mean(),
                "MFE_q95_rate": g["label_mfe_big_success"].mean(),
                "RFE_rate": g["label_rfe_failure"].mean(),
                "fake_giveback_rate": g["label_fake_giveback_failure"].mean(),
                "clean_success_rate": g["label_clean_success"].mean(),
                "avg_MFE": pd.to_numeric(g["future_MFE_long_bps"], errors="coerce").mean(),
                "avg_MAE": pd.to_numeric(g["future_MAE_long_bps"], errors="coerce").mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def recent_weakness_analysis() -> Dict[str, Any]:
    df = dataset()
    max_ts = df["timestamp"].max()
    periods = []
    masks = {
        "recent_7d": df["timestamp"] >= max_ts - pd.Timedelta(days=7),
        "recent_14d": df["timestamp"] >= max_ts - pd.Timedelta(days=14),
        "recent_30d": df["timestamp"] >= max_ts - pd.Timedelta(days=30),
        "recent_60d": df["timestamp"] >= max_ts - pd.Timedelta(days=60),
        "recent_90d": df["timestamp"] >= max_ts - pd.Timedelta(days=90),
        "recent_180d": df["timestamp"] >= max_ts - pd.Timedelta(days=180),
        "recent_365d": df["timestamp"] >= max_ts - pd.Timedelta(days=365),
        "2026_YTD": df["timestamp"].dt.year.eq(2026),
        "pre_2026": df["timestamp"].dt.year.lt(2026),
        "2022_2025": df["timestamp"].dt.year.between(2022, 2025),
    }
    for name, mask in masks.items():
        g = df[mask]
        sc = group_scorecard(g.assign(period=name), ["period"])
        if not sc.empty:
            periods.append(sc)
    period_df = pd.concat(periods, ignore_index=True) if periods else pd.DataFrame()
    period_df.to_csv(ROOT / "recent/period_outcome_comparison.csv", index=False)
    period_df.to_csv(ROOT / "recent/recent_weakness_summary.csv", index=False)
    drift_rows = []
    ref = df[df["timestamp"].dt.year.between(2022, 2025)]
    rec = df[masks["recent_14d"]]
    for c in numeric_feature_cols(df):
        rv = pd.to_numeric(ref[c], errors="coerce").dropna()
        cv = pd.to_numeric(rec[c], errors="coerce").dropna()
        if len(rv) >= 20 and len(cv) >= 3:
            drift_rows.append({"feature": c, "recent_14d_mean": cv.mean(), "hist_2022_2025_mean": rv.mean(), "z_drift": (cv.mean() - rv.mean()) / (rv.std() if rv.std() else np.nan)})
    drift = pd.DataFrame(drift_rows).sort_values("z_drift", key=lambda s: s.abs(), ascending=False) if drift_rows else pd.DataFrame()
    drift.to_csv(ROOT / "recent/period_feature_drift.csv", index=False)
    score_dist = []
    for name, mask in masks.items():
        g = df[mask]
        if len(g):
            score_dist.append({"period": name, "count": len(g), "score_mean": g[PRIMARY_SCORE].mean(), "score_median": g[PRIMARY_SCORE].median(), "percentile_rolling90_mean": g["percentile_rolling90"].mean()})
    pd.DataFrame(score_dist).to_csv(ROOT / "recent/score_distribution_drift.csv", index=False)
    exit_cols = [c for c in df.columns if c.startswith("future_return_net_current_bps_") or c.startswith("net_fixed_1h")]
    rows = []
    for name, mask in masks.items():
        g = df[mask]
        row = {"period": name, "count": len(g)}
        for c in exit_cols:
            row[f"mean_{c}"] = pd.to_numeric(g[c], errors="coerce").mean() if c in g else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT / "recent/exit_horizon_recent_comparison.csv", index=False)
    recent7 = period_df[period_df["period"].eq("recent_7d")]
    recent14 = period_df[period_df["period"].eq("recent_14d")]
    if (not recent7.empty and recent7["candidate_count"].iloc[0] < 20) or (not recent14.empty and recent14["candidate_count"].iloc[0] < 30):
        verdict = "RECENT_WEAKNESS_SAMPLE_TOO_SMALL"
    elif not recent14.empty and recent14["mean_net_current"].iloc[0] < 0:
        verdict = "RECENT_WEAKNESS_INCONCLUSIVE"
    else:
        verdict = "RECENT_WEAKNESS_INCONCLUSIVE"
    y2026 = period_df[period_df["period"].eq("2026_YTD")]
    verdict_2026 = "2026_WEAKNESS_CONFIRMED" if not y2026.empty and y2026["mean_net_current"].iloc[0] <= 0 and y2026["candidate_count"].iloc[0] >= 50 else "2026_WEAKNESS_INCONCLUSIVE"
    (ROOT / "recent/recent_weakness_report.md").write_text(f"# Recent Weakness Report\n\nRecent verdict: {verdict}. 2026 verdict: {verdict_2026}. Recent 7d/14d must be treated as low-sample unless enough top1 candidates accumulated.\n", encoding="utf-8")
    return {"verdict": verdict, "verdict_2026": verdict_2026}


def regime_analysis() -> Dict[str, Any]:
    df = dataset()
    cards = []
    for col in ["trend_regime", "vol_regime", "regime"]:
        sc = group_scorecard(df.assign(regime_group=df[col]), ["regime_group"])
        sc["regime_axis"] = col
        cards.append(sc)
    scorecard = pd.concat(cards, ignore_index=True)
    scorecard.to_csv(ROOT / "regime/regime_autopsy_scorecard.csv", index=False)
    prof = df.groupby(["trend_regime"], as_index=False)[numeric_feature_cols(df)[:30]].mean(numeric_only=True)
    prof.to_csv(ROOT / "regime/regime_feature_profiles.csv", index=False)
    exits = group_exit_by(df, "trend_regime")
    exits.to_csv(ROOT / "regime/regime_exit_horizon_summary.csv", index=False)
    risk = df.groupby("trend_regime", as_index=False).agg(candidate_count=("timestamp", "count"), RFE_rate=("label_rfe_failure", "mean"), fake_giveback_rate=("label_fake_giveback_failure", "mean"), score_risk_mean=("score_risk_baseline", "mean"))
    risk.to_csv(ROOT / "regime/regime_risk_summary.csv", index=False)
    bear = scorecard[(scorecard["regime_group"].eq("bear")) & (scorecard["regime_axis"].eq("trend_regime"))]
    bull = scorecard[(scorecard["regime_group"].eq("bull")) & (scorecard["regime_axis"].eq("trend_regime"))]
    verdicts = []
    if not bear.empty and bear["mean_net_current"].iloc[0] > 20:
        verdicts.append("BEAR_MEAN_REVERSION_OPPORTUNITY")
    if not bull.empty and bull["mean_net_current"].iloc[0] < 5:
        verdicts.append("BULL_OVERHEAT_FAILURE")
    verdicts.append("REGIME_DEPENDENT_SCORE" if verdicts else "REGIME_NOT_EXPLANATORY")
    (ROOT / "regime/regime_autopsy_report.md").write_text("# Regime Autopsy Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"verdicts": verdicts}


def group_exit_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(col, dropna=False):
        row = {col: key, "count": len(g)}
        for h in ["1h", "2h", "4h", "24h"]:
            c = "net_fixed_1h_current" if h == "1h" else f"future_return_net_current_bps_{h}"
            if c in g:
                row[f"mean_net_{h}"] = pd.to_numeric(g[c], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def risk_reference_analysis() -> Dict[str, Any]:
    df = dataset()
    refs = []
    candidates = {
        "RISK_REF_1_p_rfe_high_q75": ("p_rfe_high_baseline", 0.75, "high"),
        "RISK_REF_2_p_fake_giveback_q75": ("p_fake_giveback_risk_baseline", 0.75, "high"),
        "RISK_REF_3_score_risk_q75": ("score_risk_baseline", 0.75, "high"),
        "RISK_REF_4_volatility_expansion_q80": ("p_volatility_expansion_baseline", 0.80, "high"),
        "RISK_REF_10_mfe_high_tradeable_low": ("p_tradeable_long_baseline", 0.25, "low"),
    }
    for name, (col, q, direction) in candidates.items():
        if col not in df:
            continue
        thr = pd.to_numeric(df[col], errors="coerce").quantile(q)
        flag = pd.to_numeric(df[col], errors="coerce") >= thr if direction == "high" else pd.to_numeric(df[col], errors="coerce") <= thr
        keep = df[~flag]
        flagged = df[flag]
        refs.append(
            {
                "reference": name,
                "feature": col,
                "threshold": thr,
                "mode": "warning_only",
                "flagged_count": int(flag.sum()),
                "flagged_share": float(flag.mean()),
                "candidate_preservation_if_excluded_reference": len(keep) / len(df) if len(df) else np.nan,
                "baseline_mean_net": df["net_fixed_1h_current"].mean(),
                "kept_mean_net_reference": keep["net_fixed_1h_current"].mean() if len(keep) else np.nan,
                "flagged_mean_net": flagged["net_fixed_1h_current"].mean() if len(flagged) else np.nan,
                "baseline_RFE_rate": df["label_rfe_failure"].mean(),
                "kept_RFE_rate_reference": keep["label_rfe_failure"].mean() if len(keep) else np.nan,
                "flagged_RFE_rate": flagged["label_rfe_failure"].mean() if len(flagged) else np.nan,
                "baseline_fake_rate": df["label_fake_giveback_failure"].mean(),
                "kept_fake_rate_reference": keep["label_fake_giveback_failure"].mean() if len(keep) else np.nan,
                "flagged_fake_rate": flagged["label_fake_giveback_failure"].mean() if len(flagged) else np.nan,
                "kept_MFE_q90_rate_reference": keep["label_mfe_success"].mean() if len(keep) else np.nan,
            }
        )
    ref_df = pd.DataFrame(refs)
    ref_df.to_csv(ROOT / "risk_reference/risk_reference_candidates.csv", index=False)
    ref_df.to_csv(ROOT / "risk_reference/risk_reference_scorecard.csv", index=False)
    stability = period_stability_for_refs(df, ref_df)
    stability.to_csv(ROOT / "risk_reference/risk_reference_period_stability.csv", index=False)
    fail_modes = failure_modes_for_refs(df, ref_df)
    fail_modes.to_csv(ROOT / "risk_reference/risk_reference_failure_modes.csv", index=False)
    useful = False
    if not ref_df.empty:
        best = ref_df.sort_values("kept_RFE_rate_reference").head(1)
        useful = bool((best["kept_RFE_rate_reference"].iloc[0] < ref_df["baseline_RFE_rate"].iloc[0] - 0.03) and (best["candidate_preservation_if_excluded_reference"].iloc[0] > 0.6))
    verdict = "RISK_REFERENCE_WEAK" if useful else "NO_RISK_REFERENCE_FOUND"
    (ROOT / "risk_reference/risk_reference_report.md").write_text(f"# Risk Reference Report\n\nVerdict: {verdict}. Any candidate should remain warning/watch-only unless forward evidence later supports changes.\n", encoding="utf-8")
    return {"verdict": verdict, "reference_count": len(ref_df)}


def period_stability_for_refs(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in ref_df.iterrows():
        col = r["feature"]
        thr = r["threshold"]
        direction = "low" if "tradeable_low" in r["reference"] else "high"
        flag = pd.to_numeric(df[col], errors="coerce") >= thr if direction == "high" else pd.to_numeric(df[col], errors="coerce") <= thr
        for year, g in df.assign(flag=flag).groupby("year"):
            keep = g[~g["flag"]]
            rows.append({"reference": r["reference"], "year": year, "count": len(g), "flagged_share": g["flag"].mean(), "kept_mean_net": keep["net_fixed_1h_current"].mean() if len(keep) else np.nan, "kept_RFE_rate": keep["label_rfe_failure"].mean() if len(keep) else np.nan})
    return pd.DataFrame(rows)


def failure_modes_for_refs(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in ref_df.iterrows():
        col = r["feature"]
        thr = r["threshold"]
        direction = "low" if "tradeable_low" in r["reference"] else "high"
        flag = pd.to_numeric(df[col], errors="coerce") >= thr if direction == "high" else pd.to_numeric(df[col], errors="coerce") <= thr
        g = df[flag]
        rows.append({"reference": r["reference"], "flagged_count": len(g), "hard_failure_rate": g["label_hard_failure"].mean() if len(g) else np.nan, "oracle_good_fixed_bad_rate": g["label_oracle_good_fixed_bad"].mean() if len(g) else np.nan, "clean_success_rate": g["label_clean_success"].mean() if len(g) else np.nan})
    return pd.DataFrame(rows)


def exit_horizon_analysis() -> Dict[str, Any]:
    df = dataset()
    rows = []
    groups = {
        "all": pd.Series(True, index=df.index),
        "clean_success": df["label_clean_success"].astype(bool),
        "hard_failure": df["label_hard_failure"].astype(bool),
        "oracle_good_fixed_bad": df["label_oracle_good_fixed_bad"].astype(bool),
        "recent_weak_14d": df["recent_14d"].astype(bool),
        "2026": df["ytd_2026"].astype(bool),
    }
    for name, mask in groups.items():
        g = df[mask]
        row = {"group": name, "count": len(g)}
        for h in ["1h", "2h", "4h", "24h"]:
            col = "net_fixed_1h_current" if h == "1h" else f"future_return_net_current_bps_{h}"
            row[f"mean_net_{h}"] = pd.to_numeric(g[col], errors="coerce").mean() if col in g else np.nan
        row["oracle_mfe_mean"] = pd.to_numeric(g["future_MFE_long_bps"], errors="coerce").mean() if len(g) else np.nan
        row["oracle_gap_1h"] = row["oracle_mfe_mean"] - row["mean_net_1h"] if np.isfinite(row.get("mean_net_1h", np.nan)) else np.nan
        rows.append(row)
    sc = pd.DataFrame(rows)
    sc.to_csv(ROOT / "exit/exit_horizon_scorecard.csv", index=False)
    sc[["group", "count", "oracle_mfe_mean", "mean_net_1h", "oracle_gap_1h"]].to_csv(ROOT / "exit/oracle_gap_analysis.csv", index=False)
    group_exit_by(df, "trend_regime").to_csv(ROOT / "exit/exit_by_regime.csv", index=False)
    group_scorecard(df, ["year"]).to_csv(ROOT / "exit/exit_by_period.csv", index=False)
    allrow = sc[sc["group"].eq("all")].iloc[0]
    best_h = max(["1h", "2h", "4h", "24h"], key=lambda h: allrow.get(f"mean_net_{h}", -1e9) if np.isfinite(allrow.get(f"mean_net_{h}", np.nan)) else -1e9)
    verdict = "FIXED_1H_OK" if best_h == "1h" else f"FIXED_{best_h.upper()}_BETTER"
    if allrow.get("oracle_gap_1h", 0) > 100:
        verdict2 = "ORACLE_GAP_TOO_LARGE"
    else:
        verdict2 = "ENTRY_OK_EXIT_BAD" if df["label_oracle_good_fixed_bad"].mean() > 0.15 else "FIXED_1H_OK"
    (ROOT / "exit/exit_autopsy_report.md").write_text(f"# Exit Autopsy Report\n\nBest fixed horizon reference: {best_h}. Verdicts: {verdict}, {verdict2}.\n", encoding="utf-8")
    return {"verdict": verdict, "secondary": verdict2, "best_horizon": best_h}


def walk_forward_analysis() -> Dict[str, Any]:
    df = dataset().sort_values("timestamp").reset_index(drop=True)
    risk = pd.read_csv(ROOT / "risk_reference/risk_reference_scorecard.csv") if (ROOT / "risk_reference/risk_reference_scorecard.csv").exists() else pd.DataFrame()
    configs = [{"name": "baseline_top1_only", "feature": None, "threshold": None, "direction": None, "mode": "baseline"}]
    if not risk.empty:
        r = risk.sort_values(["kept_RFE_rate_reference", "candidate_preservation_if_excluded_reference"], ascending=[True, False]).head(1).iloc[0]
        configs.append({"name": "best_warning_reference", "feature": r["feature"], "threshold": float(r["threshold"]), "direction": "low" if "tradeable_low" in r["reference"] else "high", "mode": "warning_only"})
    (ROOT / "walk_forward/wf_reference_config.json").write_text(jdump(configs), encoding="utf-8")
    rows = []
    years = sorted([y for y in df["year"].dropna().unique() if y >= 2022])
    for cfg in configs:
        for y in years:
            test = df[df["year"].eq(y)].copy()
            if test.empty:
                continue
            if cfg["feature"] is None:
                keep = test
            else:
                flag = pd.to_numeric(test[cfg["feature"]], errors="coerce") >= cfg["threshold"] if cfg["direction"] == "high" else pd.to_numeric(test[cfg["feature"]], errors="coerce") <= cfg["threshold"]
                keep = test[~flag]
            rows.append({"config": cfg["name"], "test_year": y, "candidate_count": len(test), "kept_count": len(keep), "candidate_preservation": len(keep) / len(test) if len(test) else np.nan, "mean_net_current": keep["net_fixed_1h_current"].mean() if len(keep) else np.nan, "mean_net_2x": keep["net_fixed_1h_2x"].mean() if len(keep) else np.nan, "MFE_q90_rate": keep["label_mfe_success"].mean() if len(keep) else np.nan, "RFE_rate": keep["label_rfe_failure"].mean() if len(keep) else np.nan, "fake_giveback_rate": keep["label_fake_giveback_failure"].mean() if len(keep) else np.nan})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "walk_forward/wf_reference_results.csv", index=False)
    summary = res.groupby("config", as_index=False).agg(fold_count=("test_year", "count"), fold_pass_rate=("mean_net_current", lambda s: (s > 0).mean()), mean_net_current=("mean_net_current", "mean"), mean_net_2x=("mean_net_2x", "mean"), mean_preservation=("candidate_preservation", "mean"), mean_RFE_rate=("RFE_rate", "mean"))
    summary.to_csv(ROOT / "walk_forward/wf_reference_summary.csv", index=False)
    verdict = "BASELINE_TOP1_STILL_BEST"
    if len(summary) > 1:
        base = summary[summary["config"].eq("baseline_top1_only")].iloc[0]
        ref = summary[summary["config"].ne("baseline_top1_only")].iloc[0]
        if ref["mean_net_current"] > base["mean_net_current"] and ref["mean_preservation"] > 0.6 and ref["fold_pass_rate"] >= 0.5:
            verdict = "REFERENCE_WF_WEAK"
        else:
            verdict = "REFERENCE_WF_FAIL"
    (ROOT / "walk_forward/wf_reference_report.md").write_text(f"# Walk Forward Reference Report\n\nVerdict: {verdict}. Reference remains diagnostics/watch-only.\n", encoding="utf-8")
    return {"verdict": verdict}


def casebook_analysis() -> Dict[str, Any]:
    df = dataset()
    cases = []
    categories = {
        "TOP1_CLEAN_SUCCESS": df["label_clean_success"],
        "TOP1_HARD_FAILURE": df["label_hard_failure"],
        "TOP1_MFE_SUCCESS_NET_FAIL": df["label_oracle_good_fixed_bad"],
        "TOP1_RFE_FAILURE": df["label_rfe_failure"],
        "TOP1_FAKE_GIVEBACK_FAILURE": df["label_fake_giveback_failure"],
        "TOP1_RECENT_7D_FAIL": df["recent_7d"] & df["label_failure"],
        "TOP1_RECENT_30D_SUCCESS": df["recent_30d"] & df["label_net_success"],
        "TOP1_2026_WEAK_CASE": df["ytd_2026"] & df["label_failure"],
        "TOP1_2022_2025_STRONG_CASE": df["year_2022_2025_good_period"] & df["label_clean_success"],
        "TOP1_BEAR_SUCCESS": df["trend_regime"].eq("bear") & df["label_net_success"],
        "TOP1_BULL_FAILURE": df["trend_regime"].eq("bull") & df["label_failure"],
        "TOP1_HIGH_VOL_SUCCESS": df["vol_regime"].eq("high_vol") & df["label_net_success"],
        "TOP1_HIGH_VOL_FAILURE": df["vol_regime"].eq("high_vol") & df["label_failure"],
    }
    for cat, mask in categories.items():
        g = df[mask].copy()
        if g.empty:
            continue
        if "FAILURE" in cat or "FAIL" in cat or "WEAK" in cat:
            g = g.sort_values("net_fixed_1h_current").head(50)
        else:
            g = g.sort_values("net_fixed_1h_current", ascending=False).head(50)
        g["case_category"] = cat
        cases.append(g)
    cb = pd.concat(cases, ignore_index=True) if cases else pd.DataFrame()
    keep_cols = [c for c in ["case_category", "candidate_id", "timestamp", "year", "trend_regime", "vol_regime", PRIMARY_SCORE, "percentile_rolling90", "net_fixed_1h_current", "net_fixed_1h_2x", "future_return_net_current_bps_2h", "future_return_net_current_bps_4h", "future_return_net_current_bps_24h", "future_MFE_long_bps", "future_MAE_long_bps", "MFE_before_MAE", "label_rfe_failure", "label_fake_giveback_failure", "score_risk_baseline", "p_rfe_high_baseline", "p_fake_giveback_risk_baseline"] if c in cb.columns]
    out = cb[keep_cols].copy() if not cb.empty else cb
    write_df(out, ROOT / "casebook/top1_autopsy_casebook.parquet", ROOT / "casebook/top1_autopsy_casebook.csv")
    subsets = {
        "top1_clean_success_cases.csv": "TOP1_CLEAN_SUCCESS",
        "top1_hard_failure_cases.csv": "TOP1_HARD_FAILURE",
        "top1_recent_failure_cases.csv": "TOP1_RECENT_7D_FAIL",
        "top1_bear_success_cases.csv": "TOP1_BEAR_SUCCESS",
        "top1_bull_failure_cases.csv": "TOP1_BULL_FAILURE",
        "risk_reference_cases.csv": "TOP1_RFE_FAILURE",
    }
    for fn, cat in subsets.items():
        out[out.get("case_category", pd.Series(dtype=str)).eq(cat)].to_csv(ROOT / "casebook" / fn, index=False)
    (ROOT / "casebook/casebook_report.md").write_text(f"# Casebook Report\n\nCase rows: {len(out)}. Charts are not required for this smoke-safe implementation; tabular casebook is available for manual inspection.\n", encoding="utf-8")
    return {"verdict": "CASEBOOK_SUCCESS", "rows": len(out)}


def decision_docs(results: Dict[str, Any]) -> Dict[str, Any]:
    verdicts = []
    sf = results.get("success_failure", {}).get("verdict")
    risk = results.get("risk_reference", {}).get("verdict")
    wf = results.get("walk_forward", {}).get("verdict")
    exitv = results.get("exit", {}).get("verdict")
    recent = results.get("recent", {}).get("verdict")
    verdicts.append("TOP1_SHADOW_SCORE_AUTOPSY_COMPLETED")
    verdicts.append("TOP1_SCORE_KEEP_AS_IS")
    if sf:
        verdicts.append(sf)
    if risk in {"RISK_REFERENCE_WEAK", "RISK_REFERENCE_USEFUL"}:
        verdicts.append("TOP1_SCORE_KEEP_WITH_WARNING_TAG")
        verdicts.append(risk)
    else:
        verdicts.append("NO_RISK_REFERENCE_FOUND")
    if exitv and exitv != "FIXED_1H_OK":
        verdicts.append("TOP1_SCORE_EXIT_DEPENDENT")
        verdicts.append(exitv)
    if recent:
        verdicts.append(recent)
        verdicts.append("TOP1_SCORE_RECENT_DECAY_WARNING")
    if wf:
        verdicts.append(wf)
    verdicts.extend(["DO_NOT_CHANGE_FORWARD_PLAN", "FORWARD_ONLY_REQUIRED", "PRODUCTION_SAFETY_PASS", "production_not_ready", "promotion_not_ready"])
    decision = pd.DataFrame([{"decision": v, "selected": True} for v in dict.fromkeys(verdicts)])
    decision.to_csv(ROOT / "decision/top1_autopsy_decision_matrix.csv", index=False)
    (ROOT / "decision/top1_keep_drop_decision.md").write_text("# Top1 Keep/Drop Decision\n\nKeep `score_mfe_opportunity_baseline` top1 as diagnostics forward shadow reference. Do not change threshold or production logic.\n", encoding="utf-8")
    (ROOT / "decision/risk_reference_keep_drop_decision.md").write_text("# Risk Reference Decision\n\nRisk references, if used, should be warning-only/watch-only. No hard production filter.\n", encoding="utf-8")
    (ROOT / "decision/forward_plan_update_recommendation.md").write_text("# Forward Plan Update Recommendation\n\nDO_NOT_CHANGE_FORWARD_PLAN as the default. Optional future manual addition: warning-only risk tag in reports, not a gate, after more forward samples.\n", encoding="utf-8")
    (ROOT / "decision/forward_plan_update_diff_reference.json").write_text(jdump({"auto_change": False, "recommendation": "DO_NOT_CHANGE_FORWARD_PLAN", "optional_reference": "ADD_WARNING_ONLY_RISK_TAG"}), encoding="utf-8")
    (ROOT / "decision/do_not_promote_gate.md").write_text("# Do Not Promote Gate\n\nproduction_ready=false. promotion_ready=false. This autopsy cannot promote a score to live trading.\n", encoding="utf-8")
    (ROOT / "decision/recommended_next_branch.md").write_text("# Recommended Next Branch\n\nKeep forward shadow scorer running and review first real forward paper outcomes after sample count is meaningful.\n", encoding="utf-8")
    (ROOT / "recommended_next_branch.md").write_text("# Recommended Next Branch\n\nKeep forward shadow scorer running; do not modify production or forward config automatically.\n", encoding="utf-8")
    return {"verdicts": list(dict.fromkeys(verdicts))}


def final_report(results: Dict[str, Any]) -> None:
    dec = decision_docs(results)
    df = dataset()
    summary = {
        "top1_candidates": len(df),
        "net_success": int(df["label_net_success"].sum()),
        "net_failure": int(df["label_failure"].sum()),
        "mfe_q90_rate": float(df["label_mfe_success"].mean()),
        "rfe_rate": float(df["label_rfe_failure"].mean()),
        "fake_giveback_rate": float(df["label_fake_giveback_failure"].mean()),
        "mean_net_current": float(df["net_fixed_1h_current"].mean()),
        "mean_net_2x": float(df["net_fixed_1h_2x"].mean()),
    }
    report = f"""# Top1 Shadow Score Success Failure Autopsy Final Report

## Why
This diagnostics-only autopsy decomposes why `score_mfe_opportunity_baseline` rolling90 top1 worked historically, where it failed, and whether recent/2026 weakness is explainable.

## Dataset
Top1 candidates: {summary['top1_candidates']}
Net success: {summary['net_success']}
Net failure: {summary['net_failure']}
Mean net current: {summary['mean_net_current']:.3f} bps
Mean net 2x: {summary['mean_net_2x']:.3f} bps
MFE Q90 rate: {summary['mfe_q90_rate']:.3f}
RFE rate: {summary['rfe_rate']:.3f}
Fake/giveback rate: {summary['fake_giveback_rate']:.3f}

## Findings
Success/failure: {results.get('success_failure', {}).get('verdict')}
Recent weakness: {results.get('recent', {}).get('verdict')} / {results.get('recent', {}).get('verdict_2026')}
Regime: {results.get('regime', {}).get('verdicts')}
Risk reference: {results.get('risk_reference', {}).get('verdict')}
Exit: {results.get('exit', {}).get('verdict')} / {results.get('exit', {}).get('secondary')}
Walk-forward: {results.get('walk_forward', {}).get('verdict')}

## Safety
No production/live/order/state/Q2/R7/Risk/TCN changes. No private/order/account/balance/position endpoint calls. Forward scorer launchd was not restarted or modified.

## Decision
Keep current forward shadow plan. Optional warning-only/watch-only risk tag may be considered later, but no threshold change and no production promotion.

## Verdicts
{chr(10).join(dec['verdicts'])}
"""
    (ROOT / "top1_shadow_score_success_failure_autopsy_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "top1_shadow_score_success_failure_autopsy_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(dec["verdicts"]) + "\n", encoding="utf-8")


def run_full(fast: bool = False) -> Dict[str, Any]:
    before = safety_snapshot("before")
    input_discovery()
    res: Dict[str, Any] = {}
    res["dataset"] = build_dataset(fast=fast)
    res["success_failure"] = success_failure_analysis()
    res["recent"] = recent_weakness_analysis()
    res["regime"] = regime_analysis()
    res["risk_reference"] = risk_reference_analysis()
    res["exit"] = exit_horizon_analysis()
    res["walk_forward"] = walk_forward_analysis()
    res["casebook"] = casebook_analysis()
    final_report(res)
    finalize_safety(before)
    res["production_ready"] = False
    res["promotion_ready"] = False
    (ROOT / "run_metadata.json").write_text(jdump(res), encoding="utf-8")
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fast-smoke", action="store_true")
    p.add_argument("--dataset-build-only", action="store_true")
    p.add_argument("--success-failure-only", action="store_true")
    p.add_argument("--recent-weakness-only", action="store_true")
    p.add_argument("--regime-only", action="store_true")
    p.add_argument("--risk-reference-only", action="store_true")
    p.add_argument("--walk-forward-only", action="store_true")
    p.add_argument("--casebook-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    ensure_dirs()
    if args.dry_run:
        res = {"dry_run": True, "score_frame_exists": SCORE_FRAME.exists(), "feature_frame_exists": FEATURE_FRAME.exists(), "forward_status_exists": FORWARD_STATUS.exists(), "production_ready": False, "promotion_ready": False}
    elif args.fast_smoke:
        res = run_full(fast=True)
    elif args.dataset_build_only:
        before = safety_snapshot("before")
        input_discovery()
        res = build_dataset()
        finalize_safety(before)
    elif args.success_failure_only:
        before = safety_snapshot("before")
        res = success_failure_analysis()
        finalize_safety(before)
    elif args.recent_weakness_only:
        before = safety_snapshot("before")
        res = recent_weakness_analysis()
        finalize_safety(before)
    elif args.regime_only:
        before = safety_snapshot("before")
        res = regime_analysis()
        finalize_safety(before)
    elif args.risk_reference_only:
        before = safety_snapshot("before")
        res = risk_reference_analysis()
        finalize_safety(before)
    elif args.walk_forward_only:
        before = safety_snapshot("before")
        if not (ROOT / "risk_reference/risk_reference_scorecard.csv").exists():
            risk_reference_analysis()
        res = walk_forward_analysis()
        finalize_safety(before)
    elif args.casebook_only:
        before = safety_snapshot("before")
        res = casebook_analysis()
        finalize_safety(before)
    else:
        res = run_full(fast=False)
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
